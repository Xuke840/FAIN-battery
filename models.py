from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
from physics import CFG, ocp_n_np, ocp_p_np, ocp_n_torch, ocp_p_torch, exchange_current_torch, faradaic_torch, torch_current_profile, particle_flux_factor

def current_drive(t):
    if abs(CFG.I_app) < 1e-14:
        return torch.zeros_like(t)
    return torch_current_profile(t) / CFG.I_app

def integrated_current_torch(t):
    tr = float(CFG.current_ramp_time)
    I = float(CFG.I_app)
    if tr <= 0:
        return I * t
    tpos = torch.clamp(t, min=0.0)
    early = I * tpos * tpos / (2.0 * tr)
    late = I * (tpos - 0.5 * tr)
    return torch.where(tpos <= tr, early, late)

def halfspace_current_convolution(t):
    tr = float(CFG.current_ramp_time)
    I = float(CFG.I_app)
    eps = torch.finfo(t.dtype).eps
    tp = torch.clamp(t, min=0.0)
    if tr <= 0:
        return 2.0 * I * torch.sqrt(torch.clamp(tp, min=eps))
    sqt = torch.sqrt(torch.clamp(tp, min=eps))
    early = I * (4.0 / 3.0) * tp * sqt / tr
    u = torch.clamp(tp - tr, min=0.0)
    squ = torch.sqrt(u)
    first = I / tr * (2.0 * tp * (sqt - squ) - 2.0 / 3.0 * (tp * sqt - u * squ))
    late = first + 2.0 * I * squ
    out = torch.where(tp <= tr, early, late)
    return torch.where(tp <= 0.0, torch.zeros_like(out), out)

def a_priori_potential_scale():
    I = abs(float(CFG.I_app))
    kappa = max(min(CFG.kappa_n, CFG.kappa_s, CFG.kappa_p), 1e-12)
    sigma = max(min(CFG.sigma_n, CFG.sigma_p), 1e-12)
    ohm = I * CFG.Lx / kappa + I * CFG.Lx / sigma
    return max(ohm, 0.1)

class ResidualBlock(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.act = nn.Tanh()
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(x + self.fc2(self.act(self.fc1(x))))

class SharedBackbone(nn.Module):

    def __init__(self, width=64, depth=4):
        super().__init__()
        self.inp = nn.Linear(4, width)
        self.act = nn.Tanh()
        self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(depth)])

    def forward(self, z):
        tt = z[:, 2:3]
        drive = current_drive(tt)
        q = torch.cat([2 * z[:, 0:1] / CFG.Lx - 1, 2 * z[:, 1:2] / CFG.Ly - 1, 2 * tt / CFG.t_final - 1, 2 * drive - 1], 1)
        h = self.act(self.inp(q))
        for b in self.blocks:
            h = b(h)
        return h

class FieldInteractionDecoder(nn.Module):

    def __init__(self, width, n_fields=3):
        super().__init__()
        self.n_fields = n_fields
        self.tokens = nn.Parameter(torch.zeros(n_fields, width))
        nn.init.normal_(self.tokens, std=0.02)
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.ff = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)
        self.heads = nn.ModuleList([nn.Linear(width, 1) for _ in range(n_fields)])
        self.scale = width ** (-0.5)
        for h in self.heads:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    def forward(self, h, alpha=1.0):
        tok = h[:, None, :] + self.tokens[None, :, :]
        q = self.q(tok)
        k = self.k(tok)
        v = self.v(tok)
        w = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)
        tok = self.norm(tok + alpha * self.proj(torch.matmul(w, v)) + self.ff(tok))
        return [self.heads[i](tok[:, i, :]) for i in range(self.n_fields)]

class SolidPotentialCorrection(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.offset = nn.Linear(width, 1)
        self.enddiff = nn.Linear(width, 1)
        self.local = nn.Linear(width, 1)
        for h in (self.offset, self.enddiff, self.local):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    def forward(self, h, hmid, xi):
        H = 3.0 * xi * xi - 2.0 * xi * xi * xi
        B = 16.0 * xi * xi * (1.0 - xi) * (1.0 - xi)
        return torch.tanh(self.offset(hmid)) + H * torch.tanh(self.enddiff(hmid)) + B * torch.tanh(self.local(h))

class ParticleMomentNet(nn.Module):

    def __init__(self, k, width=48, depth=3, n_modes=3, n_spatial_modes=6):
        super().__init__()
        self.k = k
        self.n_modes = n_modes
        self.n_spatial_modes = n_spatial_modes
        self.inp = nn.Linear(4, width)
        self.act = nn.Tanh()
        self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(depth)])
        self.avg_out = nn.Linear(width, n_spatial_modes)
        self.surf_out = nn.Linear(width, n_spatial_modes)
        self.mode_out = nn.Linear(width, n_modes)
        for h in (self.avg_out, self.surf_out, self.mode_out):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
        self.x0 = 0.0 if k == 'n' else CFG.x_s_end
        self.x1 = CFG.x_n_end if k == 'n' else CFG.Lx
        self.theta0 = CFG.theta_n0 if k == 'n' else CFG.theta_p0

    def trunk(self, z):
        tt = z[:, 2:3]
        drive = current_drive(tt)
        q = torch.cat([2 * (z[:, 0:1] - self.x0) / (self.x1 - self.x0) - 1, 2 * z[:, 1:2] / CFG.Ly - 1, 2 * tt / CFG.t_final - 1, 2 * drive - 1], 1)
        h = self.act(self.inp(q))
        for b in self.blocks:
            h = b(h)
        return h

    def time_coeffs(self, z):
        zc = torch.cat([torch.zeros_like(z[:, 0:1]) + 0.5 * (self.x0 + self.x1), torch.zeros_like(z[:, 1:2]) + 0.5 * CFG.Ly, z[:, 2:3]], 1)
        h = self.trunk(zc)
        return (self.avg_out(h), self.surf_out(h))

    def spatial_basis(self, z):
        xi = (z[:, 0:1] - self.x0) / (self.x1 - self.x0)
        yy = z[:, 1:2] / CFG.Ly
        p = math.pi
        b1 = torch.cos(p * xi)
        b2 = torch.cos(2 * p * xi)
        b3 = torch.cos(p * yy)
        b4 = torch.cos(2 * p * yy)
        b5 = b1 * b3
        b6 = b2 * b3
        return torch.cat([b1, b2, b3, b4, b5, b6], 1)

    def corrections(self, z):
        ca, cs = self.time_coeffs(z)
        B = self.spatial_basis(z)
        ravg = torch.mean(torch.tanh(ca) * B, dim=1, keepdim=True)
        rsurf = torch.mean(torch.tanh(cs) * B, dim=1, keepdim=True)
        return (ravg, rsurf)

class FAIN(nn.Module):

    def __init__(self, width=64, depth=4, particle_width=48, particle_depth=3, reaction_modes=0):
        super().__init__()
        self.width = width
        self.depth = depth
        self.particle_width = particle_width
        self.particle_depth = particle_depth
        self.backbone = SharedBackbone(width, depth)
        self.dec_shared = FieldInteractionDecoder(width, 3)
        self.dec_n = FieldInteractionDecoder(width, 3)
        self.dec_s = FieldInteractionDecoder(width, 3)
        self.dec_p = FieldInteractionDecoder(width, 3)
        self.solid_n = SolidPotentialCorrection(width)
        self.solid_p = SolidPotentialCorrection(width)
        self.particle_n = ParticleMomentNet('n', particle_width, particle_depth, 3)
        self.particle_p = ParticleMomentNet('p', particle_width, particle_depth, 3)
        self.alpha = 0.0
        self.phi_scale = float(a_priori_potential_scale())
        self.psi_correction_fraction = 0.01
        self.solid_correction_fraction = 0.25
        self.particle_avg_correction_fraction = 0.05
        self.particle_surface_correction_fraction = 0.08
        self.Un0 = float(ocp_n_np(CFG.theta_n0))
        self.Up0 = float(ocp_p_np(CFG.theta_p0))

    def _region_coords(self, z, k):
        if k == 'n':
            x0, x1 = (0.0, CFG.x_n_end)
        elif k == 's':
            x0, x1 = (CFG.x_n_end, CFG.x_s_end)
        elif k == 'p':
            x0, x1 = (CFG.x_s_end, CFG.Lx)
        else:
            raise ValueError(k)
        xi = (z[:, 0:1] - x0) / (x1 - x0)
        bubble = 4.0 * xi * (1.0 - xi)
        return (x0, x1, xi, bubble)

    def _raw_common(self, z, k):
        h = self.backbone(z)
        shared = self.dec_shared(h, self.alpha)
        reg = self.dec_n(h, self.alpha) if k == 'n' else self.dec_s(h, self.alpha) if k == 's' else self.dec_p(h, self.alpha)
        _, _, _, b = self._region_coords(z, k)
        return ([shared[i] + b * reg[i] for i in range(3)], h)

    def _psi_base(self, x, t, k):
        I = torch_current_profile(t)
        Ln = CFG.x_n_end
        Ls = CFG.x_s_end - CFG.x_n_end
        Lp = CFG.Lx - CFG.x_s_end
        if k == 'n':
            return I * x * x / (2.0 * CFG.kappa_n * Ln)
        if k == 's':
            q = x - Ln
            return I * Ln / (2.0 * CFG.kappa_n) + I * q / CFG.kappa_s
        if k == 'p':
            q = x - CFG.x_s_end
            return I * Ln / (2.0 * CFG.kappa_n) + I * Ls / CFG.kappa_s + I * (q - q * q / (2.0 * Lp)) / CFG.kappa_p
        raise ValueError(k)

    def _avg_psi_base(self, t, k):
        I = torch_current_profile(t)
        Ln = CFG.x_n_end
        Ls = CFG.x_s_end - CFG.x_n_end
        Lp = CFG.Lx - CFG.x_s_end
        if k == 'n':
            return I * Ln / (6.0 * CFG.kappa_n)
        if k == 'p':
            return I * Ln / (2.0 * CFG.kappa_n) + I * Ls / CFG.kappa_s + I * Lp / (3.0 * CFG.kappa_p)
        raise ValueError(k)

    def _particle_baselines(self, t, k):
        if k == 'n':
            theta0, Cs, L, Ds, sgn = (CFG.theta_n0, CFG.Csolid_n, CFG.x_n_end, CFG.D_s_n, -1.0)
        else:
            theta0, Cs, L, Ds, sgn = (CFG.theta_p0, CFG.Csolid_p, CFG.Lx - CFG.x_s_end, CFG.D_s_p, 1.0)
        integ = integrated_current_torch(t)
        theta_bar = theta0 - sgn * CFG.gamma_li * integ / (Cs * L)
        chi = particle_flux_factor(k)
        conv = halfspace_current_convolution(t)
        theta_s = theta0 - chi * sgn / L * conv / math.sqrt(math.pi * Ds)
        return (theta_bar, theta_s)

    @staticmethod
    def _safe_logit(x):
        x = torch.clamp(x, 1e-05, 1.0 - 1e-05)
        return torch.log(x / (1.0 - x))

    def particle_moments(self, z, k):
        net = self.particle_n if k == 'n' else self.particle_p
        avg0, surf0 = self._particle_baselines(z[:, 2:3], k)
        ravg, rsurf = net.corrections(z)
        gate = torch.clamp(current_drive(z[:, 2:3]) + z[:, 2:3] / CFG.t_final, min=0.0, max=1.0)
        avg = avg0 + gate * self.particle_avg_correction_fraction * avg0 * (1.0 - avg0) * ravg
        surf = surf0 + gate * self.particle_surface_correction_fraction * surf0 * (1.0 - surf0) * rsurf
        return (avg, surf)

    def _common_fields(self, z, k):
        raw, h = self._raw_common(z, k)
        rce, rpsi, rT = raw
        tt = z[:, 2:3]
        tau = tt / CFG.t_final
        drive = current_drive(tt)
        gate = torch.clamp(drive + tau, min=0.0, max=1.0)
        ce = CFG.c_e0 * torch.exp(tau * torch.tanh(rce))
        T = CFG.T0 * torch.exp(0.5 * tau * torch.tanh(rT))
        psi = self._psi_base(z[:, 0:1], tt, k) + self.phi_scale * self.psi_correction_fraction * gate * torch.tanh(rpsi)
        phi_e = CFG.chi_ln_ce * torch.log(torch.clamp(ce / CFG.c_e0, min=1e-12)) + psi
        return ({'ce': ce, 'T': T, 'psi_e': psi, 'phi_e': phi_e}, h)

    def _solid_base(self, z, k, theta_s_base):
        tt = z[:, 2:3]
        I = torch_current_profile(tt)
        if k == 'n':
            x0, L, sigma, a, k0, sgn = (0.0, CFG.x_n_end, CFG.sigma_n, CFG.a_n, CFG.k0_n, -1.0)
            U = ocp_n_torch(theta_s_base)
        else:
            x0, L, sigma, a, k0, sgn = (CFG.x_s_end, CFG.Lx - CFG.x_s_end, CFG.sigma_p, CFG.a_p, CFG.k0_p, 1.0)
            U = ocp_p_torch(theta_s_base)
        xi = (z[:, 0:1] - x0) / L
        if k == 'n':
            shape = I * L / sigma * (xi - 0.5 * xi * xi)
            mean_shape = I * L / (3.0 * sigma)
        else:
            shape = I * L / sigma * (0.5 * xi * xi)
            mean_shape = I * L / (6.0 * sigma)
        i0 = k0 * torch.sqrt(torch.clamp(CFG.c_e0 * theta_s_base * (1.0 - theta_s_base), min=1e-12)) * torch.exp(torch.zeros_like(tt))
        jbar = sgn * I / (a * L)
        eta = CFG.T0 / CFG.beta_bv * torch.asinh(jbar / (2.0 * torch.clamp(i0, min=1e-12)))
        c = self._avg_psi_base(tt, k) + U + eta - mean_shape
        return (c + shape, xi)

    def _solid_potential(self, z, k, h, theta_s_base):
        base, xi = self._solid_base(z, k, theta_s_base)
        xmid = 0.5 * CFG.x_n_end if k == 'n' else 0.5 * (CFG.x_s_end + CFG.Lx)
        zmid = torch.cat([torch.zeros_like(z[:, 0:1]) + xmid, z[:, 1:2], z[:, 2:3]], 1)
        hmid = self.backbone(zmid)
        corr = (self.solid_n if k == 'n' else self.solid_p)(h, hmid, xi)
        gate = torch.clamp(current_drive(z[:, 2:3]) + z[:, 2:3] / CFG.t_final, min=0.0, max=1.0)
        return base + self.phi_scale * self.solid_correction_fraction * gate * corr

    def region(self, z, k):
        common, h = self._common_fields(z, k)
        out = dict(common)
        if k in ('n', 'p'):
            theta_avg, theta_surf = self.particle_moments(z, k)
            _, theta_s_base = self._particle_baselines(z[:, 2:3], k)
            phi_s = self._solid_potential(z, k, h, theta_s_base)
            j, eta, i0, U, arg = faradaic_torch(k, common['ce'], theta_surf, phi_s, common['phi_e'], common['T'])
            zr = torch.cat([z[:, 0:2], torch.ones_like(z[:, 0:1]), z[:, 2:3]], 1)
            out.update({'phi_s': phi_s, 'theta_avg': theta_avg, 'theta_surf': theta_surf, 'zr_surf': zr, 'j': j, 'eta': eta, 'i0': i0, 'U': U, 'bv_arg': arg})
        return out

    @staticmethod
    def _boundary_layer_basis(r, lam):
        return torch.cosh(lam * r) / math.cosh(lam)

    @staticmethod
    def _boundary_layer_mean(lam):
        th = math.tanh(lam)
        return 3.0 * (th / lam - 2.0 / (lam * lam) + 2.0 * th / lam ** 3)

    def particle(self, zr, k, j=None):
        z = torch.cat([zr[:, 0:2], zr[:, 3:4]], 1)
        avg, surf = self.particle_moments(z, k)
        if j is None:
            common, h = self._common_fields(z, k)
            _, theta_s_base = self._particle_baselines(z[:, 2:3], k)
            phi_s = self._solid_potential(z, k, h, theta_s_base)
            j = faradaic_torch(k, common['ce'], surf, phi_s, common['phi_e'], common['T'])[0]
        Ds = CFG.D_s_n if k == 'n' else CFG.D_s_p
        chi = particle_flux_factor(k)
        d = -chi * j / Ds
        lam1 = 1.0 / math.sqrt(max(Ds * CFG.t_final, 1e-12))
        lam2 = 2.0 * lam1
        m1 = self._boundary_layer_mean(lam1)
        m2 = self._boundary_layer_mean(lam2)
        q1 = lam1 * math.tanh(lam1)
        q2 = lam2 * math.tanh(lam2)
        a11 = 1.0 - m1
        a12 = 1.0 - m2
        det = a11 * q2 - a12 * q1
        rhs1 = surf - avg
        B = (rhs1 * q2 - a12 * d) / det
        C = (a11 * d - rhs1 * q1) / det
        A = surf - B - C
        r = zr[:, 2:3]
        b1 = self._boundary_layer_basis(r, lam1)
        b2 = self._boundary_layer_basis(r, lam2)
        return A + B * b1 + C * b2

    def architecture_dict(self):
        return {'name': 'FAIN', 'full_name': 'Field-Aware Interaction Network', 'version': 'v0.13-particle-boundary-layer', 'physics_protocol_feature': True, 'electrochemical_potential_representation': True, 'analytic_ionic_current_baseline': True, 'bounded_psi_correction': True, 'psi_correction_fraction': self.psi_correction_fraction, 'primary_solid_potential': True, 'BV_derived_reaction': True, 'hard_galvanostatic_reaction': False, 'hard_BV_closure': True, 'hard_solid_current_BCs': True, 'hard_interface_traces': True, 'particle_moment_representation': True, 'hard_particle_radial_BCs': True, 'diffusion_boundary_layer_particle_basis': True, 'baseline_preserving_particle_moments': True, 'zero_spatial_mean_particle_corrections': True, 'particle_avg_correction_fraction': self.particle_avg_correction_fraction, 'particle_surface_correction_fraction': self.particle_surface_correction_fraction, 'zero_moment_radial_modes': 0, 'reference_tuned_output_envelopes': False, 'width': self.width, 'depth': self.depth, 'particle_width': self.particle_width, 'particle_depth': self.particle_depth, 'parameters': sum((p.numel() for p in self.parameters()))}

class SIRENDecoder(nn.Module):

    def __init__(self, width: int, n_fields: int=3):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(width, 1) for _ in range(n_fields)])
        for h in self.heads:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    def forward(self, h, alpha=1.0):
        return [q(h) for q in self.heads]

class SineLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int, omega: float, first: bool=False):
        super().__init__()
        self.omega = float(omega)
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if first:
                bound = 1.0 / max(in_features, 1)
            else:
                bound = math.sqrt(6.0 / max(in_features, 1)) / self.omega
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))

class SIRENBackbone(nn.Module):

    def __init__(self, width: int=96, depth: int=4, first_omega: float=10.0, hidden_omega: float=1.0):
        super().__init__()
        layers = [SineLayer(4, width, first_omega, first=True)]
        for _ in range(max(depth - 1, 0)):
            layers.append(SineLayer(width, width, hidden_omega, first=False))
        self.layers = nn.ModuleList(layers)

    def forward(self, z):
        tt = z[:, 2:3]
        drive = current_drive(tt)
        h = torch.cat([2 * z[:, 0:1] / CFG.Lx - 1, 2 * z[:, 1:2] / CFG.Ly - 1, 2 * tt / CFG.t_final - 1, 2 * drive - 1], dim=1)
        for layer in self.layers:
            h = layer(h)
        return h

class TanhBackbone(nn.Module):

    def __init__(self, width: int=96, depth: int=4):
        super().__init__()
        self.inp = nn.Linear(4, width)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(max(depth - 1, 0))])
        self.act = nn.Tanh()

    def forward(self, z):
        tt = z[:, 2:3]
        drive = current_drive(tt)
        q = torch.cat([2 * z[:, 0:1] / CFG.Lx - 1, 2 * z[:, 1:2] / CFG.Ly - 1, 2 * tt / CFG.t_final - 1, 2 * drive - 1], dim=1)
        h = self.act(self.inp(q))
        for fc in self.hidden:
            h = self.act(fc(h))
        return h

class SIRENPINN(FAIN):

    def __init__(self, width=96, depth=4, particle_width=64, particle_depth=3):
        super().__init__(width, depth, particle_width, particle_depth)
        self.backbone = SIRENBackbone(width, depth)
        self.dec_shared = SIRENDecoder(width, 3)
        self.dec_n = SIRENDecoder(width, 3)
        self.dec_s = SIRENDecoder(width, 3)
        self.dec_p = SIRENDecoder(width, 3)
        self.alpha = 1.0

    def architecture_dict(self):
        d = super().architecture_dict()
        d.update({'name': 'SIRENPINN', 'full_name': 'SIREN physics-informed baseline', 'version': 'v0.13-baseline-siren-pinn', 'parameters': sum((p.numel() for p in self.parameters())), 'field_aware_interaction': False, 'residual_backbone': False, 'attention_decoder': False, 'siren_backbone': True, 'siren_first_omega': 10.0, 'siren_hidden_omega': 1.0, 'xpinn_domain_decomposition': False, 'same_v013_physics_representation': True, 'baseline_role': 'baseline'})
        return d

class XPINNPINN(FAIN):

    def __init__(self, width=96, depth=4, particle_width=64, particle_depth=3):
        super().__init__(width, depth, particle_width, particle_depth)
        del self.backbone
        del self.dec_shared
        del self.dec_n
        del self.dec_s
        del self.dec_p
        self.back_n = TanhBackbone(width, depth)
        self.back_s = TanhBackbone(width, depth)
        self.back_p = TanhBackbone(width, depth)
        self.head_n = nn.Linear(width, 3)
        self.head_s = nn.Linear(width, 3)
        self.head_p = nn.Linear(width, 3)
        for h in (self.head_n, self.head_s, self.head_p):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
        self.alpha = 1.0

    def _back(self, z, k):
        if k == 'n':
            return self.back_n(z)
        if k == 's':
            return self.back_s(z)
        if k == 'p':
            return self.back_p(z)
        raise ValueError(k)

    def _head(self, h, k):
        if k == 'n':
            return self.head_n(h)
        if k == 's':
            return self.head_s(h)
        if k == 'p':
            return self.head_p(h)
        raise ValueError(k)

    def _raw_common(self, z, k):
        h = self._back(z, k)
        m = self._head(h, k)
        return ([m[:, 0:1], m[:, 1:2], m[:, 2:3]], h)

    def _solid_potential(self, z, k, h, theta_s_base):
        base, xi = self._solid_base(z, k, theta_s_base)
        xmid = 0.5 * CFG.x_n_end if k == 'n' else 0.5 * (CFG.x_s_end + CFG.Lx)
        zmid = torch.cat([torch.zeros_like(z[:, 0:1]) + xmid, z[:, 1:2], z[:, 2:3]], 1)
        hmid = self._back(zmid, k)
        corr = (self.solid_n if k == 'n' else self.solid_p)(h, hmid, xi)
        gate = torch.clamp(current_drive(z[:, 2:3]) + z[:, 2:3] / CFG.t_final, min=0.0, max=1.0)
        return base + self.phi_scale * self.solid_correction_fraction * gate * corr

    def architecture_dict(self):
        d = super().architecture_dict()
        d.update({'name': 'XPINNPINN', 'full_name': 'Three-subdomain XPINN physics-informed baseline', 'version': 'v0.13-baseline-xpinn', 'parameters': sum((p.numel() for p in self.parameters())), 'field_aware_interaction': False, 'residual_backbone': False, 'attention_decoder': False, 'siren_backbone': False, 'xpinn_domain_decomposition': True, 'hard_interface_traces': False, 'explicit_interface_value_loss': True, 'same_v013_physics_representation': True, 'baseline_role': 'baseline'})
        return d

def build_extra_baseline(name: str, width=96, depth=4, particle_width=64, particle_depth=3):
    key = name.lower().replace('-', '_')
    if key in {'siren', 'siren_pinn'}:
        return SIRENPINN(width, depth, particle_width, particle_depth)
    if key in {'xpinn', 'xpinn_pinn'}:
        return XPINNPINN(width, depth, particle_width, particle_depth)
    raise ValueError(f'Unknown baseline {name!r}; choose siren_pinn or xpinn.')

class PlainBackbone(nn.Module):

    def __init__(self, width=96, depth=4):
        super().__init__()
        self.inp = nn.Linear(4, width)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(max(depth - 1, 0))])
        self.act = nn.Tanh()

    def forward(self, z):
        tt = z[:, 2:3]
        drive = current_drive(tt)
        q = torch.cat([2 * z[:, 0:1] / CFG.Lx - 1, 2 * z[:, 1:2] / CFG.Ly - 1, 2 * tt / CFG.t_final - 1, 2 * drive - 1], dim=1)
        h = self.act(self.inp(q))
        for fc in self.hidden:
            h = self.act(fc(h))
        return h

class MLPDecoder(nn.Module):

    def __init__(self, width, n_fields=3):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, n_fields)
        self.act = nn.Tanh()
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h, alpha=1.0):
        y = self.fc2(self.act(self.fc1(h)))
        return [y[:, i:i + 1] for i in range(y.shape[1])]

class _PlainDecoderBaseline(FAIN):

    def _replace_macro(self, width, depth, residual_backbone=False):
        self.backbone = SharedBackbone(width, depth) if residual_backbone else PlainBackbone(width, depth)
        self.dec_shared = MLPDecoder(width, 3)
        self.dec_n = MLPDecoder(width, 3)
        self.dec_s = MLPDecoder(width, 3)
        self.dec_p = MLPDecoder(width, 3)
        self.alpha = 1.0

class PurePINN(_PlainDecoderBaseline):

    def __init__(self, width=96, depth=4, particle_width=64, particle_depth=3):
        super().__init__(width, depth, particle_width, particle_depth)
        self._replace_macro(width, depth, False)

    def architecture_dict(self):
        d = super().architecture_dict()
        d.update({'name': 'PurePINN', 'full_name': 'Plain-MLP physics-informed baseline', 'version': 'v0.13-baseline-pure-pinn', 'parameters': sum((p.numel() for p in self.parameters())), 'field_aware_interaction': False, 'residual_backbone': False, 'attention_decoder': False, 'same_v013_physics_representation': True, 'baseline_role': 'baseline'})
        return d

class ResPINN(_PlainDecoderBaseline):

    def __init__(self, width=96, depth=4, particle_width=64, particle_depth=3):
        super().__init__(width, depth, particle_width, particle_depth)
        self._replace_macro(width, depth, True)

    def architecture_dict(self):
        d = super().architecture_dict()
        d.update({'name': 'ResPINN', 'full_name': 'Residual-MLP baseline without field-aware interaction', 'version': 'v0.13-baseline-res-pinn', 'parameters': sum((p.numel() for p in self.parameters())), 'field_aware_interaction': False, 'residual_backbone': True, 'attention_decoder': False, 'same_v013_physics_representation': True, 'baseline_role': 'baseline'})
        return d

class ParameterMatchedMLP(PurePINN):

    def __init__(self, width=205, depth=4, particle_width=64, particle_depth=3):
        super().__init__(width, depth, particle_width, particle_depth)

    def architecture_dict(self):
        d = super().architecture_dict()
        d.update({'name': 'ParameterMatchedMLP', 'full_name': 'Parameter-matched plain-MLP physics-informed baseline', 'version': 'v0.13-baseline-parameter-matched-mlp', 'parameters': sum((p.numel() for p in self.parameters())), 'parameter_matched_to': 'FAIN-v0.13'})
        return d

def build_archived_baseline(name, width=None, depth=4, particle_width=64, particle_depth=3):
    key = name.lower().replace('-', '_')
    if key in {'pure', 'pure_pinn'}:
        return PurePINN(width or 96, depth, particle_width, particle_depth)
    if key in {'res', 'res_pinn'}:
        return ResPINN(width or 96, depth, particle_width, particle_depth)
    if key in {'param_mlp', 'parameter_matched_mlp', 'parameter_matched'}:
        return ParameterMatchedMLP(width or 205, depth, particle_width, particle_depth)
    raise ValueError(name)

def build_model(name, width=None, depth=4, particle_width=64, particle_depth=3):
    key = name.lower().replace('-', '_')
    if key in {'fain'}:
        return FAIN(width or 96, depth, particle_width, particle_depth)
    if key in {'pure', 'pure_pinn'}:
        return PurePINN(width or 96, depth, particle_width, particle_depth)
    if key in {'res', 'res_pinn'}:
        return ResPINN(width or 96, depth, particle_width, particle_depth)
    if key in {'param_mlp', 'parameter_matched_mlp', 'parameter_matched'}:
        return ParameterMatchedMLP(width or 205, depth, particle_width, particle_depth)
    if key in {'siren', 'siren_pinn'}:
        return SIRENPINN(width or 96, depth, particle_width, particle_depth)
    if key in {'xpinn', 'xpinn_pinn'}:
        return XPINNPINN(width or 96, depth, particle_width, particle_depth)
    raise ValueError(name)