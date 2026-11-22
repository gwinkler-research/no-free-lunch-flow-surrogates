# =====================================================================
# Linear DMDc baseline (least-squares A, B) on the same frozen KVS
# ConvAE latent that the nonlinear AR rows use. Feeds the "dmdc" rows (linear
# floor) of the KVS results table. The adapter itself handles KVS only;
# scripts/cmp3d_dmdc_gate2.py reuses _dmdc_solve and _linear_rollout on the CMP
# 3D-POD latent.
#
# Latent and control normalisation use TRAIN statistics only (see fit()). The
# operator is made Schur-stable (spectral radius <= 1) before the rollout. A
# stable linear map has no limit cycle, so this row cannot reproduce the
# self-excited vortex shedding of the KVS.
# =====================================================================
"""dmdc adapter: linear DMDc (Dynamic Mode Decomposition with control).

No torch training, no GPU, no epoch loop. The fit is one SVD-based
least-squares solve on the normalised TRAIN latents and controls:

    Z0  = stacked z_t              over all (case, t<T-1) pairs    (P, d)
    Zp  = stacked z_{t+1}          aligned with Z0                  (P, d)
    U   = stacked control u_t      aligned with Z0                  (P, c)
    Phi = [Z0 | U]                                                  (P, d+c)
    M, *_ = np.linalg.lstsq(Phi, Zp, rcond=rcond)   # SVD-based
    A = M[:d].T          (d, d)        z_{t+1} = A z_t + B u_t
    B = M[d:].T          (d, c)

i.e.  [A|B] = Zp.T @ pinv([Z0|U]).T  via SVD lstsq. rcond and cond([Z0;U]) are
logged and stored.

Stabilisation. If the unconstrained fit A_ls has spectral radius above the cap
(stab_rho_cap, default 1), it is replaced by a Schur-stable operator. B stays at the
joint lstsq value. cfg["model"]["stab_method"] selects how:

  * "constrained" (default): at best a local minimiser of the least-squares objective on
    the Schur-stable set,
        minimize  ||Z' - A Z - B U||_F     s.t.  spectral_radius(A) <= cap,
    found by projected gradient descent with a conditional-gradient step, started
    at the "clip" operator (_constrained_stable_A).
  * "clip": eigenvalue-magnitude clipping |lam| -> min(|lam|, cap) with the
    eigenvectors kept (_project_schur). The same projection is used inside
    "constrained".
  * "none": the raw A_ls.

If spectral_radius(A_ls) <= cap, A_ls is used unchanged by every method.

Stored in the checkpoint, meta and run_meta: spectral_radius_raw (before),
spectral_radius (after), the method, PGD iterations and final gradient-mapping norm,
n_eig_clipped, and the three least-squares residuals, which satisfy
ls_resid_raw <= ls_resid_constrained <= ls_resid_clip.

KVS path: encoder = latent_dynamics._load_frozen_encoder(cfg["encoder_ref"]), the same
bc32 ConvAE as the KVS AR rows; latents via latent_dynamics._encode_traj on the raster
frames (view.grid.frames_for); control u = _common.ramp_s_dsdt(theta[:,1:], t_grid) ->
(s, ds/dt). Prediction encodes only the t=0 frame, rolls the linear map forward, decodes
via the ConvAE and maps to the eval nodes with
_common.assemble_kvs_field_arrays (target_type "field"). The rollout is NaN/Inf-safe
(_Z_CLIP guard).

best.pt and last.pt hold the full model (A, B, normalisation stats, encoder_ref,
dataset) in results/benchmark/runs/<model_name>_seed<seed>_ckpts/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from benchmark.adapters import BenchmarkAdapter, register_adapter
from benchmark.adapters import _common
from benchmark.adapters._common import save_ckpt
# Same frozen ConvAE load and encode path as the KVS AR rows.
from benchmark.adapters.latent_dynamics import (_encode_traj,
                                                _load_frozen_encoder)
from tools.common.seeding import set_seed

REPO = Path(__file__).resolve().parents[2]
# Overflow guard: linear-rollout latents are clipped to this finite magnitude, so a
# divergent rollout stays visible in the metrics while the decode does not crash
# and the prediction_io NaN/Inf check passes. It does not modify A.
_Z_CLIP = 1e12


def _device(cfg_model: dict) -> torch.device:
    d = str(cfg_model.get("device", "cpu"))
    if d == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def _project_schur(A: np.ndarray, rho_cap: float = 1.0):
    """Approximate projection of A onto the Schur-stable set
    {spectral_radius <= rho_cap} by eigenvalue-magnitude clipping.

    A is a real, generally NON-NORMAL operator -> complex eigenpairs in
    conjugate pairs. Eigendecompose A = V diag(lam) V^-1, clip
    |lam_i| -> min(|lam_i|, rho_cap) keeping arg(lam_i) (so conjugate
    pairs stay conjugate), reconstruct A_proj = Re(V diag(lam') V^-1).
    Re(.) removes the O(eps) imaginary residual; exact when conjugate
    pairs are clipped identically (they are, since |lam|=|conj(lam)|).

    Fallback (V ill-conditioned / defective / non-finite recon /
    rho does not drop): uniformly scale A_proj = (rho_cap/rho_raw)*A
    (eigenvalues scale linearly -> rho exactly rho_cap; real, no
    eigendecomposition). More conservative but always defined. This is
    both the stab_method="clip" operator and the projection Proj() used
    by the constrained solver.

    Returns (A_proj (d,d) real float64, info) with rho_raw, rho_stab,
    n_eig_clipped, stab_method ("eig_clip" | "scaled_spectral_fallback"
    | "none"), cond_V.
    """
    A = np.asarray(A, dtype=np.float64)
    d = A.shape[0]
    ev_raw = np.linalg.eigvals(A)
    rho_raw = float(np.max(np.abs(ev_raw))) if ev_raw.size else 0.0
    tol = 1e-8
    # Nothing to do if already Schur-stable.
    if rho_raw <= rho_cap + tol:
        return A.copy(), {
            "rho_raw": rho_raw, "rho_stab": rho_raw,
            "n_eig_clipped": 0, "stab_method": "none",
            "cond_V": float("nan"),
        }
    info = {"rho_raw": rho_raw}
    try:
        lam, V = np.linalg.eig(A)
        cond_V = float(np.linalg.cond(V))
        mag = np.abs(lam)
        clip_mask = mag > rho_cap
        n_clipped = int(np.count_nonzero(clip_mask))
        scale = np.ones_like(mag)
        # cap magnitude, keep phase: lam' = lam * (rho_cap/|lam|)
        scale[clip_mask] = rho_cap / mag[clip_mask]
        lam_clipped = lam * scale
        A_rec = V @ np.diag(lam_clipped) @ np.linalg.inv(V)
        A_proj = np.real(A_rec).astype(np.float64)
        imag_resid = (float(np.max(np.abs(np.imag(A_rec))))
                      if A_rec.size else 0.0)
        rho_stab = (float(np.max(np.abs(np.linalg.eigvals(A_proj))))
                    if A_proj.size else 0.0)
        ok = (np.all(np.isfinite(A_proj))
              and rho_stab <= rho_cap + 1e-6
              and np.isfinite(cond_V) and cond_V < 1e12
              and imag_resid < 1e-3)
        if ok:
            info.update({
                "rho_stab": rho_stab, "n_eig_clipped": n_clipped,
                "stab_method": "eig_clip", "cond_V": cond_V,
                "imag_resid": imag_resid,
            })
            return A_proj, info
    except np.linalg.LinAlgError:
        cond_V = float("inf")
    # ---- robust fallback: scaled spectral projection -------------------
    alpha = rho_cap / rho_raw
    A_proj = (alpha * A).astype(np.float64)
    rho_stab = (float(np.max(np.abs(np.linalg.eigvals(A_proj))))
                if A_proj.size else 0.0)
    info.update({
        "rho_stab": rho_stab, "n_eig_clipped": int(d),
        "stab_method": "scaled_spectral_fallback",
        "cond_V": cond_V,
    })
    return A_proj, info


def _ls_residual(A: np.ndarray, B: np.ndarray, Z0: np.ndarray,
                 U0: np.ndarray, Zp: np.ndarray) -> float:
    """Frobenius LS residual ||Z0 A^T + U0 B^T - Zp||_F of the
    one-step DMDc objective (rows = pairs). B fixed at joint-lstsq."""
    R = Z0 @ A.T + U0 @ B.T - Zp
    return float(np.sqrt(np.sum(R * R)))


def _constrained_stable_A(A_ls: np.ndarray, B: np.ndarray,
                          Z0: np.ndarray, U0: np.ndarray, Zp: np.ndarray,
                          rho_cap: float = 1.0, delta: float = 1e-6,
                          max_iter: int = 500, tol: float = 1e-7,
                          A_init=None, init_info=None):
    """Constrained stable A via projected gradient descent with a
    conditional-gradient (Frank-Wolfe-style) step.

        minimize  f(A) = ||Z0 A^T + U0 B^T - Zp||^2_F
        s.t.      spectral_radius(A) <= rho_cap

    B is fixed at the joint-lstsq value. f is a convex quadratic in A;
    the Schur set is non-convex, so the result is at best a local
    constrained minimiser of the LS objective. The radial eigenvalue-clip projection
    Proj (`_project_schur`, cap rho_cap-delta) is highly non-smooth on
    strongly non-normal operators: a pure first-order step from the clip
    operator is immediately clipped back, so plain projected gradient
    stalls at the clip. A conditional-gradient direction avoids this:
    the unconstrained optimum A_ls is the linear-minimisation oracle of
    the convex f, and the feasible search arc

        A(alpha) = Proj( A + alpha (A_ls - A) ),  alpha in (0, 1]

    is a descent path along the Schur boundary. Each iteration first
    line-searches this arc, then takes a projected-gradient step with a
    backtracking step size starting at 8*eta (eta = 1/L, L = 2||Z0^T Z0||_2
    the Lipschitz constant of grad f). Either move is accepted only on strict
    decrease of the objective, and the best feasible iterate is kept.

    grad f(A) = 2 (A Z0^T Z0 + B U0^T Z0 - Zp^T Z0).

    Start: the eigenvalue-clip operator (stab_method="clip"). Because the
    scheme only accepts decreases of the objective and keeps the best
    feasible iterate, its residual is never above the clip residual.
    Deterministic: fixed start, fixed alpha grid, no RNG.

    Method: projected gradient + conditional gradient (Frank-Wolfe) for a
    convex objective over a non-convex constraint; the Proj step is the
    eigenvalue clip of _project_schur.

    Returns (A_c (d,d) real, info) with stab_method="constrained",
    pgd_iters, pgd_gradmap, rho_stab, n_eig_clipped, cond_V.
    """
    A_ls = np.asarray(A_ls, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    Z0 = np.asarray(Z0, dtype=np.float64)
    U0 = np.asarray(U0, dtype=np.float64)
    Zp = np.asarray(Zp, dtype=np.float64)
    cap = float(rho_cap) - float(delta)

    # precompute the quadratic's constant pieces
    G = Z0.T @ Z0                        # (d,d)  == Z0^T Z0  (PSD)
    C = (B @ (U0.T @ Z0)) - (Zp.T @ Z0)  # (d,d)  const part of grad/2
    L = 2.0 * float(np.linalg.norm(G, 2))
    eta = 1.0 / L if L > 0 else 1.0      # exact-Lipschitz safe step

    def grad(A):
        return 2.0 * (A @ G + C)

    # Start at the eigenvalue-clip operator; the objective starts at the clip
    # residual and every accepted move lowers it, so constrained <= clip.
    if A_init is not None:
        A_clip = np.asarray(A_init, dtype=np.float64).copy()
        pinfo = dict(init_info) if init_info else {}
    else:
        A_clip, pinfo = _project_schur(A_ls, rho_cap=float(rho_cap))
    A = A_clip.copy()
    f = _ls_residual(A, B, Z0, U0, Zp) ** 2
    best_A, best_f = A.copy(), f             # never worse than clip
    # conditional-gradient alpha grid (pull toward the unconstrained
    # optimum A_ls), deterministic and geometrically spaced.
    alpha_grid = (0.9, 0.6, 0.4, 0.25, 0.15, 0.08,
                  0.04, 0.02, 0.01, 5e-3, 2e-3)
    gradmap = float("inf")
    it = 0
    for it in range(1, max_iter + 1):
        improved = False
        # (1) conditional-gradient: feasible arc toward A_ls (escapes
        #     the non-smooth clip fixed point)
        for a in alpha_grid:
            cand, _ = _project_schur(A + a * (A_ls - A), rho_cap=cap)
            f_cand = _ls_residual(cand, B, Z0, U0, Zp) ** 2
            if f_cand < best_f - 1e-13 * max(1.0, best_f):
                A, f = cand, f_cand
                best_A, best_f = cand.copy(), f_cand
                improved = True
                break
        # (2) projected-gradient step with backtracking step size
        g = grad(A)
        gnorm = float(np.linalg.norm(g))
        s = eta * 8.0
        for _ in range(40):
            cand, _ = _project_schur(A - s * g, rho_cap=cap)
            f_cand = _ls_residual(cand, B, Z0, U0, Zp) ** 2
            if f_cand < best_f - 1e-13 * max(1.0, best_f):
                A, f = cand, f_cand
                best_A, best_f = cand.copy(), f_cand
                improved = True
                gradmap = s * gnorm / (1.0 + float(np.linalg.norm(A)))
                break
            s *= 0.5
        if not improved:
            gradmap = 0.0
            break
        if gradmap < tol:
            break

    A = best_A
    rho_stab = (float(np.max(np.abs(np.linalg.eigvals(A))))
                if A.size else 0.0)
    # n_eig_clipped counts the eigenvalues of the unconstrained A_ls outside
    # the disk, i.e. how much the raw fit had to be stabilised. It comes from
    # the initial projection of A_ls (> 0 whenever A_ls is unstable); later
    # iterates can clip 0 eigenvalues, which would understate it.
    info = {
        "rho_raw": float(np.max(np.abs(np.linalg.eigvals(A_ls))))
        if A_ls.size else 0.0,
        "rho_stab": rho_stab,
        "n_eig_clipped": int(pinfo.get("n_eig_clipped", 0)),
        "stab_method": "constrained",
        "cond_V": float(pinfo.get("cond_V", float("nan"))),
        "pgd_iters": int(it),
        "pgd_gradmap": float(gradmap),
    }
    return np.ascontiguousarray(A.astype(np.float64)), info


def _dmdc_solve(z_seq: np.ndarray, u_seq: np.ndarray, rcond: float,
                stab_rho_cap: float = 1.0, stab_method: str = "constrained",
                pgd_max_iter: int = 500, pgd_tol: float = 1e-7,
                pgd_delta: float = 1e-6):
    """DMDc: closed-form least squares, then mandatory stabilization.

    z_seq : (N, T, d) latent trajectories (normalized)
    u_seq : (N, T, c) control trajectories (normalized)
    Stacks pairs (z_t -> z_{t+1}) with the time-aligned control u_t and
    solves  [A|B] = Z_plus @ pinv([Z0; U])  via np.linalg.lstsq (SVD-based)
    with an explicit rcond. A_ls is the unconstrained least-squares fit.
    If rho(A_ls) <= cap it is used unchanged. Otherwise the operator is
    made Schur-stable per `stab_method`:

      * "constrained" (default): projected gradient descent to a local
        minimum of the LS objective on the Schur set (_constrained_stable_A).
      * "clip": eigenvalue-magnitude clip (_project_schur).
      * "none": no stabilisation (the raw A_ls).

    B is fixed at the joint-lstsq value for every method. Returns
    (A (d,d), B (d,c), info) with rcond, cond([Z0;U]), eigvals,
    spectral_radius (post), spectral_radius_raw (pre), n_eig_clipped,
    stab_method, the three LS residuals ls_resid_raw <=
    ls_resid_constrained <= ls_resid_clip, and pgd_iters + pgd_gradmap
    for the constrained case.
    """
    z = np.asarray(z_seq, dtype=np.float64)
    u = np.asarray(u_seq, dtype=np.float64)
    N, T, d = z.shape
    c = u.shape[2]
    # pairs over every case and every t in [0, T-2]
    Z0 = z[:, :-1, :].reshape(-1, d)            # (P, d)
    Zp = z[:, 1:, :].reshape(-1, d)             # (P, d)
    U0 = u[:, :-1, :].reshape(-1, c)            # (P, c)
    Phi = np.concatenate([Z0, U0], axis=1)      # (P, d+c)  == [Z0 | U]
    # cond([Z0;U]): numerical-conditioning diagnostic
    try:
        cond_phi = float(np.linalg.cond(Phi))
    except np.linalg.LinAlgError:
        cond_phi = float("inf")
    # SVD-based least squares: M solves Phi @ M ~= Zp  ->  M (d+c, d)
    M, _res, _rank, _sv = np.linalg.lstsq(Phi, Zp, rcond=rcond)
    A_raw = np.ascontiguousarray(M[:d, :].T)    # (d, d) optimal lstsq fit
    B = np.ascontiguousarray(M[d:, :].T)        # (d, c)
    ev_raw = np.linalg.eigvals(A_raw)
    spectral_radius_raw = (float(np.max(np.abs(ev_raw)))
                           if ev_raw.size else 0.0)

    method = str(stab_method)
    cap = float(stab_rho_cap)
    # LS residual of the unconstrained optimum (lower bound of the ordering)
    ls_resid_raw = _ls_residual(A_raw, B, Z0, U0, Zp)

    already_stable = spectral_radius_raw <= cap + 1e-8

    # Always compute the clip operator and its residual: it is the start of
    # "constrained" and the upper bound of raw <= constrained <= clip. If A_ls
    # is already stable, all three coincide with raw.
    if already_stable:
        A_clip, cinfo = A_raw.copy(), {
            "rho_raw": spectral_radius_raw, "rho_stab": spectral_radius_raw,
            "n_eig_clipped": 0, "stab_method": "none",
            "cond_V": float("nan")}
        ls_resid_clip = ls_resid_raw
    else:
        A_clip, cinfo = _project_schur(A_raw, rho_cap=cap)
        ls_resid_clip = _ls_residual(A_clip, B, Z0, U0, Zp)

    if method == "none":
        A_stab, sinfo = A_raw.copy(), {
            "n_eig_clipped": 0, "stab_method": "none",
            "cond_V": float("nan")}
        ls_resid_constrained = ls_resid_raw
    elif method == "clip":
        A_stab, sinfo = A_clip, cinfo
        ls_resid_constrained = ls_resid_clip
    elif method == "constrained":
        if already_stable:
            # A_ls is already stable: use it unchanged.
            A_stab, sinfo = A_raw.copy(), {
                "n_eig_clipped": 0, "stab_method": "constrained",
                "cond_V": float("nan"), "pgd_iters": 0,
                "pgd_gradmap": 0.0}
            ls_resid_constrained = ls_resid_raw
        else:
            # Start the PGD at the clip operator (same cap, no delta), so
            # constrained <= clip by construction.
            A_stab, sinfo = _constrained_stable_A(
                A_raw, B, Z0, U0, Zp, rho_cap=cap, delta=float(pgd_delta),
                max_iter=int(pgd_max_iter), tol=float(pgd_tol),
                A_init=A_clip, init_info=cinfo)
            ls_resid_constrained = _ls_residual(A_stab, B, Z0, U0, Zp)
    else:
        raise ValueError(
            f"dmdc: stab_method must be 'constrained'|'clip'|'none', "
            f"got {method!r}")

    A = np.ascontiguousarray(A_stab)
    ev = np.linalg.eigvals(A)
    spectral_radius = float(np.max(np.abs(ev))) if ev.size else 0.0
    info = {
        "rcond": float(rcond),
        "cond": cond_phi,
        "eigvals_real": ev.real.astype(np.float64),
        "eigvals_imag": ev.imag.astype(np.float64),
        "eigvals_raw_real": ev_raw.real.astype(np.float64),
        "eigvals_raw_imag": ev_raw.imag.astype(np.float64),
        "spectral_radius": spectral_radius,
        "spectral_radius_raw": spectral_radius_raw,
        "stab_rho_cap": cap,
        "n_eig_clipped": int(sinfo["n_eig_clipped"]),
        "stab_method": method,
        "cond_V": float(sinfo.get("cond_V", float("nan"))),
        "pgd_iters": int(sinfo.get("pgd_iters", 0)),
        "pgd_gradmap": float(sinfo.get("pgd_gradmap", float("nan"))),
        "ls_resid_raw": float(ls_resid_raw),
        "ls_resid_constrained": float(ls_resid_constrained),
        "ls_resid_clip": float(ls_resid_clip),
        "rank": int(_rank),
        "n_pairs": int(Phi.shape[0]),
    }
    return A.astype(np.float64), B.astype(np.float64), info


def _linear_rollout(A: np.ndarray, B: np.ndarray, z0: np.ndarray,
                    u_seq: np.ndarray) -> np.ndarray:
    """z_{t+1} = A z_t + B u_t from z0 over the eval horizon.

    z0    : (N, d)        encoded t=0 latent (normalized)
    u_seq : (N, T, c)     control (normalized)
    Returns (N, T, d). NaN/Inf-safe: any overflow is clipped to +/-_Z_CLIP,
    so the rollout and decode never crash and a divergence stays visible
    (large but finite) in the metrics. The operator A is not modified."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    z = np.asarray(z0, dtype=np.float64)        # (N, d)
    u = np.asarray(u_seq, dtype=np.float64)     # (N, T, c)
    N, T, c = u.shape
    d = A.shape[0]
    out = np.empty((N, T, d), dtype=np.float64)
    out[:, 0, :] = np.nan_to_num(z, nan=0.0, posinf=_Z_CLIP,
                                 neginf=-_Z_CLIP)
    z = out[:, 0, :]
    for t in range(T - 1):
        with np.errstate(over="ignore", invalid="ignore"):
            z = z @ A.T + u[:, t, :] @ B.T
        z = np.nan_to_num(z, nan=0.0, posinf=_Z_CLIP, neginf=-_Z_CLIP)
        z = np.clip(z, -_Z_CLIP, _Z_CLIP)
        out[:, t + 1, :] = z
    return out


@register_adapter("dmdc")
class DMDcAdapter(BenchmarkAdapter):
    """Linear DMDc baseline on the frozen KVS ConvAE latent."""

    # ----------------------------------------------------------- build ---
    def build(self, cfg: dict, seed: int) -> dict:
        set_seed(seed)
        m = cfg.get("model", {})
        dev = _device(m)
        dataset = str(cfg["dataset"])
        if dataset == "kvs":
            enc_ref = cfg["encoder_ref"]
            (ae, (H, W), latent_dim, z_mean, z_std,
             x_mean, x_std) = _load_frozen_encoder(enc_ref, dev)
            holder = {"encoder": ae, "grid_hw": [H, W],
                      "latent_dim": int(latent_dim),
                      "x_mean": x_mean, "x_std": x_std}
        else:
            raise ValueError(f"dmdc: dataset must be 'kvs', got {dataset!r}")
        holder.update({
            "dataset": dataset, "device": dev, "encoder_ref": enc_ref,
            # placeholders for the fitted operator (no torch params -> n_params=0)
            "A": None, "B": None, "n_params": 0,
            "run_meta": {"encoder_ref": enc_ref, "dataset": dataset,
                         "dynamics": "dmdc",
                         "latent_dim": int(holder["latent_dim"])},
        })
        return holder

    # ------------------------------------------------------ encode TRAIN -
    def _encode_train(self, ms: dict, view):
        """-> (z (N,T,d) raw latent, u (N,T,2) raw control (s, ds/dt)).

        The control comes from ramp_s_dsdt, the same features as the AR-MLP row."""
        ae = ms["encoder"]
        dev = ms["device"]
        frames = view.grid.frames_for(view.case_ids, dtype=np.float16)
        z = _encode_traj(ae, frames, dev, ms["x_mean"], ms["x_std"])
        # control u = (s, ds/dt) from ramp_s_dsdt(theta[:,1:], t_grid)
        t_grid = np.asarray(view.t_grid, dtype=np.float64)
        n = len(view.case_ids)
        th = np.asarray(view.theta)
        s = np.stack([_common.ramp_s_dsdt(th[i, 1:], t_grid)[0]
                      for i in range(n)], axis=0)
        ds = np.stack([_common.ramp_s_dsdt(th[i, 1:], t_grid)[1]
                       for i in range(n)], axis=0)
        u = np.stack([s, ds], axis=-1).astype(np.float32)   # (N,T,2)
        return z.astype(np.float32), u

    # ------------------------------------------------------------- fit ---
    def fit(self, ms: dict, train, val, cfg: dict, seed: int):
        """SVD-lstsq DMDc fit, then stabilisation. No epoch loop,
        no RNG (deterministic given the data)."""
        set_seed(seed)
        m = cfg.get("model", {})
        rcond = float(m.get("rcond", 1e-10))
        stab_rho_cap = float(m.get("stab_rho_cap", 1.0))
        stab_method = str(m.get("stab_method", "constrained"))
        pgd_max_iter = int(m.get("pgd_max_iter", 500))
        pgd_tol = float(m.get("pgd_tol", 1e-7))
        pgd_delta = float(m.get("pgd_delta", 1e-6))

        z_tr, u_tr = self._encode_train(ms, train)              # RAW
        N, T, d = z_tr.shape
        c = u_tr.shape[2]

        # Per-dimension latent mean/std computed from the TRAIN latents, as in the
        # branch-DeepONet AR rows.
        zf = z_tr.reshape(-1, d)
        z_mean_lat = zf.mean(0).astype(np.float64)
        z_std_lat = (zf.std(0) + 1e-8).astype(np.float64)
        z_tr_n = (z_tr - z_mean_lat) / z_std_lat

        # Per-channel control mean/std from TRAIN.
        uf = u_tr.reshape(-1, c)
        u_mean = uf.mean(0).astype(np.float64)
        u_std = (uf.std(0) + 1e-8).astype(np.float64)
        u_tr_n = (u_tr - u_mean) / u_std

        # theta statistics are stored with the checkpoint like in the AR rows;
        # theta itself is not an input of the DMDc model (its control is (s, ds/dt)).
        th_tr = np.asarray(train.theta, dtype=np.float64)
        th_mean = th_tr.mean(0)
        th_std = th_tr.std(0) + 1e-8

        A, B, info = _dmdc_solve(
            z_tr_n, u_tr_n, rcond, stab_rho_cap,
            stab_method=stab_method, pgd_max_iter=pgd_max_iter,
            pgd_tol=pgd_tol, pgd_delta=pgd_delta)

        ck_dir = (REPO / "results" / "benchmark" / "runs"
                  / f"{cfg['model_name']}_seed{seed}_ckpts")
        ck_dir.mkdir(parents=True, exist_ok=True)
        eig_list = [[float(r), float(i)] for r, i in
                    zip(info["eigvals_real"], info["eigvals_imag"])]
        payload = {
            "A": A, "B": B,
            "z_mean_lat": z_mean_lat, "z_std_lat": z_std_lat,
            "u_mean": u_mean, "u_std": u_std,
            "th_mean": th_mean, "th_std": th_std,
            "encoder_ref": ms["encoder_ref"], "dataset": ms["dataset"],
            "rcond": info["rcond"], "cond": info["cond"],
            "spectral_radius": info["spectral_radius"],
            "spectral_radius_raw": info["spectral_radius_raw"],
            "stab_rho_cap": info["stab_rho_cap"],
            "n_eig_clipped": info["n_eig_clipped"],
            "stab_method": info["stab_method"],
            "pgd_iters": info["pgd_iters"],
            "pgd_gradmap": info["pgd_gradmap"],
            "ls_resid_raw": info["ls_resid_raw"],
            "ls_resid_constrained": info["ls_resid_constrained"],
            "ls_resid_clip": info["ls_resid_clip"],
            "eigvals": eig_list,
        }
        # best.pt and last.pt are identical; each holds the full model.
        best_ckpt = save_ckpt(ck_dir, "best.pt", payload)
        last_ckpt = save_ckpt(ck_dir, "last.pt", payload)

        print(f"[dmdc:{ms['dataset']}] closed-form lstsq solved "
              f"d={d} c={c} pairs={info['n_pairs']} rcond={info['rcond']:.1e} "
              f"cond([Z0;U])={info['cond']:.3e} "
              f"spectral_radius_raw={info['spectral_radius_raw']:.4f} "
              f"-> stabilized spectral_radius={info['spectral_radius']:.6f} "
              f"(cap={info['stab_rho_cap']:.3f} "
              f"n_eig_clipped={info['n_eig_clipped']} "
              f"method={info['stab_method']} "
              f"pgd_iters={info['pgd_iters']} "
              f"pgd_gradmap={info['pgd_gradmap']:.2e}) "
              f"(rank={info['rank']})", flush=True)
        print(f"[dmdc:{ms['dataset']}] LS residual ordering "
              f"raw={info['ls_resid_raw']:.6e} <= "
              f"constrained={info['ls_resid_constrained']:.6e} <= "
              f"clip={info['ls_resid_clip']:.6e} "
              f"(the constrained solve starts at the clip operator "
              f"and accepts only decreases)", flush=True)
        if info["spectral_radius_raw"] > 1.0:
            print(f"[dmdc:{ms['dataset']}] raw unconstrained A has "
                  f"spectral radius > 1 (rho_raw={info['spectral_radius_raw']:.4f}"
                  f", method={info['stab_method']}). The rollout uses the "
                  f"operator of this method; the raw value is stored as "
                  f"spectral_radius_raw.", flush=True)
        if ms["dataset"] == "kvs":
            print("[dmdc:kvs] a stable linear map has no limit cycle, so "
                  "this row cannot reproduce self-excited vortex shedding.",
                  flush=True)

        ms.update({
            "A": A, "B": B,
            "z_mean_lat": z_mean_lat, "z_std_lat": z_std_lat,
            "u_mean": u_mean, "u_std": u_std,
            "th_mean": th_mean, "th_std": th_std,
            "best_ckpt_path": str(best_ckpt),
            "last_ckpt_path": str(last_ckpt),
            "dmdc_info": info,
            "run_meta": {**ms.get("run_meta", {}),
                         "rcond": info["rcond"], "cond": info["cond"],
                         "spectral_radius": info["spectral_radius"],
                         "spectral_radius_raw":
                             info["spectral_radius_raw"],
                         "stab_rho_cap": info["stab_rho_cap"],
                         "n_eig_clipped": info["n_eig_clipped"],
                         "stab_method": info["stab_method"],
                         "pgd_iters": info["pgd_iters"],
                         "pgd_gradmap": info["pgd_gradmap"],
                         "ls_resid_raw": info["ls_resid_raw"],
                         "ls_resid_constrained":
                             info["ls_resid_constrained"],
                         "ls_resid_clip": info["ls_resid_clip"],
                         "eigvals": eig_list,
                         "n_pairs": info["n_pairs"],
                         "rank": info["rank"]},
        })
        # no epoch loop: a single "step", no train/val curves
        hist = {"train": [float(0.0)], "val": [float("nan")]}
        return ms, hist

    # ----------------------------------------------------- predict_eval --
    def predict_eval(self, ms: dict, test, split_spec, cfg: dict):
        return self._predict_kvs(ms, test, split_spec, cfg)

    def _meta(self, ms: dict) -> dict:
        info = ms["dmdc_info"]
        return {
            "target_type": "field",
            "dataset": ms["dataset"],
            "encoder_ref": ms["encoder_ref"],
            "dynamics": "dmdc",
            "rcond": info["rcond"],
            "cond": info["cond"],
            "spectral_radius": info["spectral_radius"],
            "spectral_radius_raw": info["spectral_radius_raw"],
            "stab_rho_cap": info["stab_rho_cap"],
            "n_eig_clipped": info["n_eig_clipped"],
            "stab_method": info["stab_method"],
            "pgd_iters": info["pgd_iters"],
            "pgd_gradmap": info["pgd_gradmap"],
            "ls_resid_raw": info["ls_resid_raw"],
            "ls_resid_constrained": info["ls_resid_constrained"],
            "ls_resid_clip": info["ls_resid_clip"],
            "eigvals": [[float(r), float(i)] for r, i in
                        zip(info["eigvals_real"], info["eigvals_imag"])],
        }

    # --- KVS: ConvAE encode of t=0 -> linear rollout -> decode ----------
    def _predict_kvs(self, ms: dict, test, split_spec, cfg: dict):
        dev = ms["device"]
        ae = ms["encoder"].to(dev).eval()
        H, W = ms["grid_hw"]
        x_mean, x_std = ms["x_mean"], ms["x_std"]
        z_mean = ms["z_mean_lat"]
        z_std = ms["z_std_lat"]
        u_mean, u_std = ms["u_mean"], ms["u_std"]

        ids = test.case_ids
        n_test = len(ids)
        frames = test.grid.frames_for(ids, dtype=np.float16)
        T = frames.shape[1]
        t_grid = np.asarray(test.t_grid, dtype=np.float64)

        # control u = (s, ds/dt), built as in _encode_train
        th = np.asarray(test.theta)
        s = np.stack([_common.ramp_s_dsdt(th[i, 1:], t_grid)[0]
                      for i in range(n_test)], axis=0)
        ds = np.stack([_common.ramp_s_dsdt(th[i, 1:], t_grid)[1]
                       for i in range(n_test)], axis=0)
        u = np.stack([s, ds], axis=-1).astype(np.float32)        # (N,T,2)
        u_n = (u - u_mean) / u_std

        # encode only the t=0 frame (z0) with the frozen ConvAE
        z0_raw = _encode_traj(ae, frames[:, 0:1], dev, x_mean,
                              x_std)[:, 0, :]                     # (N,d)
        z0_n = (z0_raw - z_mean) / z_std
        z_n = _linear_rollout(ms["A"], ms["B"], z0_n, u_n)        # (N,T,d)
        z_phys_lat = (z_n * z_std + z_mean).astype(np.float32)

        pred_cfhw = np.empty((n_test, T, 3, H, W), dtype=np.float32)
        with torch.no_grad():
            for ci in range(n_test):
                zc = torch.from_numpy(z_phys_lat[ci]).to(dev)
                dec_norm = ae.decode(zc)
                dec_phys = dec_norm * x_std + x_mean
                pred_cfhw[ci] = np.nan_to_num(
                    dec_phys.cpu().numpy(), nan=0.0,
                    posinf=_Z_CLIP, neginf=-_Z_CLIP)
        pred_grid = _common.channels_first_to_grid(pred_cfhw)
        arrays = _common.assemble_kvs_field_arrays(
            np.asarray(pred_grid, dtype=np.float32), ids, np.asarray(t_grid, dtype=np.float32),
            coords_p2=test.mesh["coords_p2"],
            coords_p1=test.mesh["coords_p1"],
            eval_idx_p2=split_spec.eval_idx_p2,
            eval_idx_p1=split_spec.eval_idx_p1,
            bbox=test.grid.bbox)
        # final NaN/Inf guard (a divergence stays visible but finite)
        for k in ("u_eval", "p_eval"):
            arrays[k] = np.nan_to_num(np.asarray(arrays[k]), nan=0.0,
                                      posinf=_Z_CLIP, neginf=-_Z_CLIP
                                      ).astype(np.float32)
        return arrays, self._meta(ms)
