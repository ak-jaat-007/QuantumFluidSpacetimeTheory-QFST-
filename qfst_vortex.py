# qfst_vortex.py
"""
qfst_vortex.py

Lightweight QFST vortex module used by the pipeline scripts.

Provides:
 - QFST class with vortex density / enclosed mass / velocity methods
 - Top-level convenience functions for backwards compatibility:
     vortex_density, vortex_mass_enclosed, vortex_v,
     rho0_from_scaling, rcore_from_scaling
 - A small helper to fit rho0/r_core to a single observed rotation curve
   (least-squares). This is optional but handy for debugging.

Units convention:
 - radius r in kpc
 - mass in M_sun
 - velocities in km/s
 - densities in M_sun / kpc^3
 - G (gravitational constant) imported here: kpc (km/s)^2 / M_sun
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares

# Physical constant (consistent with SPARC usage)
G = 4.30091727003628e-6  # kpc * (km/s)^2 / M_sun

# ------------------------------
# Core QFST class
# ------------------------------
class QFST:
    def __init__(self):
        """Initialize QFST helper instance."""
        pass

    # --- density profile (phenomenological vortex profile) ---
    def vortex_density(self, r, rho0, r_core):
        """
        Vortex energy density profile (phenomenological).
            rho_v(r) = rho0 / (1 + (r/r_core)^2)

        Args:
            r : array_like radius [kpc]
            rho0 : central density [M_sun / kpc^3]
            r_core : core radius [kpc]

        Returns:
            rho_v : ndarray same shape as r
        """
        r = np.asarray(r)
        # avoid divide-by-zero if r_core==0
        r_core_safe = max(r_core, 1e-8)
        return rho0 / (1.0 + (r / r_core_safe)**2)

    # --- enclosed mass from density ---
    def vortex_mass_enclosed(self, r, rho_v):
        """
        Compute enclosed mass M(<r) = 4*pi * integral_0^r rho_v(r') r'^2 dr'
        Uses cumulative_trapezoid integration (input r must be sorted ascending).

        Args:
            r : 1D array [kpc]
            rho_v : 1D array [M_sun / kpc^3]

        Returns:
            M_enc : 1D array [M_sun] same length as r
        """
        r = np.asarray(r)
        rho_v = np.asarray(rho_v)
        if r.ndim != 1 or rho_v.ndim != 1:
            raise ValueError("r and rho_v must be 1D arrays")
        integrand = 4.0 * np.pi * rho_v * r**2
        # cumulative_trapezoid returns length len(r)-1; prepend 0 to match r
        M_enc = np.concatenate(([0.0], cumulative_trapezoid(integrand, r)))
        return M_enc

    # --- rotation contribution from vortex mass ---
    def vortex_v(self, r, rho0, r_core):
        """
        Compute vortex-induced circular velocity:
            v_vortex(r) = sqrt( G * M_enc(r) / r )

        Args:
            r : 1D array [kpc]
            rho0 : central density [M_sun / kpc^3]
            r_core : core radius [kpc]

        Returns:
            v_v : 1D array [km/s]
        """
        r = np.asarray(r)
        rho_v = self.vortex_density(r, rho0, r_core)
        M_enc = self.vortex_mass_enclosed(r, rho_v)
        r_safe = np.maximum(r, 1e-8)
        return np.sqrt(np.maximum(0.0, G * M_enc / r_safe))

    # --- scaling relations used by the pipeline (global model) ---
    def rho0_from_scaling(self, A, p, M_b):
        """
        Scaling law: rho0 = A * (M_b / 1e10)^p
        M_b in M_sun, A in same units as rho0 (M_sun/kpc^3)
        """
        return A * (M_b / 1e10)**p

    def rcore_from_scaling(self, B, q, R_scale):
        """
        Scaling: r_core = B * (R_scale)^q
        R_scale in kpc, returns r_core in kpc
        """
        return B * (R_scale**q)

# ------------------------------
# Backwards-compatible top-level functions
# ------------------------------
_qfst_singleton = QFST()

def vortex_density(r, rho0, r_core):
    return _qfst_singleton.vortex_density(r, rho0, r_core)

def vortex_mass_enclosed(r, rho_v):
    return _qfst_singleton.vortex_mass_enclosed(r, rho_v)

def vortex_v(r, rho0, r_core):
    return _qfst_singleton.vortex_v(r, rho0, r_core)

def rho0_from_scaling(A, p, M_b):
    return _qfst_singleton.rho0_from_scaling(A, p, M_b)

def rcore_from_scaling(B, q, R_scale):
    return _qfst_singleton.rcore_from_scaling(B, q, R_scale)

# ------------------------------
# Optional helper: fit rho0 & r_core to one observed rotation curve
# ------------------------------
def fit_rho_rcore_to_vobs(r, v_obs, v_err, v_baryon,
                          rho0_guess=1e8, r_core_guess=1.0,
                          bounds=([1e3, 0.01], [1e13, 50.0]),
                          xtol=1e-8, ftol=1e-8):
    """
    Fit (rho0, r_core) to observed total rotation curve assuming:
        v_total^2 = v_baryon^2 + v_vortex(rho0, r_core)^2

    Returns dictionary with fitted parameters and model arrays.
    """
    r = np.asarray(r)
    v_obs = np.asarray(v_obs)
    v_err = np.asarray(v_err)
    v_baryon = np.asarray(v_baryon)

    mask = (~np.isnan(v_obs)) & (v_err > 0)
    if np.sum(mask) == 0:
        raise ValueError("No valid data points provided for fitting")

    def residuals(x):
        rho0, r_core = x
        v_v = _qfst_singleton.vortex_v(r, rho0, r_core)
        v_tot = np.sqrt(v_baryon**2 + v_v**2)
        return (v_tot[mask] - v_obs[mask]) / v_err[mask]

    x0 = [rho0_guess, r_core_guess]
    out = least_squares(residuals, x0, bounds=bounds, xtol=xtol, ftol=ftol)
    rho0_fit, r_core_fit = out.x
    v_v_fit = _qfst_singleton.vortex_v(r, rho0_fit, r_core_fit)
    v_tot_fit = np.sqrt(v_baryon**2 + v_v_fit**2)

    return {
        "rho0": float(rho0_fit),
        "r_core": float(r_core_fit),
        "v_v": v_v_fit,
        "v_total": v_tot_fit,
        "success": bool(out.success),
        "cost": float(out.cost),
        "message": out.message
    }

# ------------------------------
# Minimal test runner when module executed directly
# ------------------------------
if __name__ == "__main__":
    # quick smoke-test
    r = np.linspace(0.1, 30.0, 100)
    rho0 = 1e9
    r_core = 2.0
    v_v = vortex_v(r, rho0, r_core)
    print("Smoke test: computed vortex v (km/s) range:", v_v.min(), v_v.max())
