# qfst_hybrid_fit.py
"""
Hybrid global + per-galaxy QFST fitting pipeline.

1) Fit global scaling law (A,p,B,q).
2) For each galaxy, fit multiplicative adjustments s_rho, s_r to allow
   per-galaxy flexibility while regularizing deviations from 1.
3) Fit per-galaxy NFW (for comparison).
4) Produce per-galaxy plots and a summary CSV.

Usage:
    pip install numpy scipy matplotlib pandas requests tqdm
    python qfst_hybrid_fit.py
"""
import os, io, glob, zipfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import cumulative_trapezoid
import pandas as pd
import requests
from tqdm import tqdm
from math import log

# ---------------- CONFIG ----------------
DOWNLOAD_SPARC = False   # set True to download if needed
SPARC_URLS = [
    "https://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip",
    "https://zenodo.org/record/1038426/files/Rotmod_LTG.zip"
]
SPARC_ZIP_PATH = "Rotmod_LTG.zip"
EXTRACT_DIR = "SPARC_Rotmod"

G = 4.30091727003628e-6  # kpc (km/s)^2 / M_sun

# Error floor and opts
ERROR_FLOOR_FRACTION = 0.05
ERROR_FLOOR_MIN = 2.0
LEAST_SQUARES_OPTIONS = dict(xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=1000)

# Regularization on per-galaxy multiplicative factors
LAMBDA_REG = 4.0   # larger -> keep s closer to 1; tune this for desired flexibility

OUT_DIR = "qfst_hybrid_plots"
OUT_CSV = "qfst_hybrid_summary.csv"

# ---------------- Utilities: download/extract/load ----------------
def download_sparc(zip_path=SPARC_ZIP_PATH, urls=SPARC_URLS):
    if os.path.exists(zip_path):
        return zip_path
    for url in urls:
        try:
            resp = requests.get(url, stream=True, timeout=90); resp.raise_for_status()
            total = int(resp.headers.get("content-length",0))
            with open(zip_path,"wb") as f:
                for chunk in tqdm(resp.iter_content(8192), total=total//8192 if total else None, unit="KB"):
                    if chunk: f.write(chunk)
            return zip_path
        except Exception as e:
            print("download failed:", e)
    raise FileNotFoundError("Download SPARC manually and place Rotmod_LTG.zip here.")

def extract_sparc(zip_path=SPARC_ZIP_PATH, out_dir=EXTRACT_DIR):
    if os.path.isdir(out_dir) and len(os.listdir(out_dir))>0:
        return out_dir
    with zipfile.ZipFile(zip_path,"r") as z: z.extractall(out_dir)
    return out_dir

def find_all_sparc_files(extracted_dir=EXTRACT_DIR):
    files = glob.glob(os.path.join(extracted_dir, "**", "*"), recursive=True)
    rot_files = [f for f in files if os.path.isfile(f) and f.lower().endswith((".dat",".txt",".asc"))]
    return rot_files

def find_sparc_file(galaxy_name, extracted_dir=EXTRACT_DIR):
    matches = glob.glob(os.path.join(extracted_dir, "**", f"*{galaxy_name}*"), recursive=True)
    if not matches: return None
    for m in matches:
        n = os.path.basename(m).lower()
        if "rotmod" in n or m.lower().endswith((".dat",".txt",".asc")):
            return m
    return matches[0]

def load_rotmod(filepath):
    text = open(filepath, "r", encoding="latin1").read()
    lines = text.splitlines()
    data_lines = [ln for ln in lines if ln.strip() and not ln.strip().startswith('#')]
    data = np.loadtxt(io.StringIO("\n".join(data_lines)))
    if data.ndim == 1: data = data.reshape(1, -1)
    if data.shape[1] >= 6:
        R, Vobs, Verr, Vdisk, Vbul, Vgas = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5]
    elif data.shape[1] == 5:
        R, Vobs, Verr, Vdisk, Vgas = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4]
        Vbul = np.zeros_like(R)
    else:
        R, Vobs = data[:,0], data[:,1]; Verr = data[:,2] if data.shape[1]>2 else np.zeros_like(R)
        Vdisk = np.zeros_like(R); Vbul=np.zeros_like(R); Vgas=np.zeros_like(R)
    return {"r":R, "v_obs":Vobs, "v_err":Verr, "v_disk":Vdisk, "v_bulge":Vbul, "v_gas":Vgas}

# ---------------- Models ----------------
class QFST:
    def vortex_density(self, r, rho0, r_core):
        return rho0 / (1.0 + (r / r_core)**2)
    def vortex_mass_enclosed(self, r, rho_v):
        integrand = 4.0 * np.pi * rho_v * r**2
        M_enc = np.concatenate(([0.0], cumulative_trapezoid(integrand, r)))
        return M_enc
    def vortex_v(self, r, rho0, r_core):
        rho_v = self.vortex_density(r, rho0, r_core)
        M_enc = self.vortex_mass_enclosed(r, rho_v)
        r_safe = np.maximum(r, 1e-6)
        return np.sqrt(G * M_enc / r_safe)
    def rho0_from_scaling(self, A, p, M_b):
        return A * (M_b / 1e10)**p
    def rcore_from_scaling(self, B, q, R_scale):
        return B * (R_scale**q)

class NFW:
    def v_nfw(self, r, rho_s, r_s):
        x = r / r_s
        pref = 4.0 * np.pi * rho_s * r_s**3
        M_enc = pref * (np.log(1.0 + x) - x/(1.0 + x))
        r_safe = np.maximum(r, 1e-6)
        return np.sqrt(G * M_enc / r_safe)

# ---------------- Helpers ----------------
def compute_baryonic_mass_scale(r, v_baryon):
    M_enc = r * v_baryon**2 / G
    M_tot = M_enc[-1]
    try:
        idx = np.where(M_enc >= 0.5*M_tot)[0][0]
        R_scale = r[idx]
    except Exception:
        R_scale = r[len(r)//2]
    return M_tot, R_scale, M_enc

def robust_sigma(v_err, v_obs):
    floor = np.maximum(ERROR_FLOOR_FRACTION * np.abs(v_obs), ERROR_FLOOR_MIN)
    sigma = np.maximum(v_err, floor)
    sigma[sigma <= 0] = floor[sigma <= 0] + 1e-6
    return sigma

# ---------------- Stage 1: global fit ----------------
def global_qfst_fit(galaxy_data_list):
    qfst = QFST()
    def stacked_res(x):
        A,p,B,q = x
        residuals = []
        for g in galaxy_data_list:
            r = g['r']; v_obs = g['v_obs']; v_err = g['v_err']; v_baryon = g['v_baryon']
            M_b = g['M_b']; R_scale = g['R_scale']
            rho0 = qfst.rho0_from_scaling(A,p,M_b)
            r_core = qfst.rcore_from_scaling(B,q,R_scale)
            v_v = qfst.vortex_v(r, rho0, r_core)
            v_tot = np.sqrt(v_baryon**2 + v_v**2)
            sigma = robust_sigma(v_err, v_obs)
            mask = (~np.isnan(v_obs)) & (sigma>0)
            if np.sum(mask)==0: continue
            residuals.extend((v_tot[mask] - v_obs[mask]) / sigma[mask])
        return np.array(residuals)
    x0 = np.array([1e9, 0.0, 1.0, 0.5])
    bounds = ([1e6, -2.0, 0.01, 0.0], [1e13, 2.0, 50.0, 2.0])
    out = least_squares(stacked_res, x0, bounds=bounds, **LEAST_SQUARES_OPTIONS)
    return out.x

# ---------------- Stage 2: per-galaxy multiplicative adjustments ----------------
def fit_per_galaxy_adjustments(g, A, p, B, q, lambda_reg=LAMBDA_REG):
    """
    Fit s_rho and s_r to adjust rho0 and r_core per galaxy.
    Minimize residuals with regularization on log(s) to discourage large deviations.
    """
    qfst = QFST()
    r = g['r']; v_obs = g['v_obs']; v_err = g['v_err']; v_baryon = g['v_baryon']
    M_b = g['M_b']; R_scale = g['R_scale']
    rho0_base = qfst.rho0_from_scaling(A,p,M_b)
    rcore_base = qfst.rcore_from_scaling(B,q,R_scale)
    sigma = robust_sigma(v_err, v_obs)
    mask = (~np.isnan(v_obs)) & (sigma>0)
    if np.sum(mask) == 0:
        return {"s_rho":1.0, "s_r":1.0, "rho0":rho0_base, "r_core":rcore_base, "v_model":np.zeros_like(r)}

    def residuals(s):
        s_rho, s_r = s
        rho0 = rho0_base * s_rho
        r_core = rcore_base * s_r
        v_v = qfst.vortex_v(r, rho0, r_core)
        v_tot = np.sqrt(v_baryon**2 + v_v**2)
        res_data = (v_tot[mask] - v_obs[mask]) / sigma[mask]
        # regularize log(s) (s positive) -> penalize multiplicative deviations
        reg = np.sqrt(lambda_reg) * np.array([np.log(max(s_rho,1e-6)), np.log(max(s_r,1e-6))])
        return np.concatenate([res_data, reg])
    # initial s values near 1
    x0 = np.array([1.0, 1.0])
    bounds = ([0.1, 0.1], [10.0, 10.0])
    out = least_squares(residuals, x0, bounds=bounds, **LEAST_SQUARES_OPTIONS)
    s_rho, s_r = out.x
    rho0 = rho0_base * s_rho
    r_core = rcore_base * s_r
    v_v = qfst.vortex_v(r, rho0, r_core)
    v_tot = np.sqrt(v_baryon**2 + v_v**2)
    return {"s_rho":s_rho, "s_r":s_r, "rho0":rho0, "r_core":r_core, "v_model":v_tot, "success":out.success}

# ---------------- NFW per-gal fit ----------------
class NFW_model:
    def v_nfw(self, r, rho_s, r_s):
        x = r / r_s
        pref = 4.0 * np.pi * rho_s * r_s**3
        M_enc = pref * (np.log(1.0 + x) - x/(1.0 + x))
        r_safe = np.maximum(r, 1e-6)
        return np.sqrt(G * M_enc / r_safe)

def fit_nfw_per_galaxy(r, v_obs, v_err, v_baryon):
    nfw = NFW_model()
    r_s0 = max(1.0, 0.2 * np.max(r))
    v_med = np.median(v_obs[np.isfinite(v_obs)])
    r_med = np.median(r)
    M_guess = v_med**2 * r_med / G
    rho_s0 = max(1e6, M_guess / (4*np.pi*r_s0**3))
    x0 = [rho_s0, r_s0]
    bounds = ([1e-6, 0.05], [1e15, 200.0])
    sigma = robust_sigma(v_err, v_obs)
    mask = (~np.isnan(v_obs)) & (sigma>0)
    if np.sum(mask) == 0: return None
    def resid(x):
        rho_s, r_s = x
        v_n = nfw.v_nfw(r, rho_s, r_s)
        v_tot = np.sqrt(v_baryon**2 + v_n**2)
        return (v_tot[mask] - v_obs[mask]) / sigma[mask]
    out = least_squares(resid, x0, bounds=bounds, **LEAST_SQUARES_OPTIONS)
    rho_s, r_s = out.x
    v_n = nfw.v_nfw(r, rho_s, r_s)
    v_tot = np.sqrt(v_baryon**2 + v_n**2)
    return {"rho_s":rho_s, "r_s":r_s, "v_model":v_tot, "success":out.success}

# ---------------- Stats helpers ----------------
def per_galaxy_stats(v_model, v_obs, v_err, n_params):
    mask = (~np.isnan(v_obs)) & (v_err>0)
    N = np.sum(mask)
    if N==0:
        return {"chi2":np.nan, "chi2_red":np.nan, "logL":np.nan, "N":0}
    sigma = robust_sigma(v_err, v_obs)[mask]
    resid = (v_model[mask] - v_obs[mask])
    chi2 = np.sum((resid/sigma)**2)
    chi2_red = chi2 / max(1, N - n_params)
    logL = -0.5 * np.sum((resid/sigma)**2 + np.log(2*np.pi*(sigma**2)))
    return {"chi2":chi2, "chi2_red":chi2_red, "logL":logL, "N":N}

def aic_bic_from_logL(total_logL, total_params, total_points):
    AIC = 2*total_params - 2*total_logL
    BIC = total_params * np.log(total_points) - 2*total_logL
    return AIC, BIC

# ---------------- Main pipeline ----------------
def main():
    if DOWNLOAD_SPARC:
        zip_path = download_sparc(); extract_sparc(zip_path)
    else:
        if not os.path.isdir(EXTRACT_DIR):
            raise RuntimeError("SPARC data missing. Set DOWNLOAD_SPARC=True or extract Rotmod_LTG.zip to " + EXTRACT_DIR)

    # build file list
    all_files = find_all_sparc_files(EXTRACT_DIR)
    if len(all_files) == 0:
        raise RuntimeError("No SPARC files found in " + EXTRACT_DIR)
    galaxy_files = {}
    for f in all_files:
        name = os.path.basename(f).split('_')[0]
        galaxy_files[name] = f

    print(f"[INFO] {len(galaxy_files)} SPARC galaxies found.")

    # load data
    galaxy_data_list = []
    for name, fpath in galaxy_files.items():
        try:
            d = load_rotmod(fpath)
        except Exception as e:
            print("[WARN] load fail", fpath, e); continue
        r = d["r"]; v_obs = d["v_obs"]; v_err = d["v_err"]
        v_baryon = np.sqrt(np.maximum(0.0, d["v_disk"]**2 + d["v_bulge"]**2 + d["v_gas"]**2))
        M_b, R_scale, M_enc = compute_baryonic_mass_scale(r, v_baryon)
        galaxy_data_list.append({"name":name, "file":fpath, "r":r, "v_obs":v_obs, "v_err":v_err, "v_baryon":v_baryon, "M_b":M_b, "R_scale":R_scale})
        print(f"[LOAD] {name}: M_b={M_b:.3e}, R_scale={R_scale:.3f}")

    # Stage 1: global fit
    print("[INFO] Running global QFST fit...")
    xbest = global_qfst_fit(galaxy_data_list)
    A,p,B,q = xbest
    print("[INFO] Global params A,p,B,q =", xbest)

    # Stage 2: per-gal adjustments
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    total_logL_qfst = 0.0
    total_logL_nfw = 0.0
    total_points = 0
    n_gals = 0

    for g in galaxy_data_list:
        name = g["name"]; r = g["r"]; v_obs = g["v_obs"]; v_err = g["v_err"]; v_baryon = g["v_baryon"]
        adjust = fit_per_galaxy_adjustments(g, A, p, B, q, lambda_reg=LAMBDA_REG)
        v_qfst = adjust["v_model"]
        stats_qfst = per_galaxy_stats(v_qfst, v_obs, v_err, n_params=2)  # count 2 effective per-gal adjustments
        total_logL_qfst += stats_qfst["logL"] if not np.isnan(stats_qfst["logL"]) else 0.0

        nfw_fit = fit_nfw_per_galaxy(r, v_obs, v_err, v_baryon)
        if nfw_fit is None:
            print("[WARN] NFW fail", name); continue
        v_nfw = nfw_fit["v_model"]
        stats_nfw = per_galaxy_stats(v_nfw, v_obs, v_err, n_params=2)
        total_logL_nfw += stats_nfw["logL"] if not np.isnan(stats_nfw["logL"]) else 0.0

        N = stats_qfst["N"]
        total_points += N
        n_gals += 1

        # plot
        sigma_plot = robust_sigma(v_err, v_obs)
        plt.figure(figsize=(7,5))
        plt.errorbar(r, v_obs, yerr=sigma_plot, fmt='o', ms=4, label='obs')
        plt.plot(r, v_baryon, '--', label='baryon')
        plt.plot(r, v_qfst, '-', label=f'QFST adj (s_rho={adjust["s_rho"]:.3f}, s_r={adjust["s_r"]:.3f})')
        plt.plot(r, v_nfw, '-.', label=f'NFW (rho_s={nfw_fit["rho_s"]:.2e}, r_s={nfw_fit["r_s"]:.2f})')
        plt.xlabel('r [kpc]'); plt.ylabel('v [km/s]'); plt.title(name)
        plt.legend(); plt.grid()
        png = os.path.join(OUT_DIR, f"{name}_hybrid_compare.png")
        plt.savefig(png, dpi=200); plt.close()

        rows.append({
            "galaxy": name, "file": os.path.basename(g["file"]), "M_b": g["M_b"], "R_scale": g["R_scale"],
            "s_rho": adjust["s_rho"], "s_r": adjust["s_r"], "rho0_qfst": adjust["rho0"], "r_core_qfst": adjust["r_core"],
            "chi2_qfst": stats_qfst["chi2"], "chi2red_qfst": stats_qfst["chi2_red"],
            "rho_s_nfw": nfw_fit["rho_s"], "r_s_nfw": nfw_fit["r_s"], "chi2_nfw": stats_nfw["chi2"], "chi2red_nfw": stats_nfw["chi2_red"],
            "N_points": N, "plot": png
        })
        print(f"[DONE] {name}: chi2_qfst={stats_qfst['chi2']:.1f}, chi2_nfw={stats_nfw['chi2']:.1f}")

    # Global model comparison
    total_params_qfst = 4 + 2*n_gals   # 4 global + 2 per-gal adjustments
    total_params_nfw = 2 * n_gals
    AIC_qfst, BIC_qfst = aic_bic_from_logL(total_logL_qfst, total_params_qfst, total_points)
    AIC_nfw, BIC_nfw = aic_bic_from_logL(total_logL_nfw, total_params_nfw, total_points)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print("\n[SUMMARY]")
    print("Global QFST params (A,p,B,q):", xbest)
    print(f"Galaxies processed: {n_gals}, total data points: {total_points}")
    print(f"Total logL QFST = {total_logL_qfst:.3f}, AIC = {AIC_qfst:.3f}, BIC = {BIC_qfst:.3f}, params={total_params_qfst}")
    print(f"Total logL NFW  = {total_logL_nfw:.3f}, AIC = {AIC_nfw:.3f}, BIC = {BIC_nfw:.3f}, params={total_params_nfw}")
    print(f"Saved summary CSV: {OUT_CSV}; plots in {OUT_DIR}")

if __name__ == "__main__":
    # required import used by functions
    from scipy.integrate import cumulative_trapezoid
    main()
