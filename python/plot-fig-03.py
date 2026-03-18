"""
Plots fig. 03 from Otneim et al. 2026. 4 panels (2x2) for Oslo + Bergen with:

(a) Example Bergen forecast (init: 2015-02-15):
    - ensemble members in grey
    - observations shown in red from 2015-02-01 to 2015-02-28
      * "observed" (solid red) before forecast init
      * "realized" (dashed red) after forecast init

(b) Precipitation forecast skill:
    - MSESS (dashed) and CRPSS (solid) vs lead time
    - optional Day-Of-Year (DOY) climatology reference
    - vertical line at lead day = 4

(c) Precipitation forecast reliability:
    - reliability regression envelope across thresholds
    - aggregated over lead_days

(d) Precipitation forecast bias:
    - aggregated QQ plot over lead_time range
    - axes swapped: x=forecast quantiles, y=observed quantiles
"""

# === 2) Imports ===
import os
import numpy as np
import xarray as xr
import xskillscore as xs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates


# === 3) User input parameters ===
# Cities / plotting
cities = ["oslo", "bergen"]
colors = {"bergen": "#E69F00", "oslo": "#0072B2"}

# Dataset selection
product    = "forecast"
start_date = "2013-01-03"
end_date   = "2021-12-30"

# Climatology period used to define the reference
clim_start = "1960-01-01"
clim_end   = "2021-12-30"

# Panel (a): example forecast
example_city      = "bergen"
example_init_date = np.datetime64("2015-02-15")
example_obs_start = np.datetime64("2015-02-02")
example_obs_end   = np.datetime64("2015-03-02")  # Feb 2015 has 28 days

# Panel (b): skill vs lead time
lead_min_skill      = 1
lead_max_skill      = 15
vline_day           = 4
use_doy_climatology = True
eps_denom           = 1e-12

# Panel (c): reliability envelope (aggregated over lead days)
lead_days_rel     = np.arange(1, 5)  # e.g. 1–4
thresholds_mm_day = [1, 5, 10, 15, 20, 25]
n_bins_rel        = 10

# Panel (d): QQ plot (aggregated over lead times)
lead_min_qq    = 1
lead_max_qq    = 4
n_quantiles    = 101
clip_endpoints = True

# ---- Data paths + filenames (moved here for clarity) ----
# Input paths
path_in_model = '/your_path_for_model_data/'
path_in_obs   = '/your_path_for_observation_data/'
path_in_clim  = '/your_path_for_climatological_data/'

# Input filename templates (filled per city)
# NOTE: these match the original script exactly.
model_file_tpl = "{path}{product}_tp24_{city}_0.25x0.25_{start}_{end}.nc"
obs_file_tpl   = "{path}observation_{product}_tp24_{city}_0.25x0.25_{start}_{end}.nc"
clim_file_tpl  = "{path}continuous_observation_tp24_{city}_0.25x0.25_1960-2023.nc"

# Output
path_out     = '/your_path_for_output_figures/' 
filename_out = "fig_03.pdf"
write2file   = False


# === 4) Matplotlib styling (ggplot theme_classic-ish) ===
classic_rc = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
}
plt.rcParams.update(classic_rc)


# === 5) IO / shared helpers ===
def build_input_filenames(city: str) -> tuple[str, str, str]:
    """Build full paths for model/obs/climatology files for a given city."""
    filename_model = model_file_tpl.format(
        path=path_in_model, product=product, city=city, start=start_date, end=end_date
    )
    filename_obs = obs_file_tpl.format(
        path=path_in_obs, product=product, city=city, start=start_date, end=end_date
    )
    filename_clim = clim_file_tpl.format(path=path_in_clim, city=city)
    return filename_model, filename_obs, filename_clim


def load_model_obs_clim(city: str):
    """Load model, observation, and climatology datasets for a given city."""
    filename_model, filename_obs, filename_clim = build_input_filenames(city)

    model = xr.open_dataset(filename_model).mean(dim=["latitude", "longitude"])
    obs   = xr.open_dataset(filename_obs).mean(dim=["latitude", "longitude"])
    clim  = (
        xr.open_dataset(filename_clim)
        .mean(dim=["latitude", "longitude"])
        .sel(time=slice(clim_start, clim_end))
    )
    return model, obs, clim


def _dims_to_aggregate(model_da):
    if "hdate" in model_da.coords:
        return ["init_time", "hdate"]
    return ["init_time"]


def quantile_levels(n=101, clip=False):
    if clip:
        return np.linspace(0.01, 0.99, n)
    return np.linspace(0.0, 1.0, n)


# =========================
# Panel (a): Example forecast time series
# =========================
def plot_panel_example_forecast(ax, model_ds, clim_ds, init_date, obs_start, obs_end):
    """
    Panel (a): Example city forecast initialised at init_date.

    - Ensemble members in grey (including a synthetic lead_time=0 point at init_date,
      equal to the observed precipitation on init_date).
    - Observations in black from obs_start..obs_end:
        * "observed" (solid) for dates <= init_date
        * "realized" (dashed) for dates >= init_date
      (the two curves share the init_date point)
    - Vertical black line at init_date
    """
    init_date = np.datetime64(init_date, "D")
    obs_start = np.datetime64(obs_start, "D")
    obs_end   = np.datetime64(obs_end, "D")

    # Observations (continuous ERA5 series)
    obs_ts = clim_ds["tp24"].sel(time=slice(obs_start, obs_end))
    t_obs  = obs_ts["time"].values.astype("datetime64[D]")
    y_obs  = np.asarray(obs_ts.values, dtype=float)

    mask_before = t_obs <= init_date
    mask_after  = t_obs >= init_date

    ax.plot(t_obs[mask_before], y_obs[mask_before], color="k", lw=2.0, label="observed", zorder=3)
    ax.plot(t_obs[mask_after],  y_obs[mask_after],  color="k", lw=2.0, ls="--", label="realized", zorder=3)

    # Observed precip at init_date (synthetic lead_time=0 for ensemble)
    obs_on_init = clim_ds["tp24"].sel(time=init_date, method="nearest").astype(float).item()

    # Ensemble forecast (nearest init_time)
    lead_max_days = int((obs_end - init_date) / np.timedelta64(1, "D"))
    lead_max_days = max(0, lead_max_days)

    fc = (
        model_ds["tp24"]
        .sel(init_time=init_date, method="nearest")
        .sel(lead_time=slice(1, lead_max_days))
    )
    if "hdate" in fc.coords:
        fc = fc.isel(hdate=0)

    if "number" not in fc.dims:
        raise ValueError("Example forecast plot requires ensemble forecasts with a 'number' dimension.")

    lt         = fc["lead_time"].values.astype(int)
    valid_time = (init_date + lt.astype("timedelta64[D]")).astype("datetime64[D]")

    for m in fc["number"].values:
        y = np.asarray(fc.sel(number=m).values, dtype=float)   # lead_time=1..N
        y_plot = np.concatenate([[obs_on_init], y])
        t_plot = np.concatenate([[init_date], valid_time])
        ax.plot(t_plot, y_plot, color="0.7", lw=1.0, alpha=0.8, zorder=1)

    ax.axvline(init_date, color="k", lw=1.0, ls="-")

    ax.set_title("(a) Example precipitation forecast for Bergen\n initialized on 2015-02-15")
    ax.set_ylabel("precipitation (mm/day)")
    ax.set_xlim(obs_start, obs_end)
    ax.set_ylim(0, 62)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ensemble_handle = Line2D([0], [0], color="0.7", lw=2, label="ensemble forecast")
    observed_handle = Line2D([0], [0], color="k", lw=2, label="observed")
    realized_handle = Line2D([0], [0], color="k", lw=2, ls="--", label="realized")
    ax.legend(handles=[ensemble_handle, observed_handle, realized_handle], frameon=False, loc="upper left")


# =========================
# Panel (b): Skill (MSESS/CRPSS) with optional DOY climatology
# =========================
def _get_verification_doy(template_da):
    if "valid_time" in template_da.coords:
        vt = template_da["valid_time"]
        if np.issubdtype(vt.dtype, np.datetime64):
            return vt.dt.dayofyear

    if ("init_time" in template_da.coords) and ("lead_time" in template_da.coords):
        it = template_da["init_time"]
        lt = template_da["lead_time"]
        if np.issubdtype(it.dtype, np.datetime64):
            lt_td = lt.astype("timedelta64[D]")
            vt = it + lt_td
            return vt.dt.dayofyear

    if "hdate" in template_da.coords:
        hd = template_da["hdate"]
        if np.issubdtype(hd.dtype, np.datetime64):
            return hd.dt.dayofyear
        if np.issubdtype(hd.dtype, np.integer) or np.issubdtype(hd.dtype, np.floating):
            return hd.astype(int)

    raise ValueError(
        "Could not infer verification day-of-year. Need one of: "
        "valid_time(datetime), init_time(datetime)+lead_time(days), or hdate(datetime/DOY)."
    )


def _climatology_mean_like(clim_ts, template_da):
    clim_mean = clim_ts.mean(dim="time", skipna=True)
    return xr.full_like(template_da, float(clim_mean.values))


def _climatology_mean_doy_like(clim_ts, template_da):
    clim_by_doy = clim_ts.groupby("time.dayofyear").mean("time", skipna=True)
    doy = _get_verification_doy(template_da)
    return clim_by_doy.sel(dayofyear=doy).broadcast_like(template_da)


def _make_climatology_ensemble_alltime(clim_ts, template_da, member_dim="clim_member"):
    n = clim_ts.sizes["time"]
    ens = clim_ts.assign_coords({member_dim: ("time", np.arange(n))})
    ens = ens.swap_dims({"time": member_dim}).drop_vars("time")
    for d in template_da.dims:
        if d not in ens.dims:
            ens = ens.expand_dims({d: template_da[d]})
    return ens, member_dim


def _make_climatology_ensemble_doy(clim_ts, template_da, member_dim="clim_member"):
    doy_index = clim_ts["time"].dt.dayofyear.values
    values = clim_ts.values

    doy_list = np.arange(1, 367)
    per_doy_vals, max_n = [], 0
    for d in doy_list:
        v = values[doy_index == d]
        v = v[np.isfinite(v)]
        per_doy_vals.append(v)
        max_n = max(max_n, v.size)

    padded = np.full((366, max_n), np.nan, dtype=float)
    for i, v in enumerate(per_doy_vals):
        if v.size > 0:
            padded[i, : v.size] = v

    clim_doy_ens = xr.DataArray(
        padded,
        dims=["dayofyear", member_dim],
        coords={"dayofyear": doy_list, member_dim: np.arange(max_n)},
        name=clim_ts.name,
    )

    doy = _get_verification_doy(template_da)
    ens_selected = clim_doy_ens.sel(dayofyear=doy)
    for d in template_da.dims:
        if d not in ens_selected.dims:
            ens_selected = ens_selected.expand_dims({d: template_da[d]})
    return ens_selected, member_dim


def compute_msess(model_ds, obs_ds, clim_ds, lead_min, lead_max, use_doy=False, eps_denom=1e-12):
    model_da = model_ds["tp24"].sel(lead_time=slice(lead_min, lead_max))
    obs_da   = obs_ds["tp24"].sel(lead_time=slice(lead_min, lead_max))
    clim_ts  = clim_ds["tp24"]

    dim_agg = _dims_to_aggregate(model_da)
    fcst_det = model_da.mean(dim="number") if "number" in model_da.dims else model_da
    ref_det  = _climatology_mean_doy_like(clim_ts, obs_da) if use_doy else _climatology_mean_like(clim_ts, obs_da)

    mse_fcst = xs.mse(fcst_det, obs_da, dim=dim_agg)
    mse_ref  = xs.mse(ref_det,  obs_da, dim=dim_agg)

    mse_ref = mse_ref.where(np.abs(mse_ref) > eps_denom, other=np.nan)
    return 1.0 - (mse_fcst / mse_ref)


def compute_crpss(model_ds, obs_ds, clim_ds, lead_min, lead_max, use_doy=False, eps_denom=1e-12):
    model_da = model_ds["tp24"].sel(lead_time=slice(lead_min, lead_max))
    obs_da   = obs_ds["tp24"].sel(lead_time=slice(lead_min, lead_max))
    clim_ts  = clim_ds["tp24"]

    if "number" not in model_da.dims:
        raise ValueError("CRPSS requires ensemble forecasts: missing 'number' dimension in model data.")

    dim_agg = _dims_to_aggregate(model_da)
    crps_fcst = xs.crps_ensemble(obs_da, model_da, member_dim="number", dim=dim_agg)

    if use_doy:
        clim_ens, clim_member_dim = _make_climatology_ensemble_doy(clim_ts, obs_da, member_dim="clim_member")
    else:
        clim_ens, clim_member_dim = _make_climatology_ensemble_alltime(clim_ts, obs_da, member_dim="clim_member")

    crps_ref = xs.crps_ensemble(obs_da, clim_ens, member_dim=clim_member_dim, dim=dim_agg)
    crps_ref = crps_ref.where(np.abs(crps_ref) > eps_denom, other=np.nan)
    return 1.0 - (crps_fcst / crps_ref)


def plot_panel_skill(ax, skill_by_city, lead_min, lead_max, vline_day):
    first_city = cities[0]
    lead_times = skill_by_city[first_city]["msess"].lead_time.values

    for city in cities:
        c = colors[city]
        ax.plot(lead_times, skill_by_city[city]["msess"].values, color=c, ls="--", lw=2)
        ax.plot(lead_times, skill_by_city[city]["crpss"].values, color=c, ls="-",  lw=2)

    ax.axvline(vline_day, color="k", lw=1.0)

    ax.set_title("(b) Precipitation forecast skill")
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("Skill score")
    ax.set_xlim(lead_min, lead_max)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(lead_times)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    city_handles = [
        Line2D([0], [0], color=colors["bergen"], lw=3, label="Bergen"),
        Line2D([0], [0], color=colors["oslo"],   lw=3, label="Oslo"),
    ]
    style_handles = [
        Line2D([0], [0], color="k", lw=1.5, ls="--", label="MSESS"),
        Line2D([0], [0], color="k", lw=1.5, ls="-",  label="CRPSS"),
    ]
    ax.legend(handles=city_handles + style_handles, frameon=False, fontsize=11, loc="best", ncol=2)


# =========================
# Panel (c): Reliability regression envelope
# =========================
def compute_prob_forecast_and_obs_event(model_ds, obs_ds, lead_days, threshold):
    model_da = model_ds["tp24"].sel(lead_time=lead_days)
    obs_da   = obs_ds["tp24"].sel(lead_time=lead_days)

    if "number" not in model_da.dims:
        raise ValueError("Model data has no 'number' ensemble dimension.")

    prob_fcst = (model_da > threshold).mean(dim="number")
    obs_event = (obs_da > threshold)
    dims_sample = _dims_to_aggregate(model_da) + ["lead_time"]
    return prob_fcst, obs_event, dims_sample


def compute_reliability_and_counts(prob_fcst, obs_event, dims_sample, n_bins):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[-1] = 1.0 + 1e-8
    p_centers = 0.5 * (edges[:-1] + edges[1:])

    rel_da = xs.reliability(
        obs_event.astype(int),
        prob_fcst,
        dim=dims_sample,
        probability_bin_edges=edges,
    )
    rel = np.asarray(rel_da, dtype=float)

    stacked = prob_fcst.stack(sample=dims_sample).values
    stacked = stacked[np.isfinite(stacked)]
    counts, _ = np.histogram(stacked, bins=edges)
    return p_centers, rel, counts


def weighted_linear_fit(x, y, w):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(y) & (w > 0)
    if m.sum() < 2:
        return np.nan, np.nan
    return np.polyfit(x[m], y[m], deg=1, w=w[m])


def plot_panel_reliability_envelope(ax, fits_by_city, thresholds, lead_days):
    xx = np.linspace(0, 1, 401)
    ax.plot([0, 1], [0, 1], "--", color="k", lw=1.5)

    for city in cities:
        c = colors[city]
        ys = []
        for thr in thresholds:
            a, b = fits_by_city[city].get(thr, (np.nan, np.nan))
            if np.isfinite(a) and np.isfinite(b):
                ys.append(a * xx + b)
        if len(ys) == 0:
            continue

        Y = np.vstack(ys)
        ax.fill_between(xx, np.nanmin(Y, axis=0), np.nanmax(Y, axis=0), color=c, alpha=0.75)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("(c) Forecast reliability for a range of precipitation\n thresholds")


# =========================
# Panel (d): Aggregated QQ plot (axes swapped)
# =========================
def compute_aggregated_quantiles(model_ds, obs_ds, lead_min, lead_max, q_levels):
    model_da = model_ds["tp24"].sel(lead_time=slice(lead_min, lead_max))
    obs_da   = obs_ds["tp24"].sel(lead_time=slice(lead_min, lead_max))

    if "number" not in model_da.dims:
        raise ValueError("Model data has no 'number' ensemble dimension.")

    if "hdate" in model_da.coords:
        model_dims = ["init_time", "number", "lead_time", "hdate"]
        obs_dims   = ["init_time", "lead_time", "hdate"]
    else:
        model_dims = ["init_time", "number", "lead_time"]
        obs_dims   = ["init_time", "lead_time"]

    model_q = model_da.quantile(q_levels, dim=model_dims, skipna=True)
    obs_q   = obs_da.quantile(q_levels,   dim=obs_dims,   skipna=True)
    return obs_q, model_q


def plot_panel_qq(ax, qq_by_city):
    all_vals = []
    for city in cities:
        obs_q, mod_q = qq_by_city[city]
        all_vals.append(obs_q.values)
        all_vals.append(mod_q.values)

    all_vals = np.concatenate(all_vals)
    finite = all_vals[np.isfinite(all_vals)]
    vmin = float(finite.min())
    vmax = float(finite.max())

    for city in reversed(cities):
        obs_q, mod_q = qq_by_city[city]
        ax.plot(
            mod_q.values, obs_q.values,
            linestyle="",
            marker="o",
            markersize=6,
            markerfacecolor=colors[city],
            markeredgecolor=colors[city],
            alpha=0.75,
        )

    ax.plot([vmin, vmax], [vmin, vmax], "--", color="k", lw=1.5)
    ax.set_xlim(vmin, 45)
    ax.set_ylim(vmin, 45)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Forecast quantiles (mm/day)")
    ax.set_ylabel("Observed quantiles (mm/day)")
    ax.set_title("(d) Precipitation forecast bias")


# =========================
# Main
# =========================
if __name__ == "__main__":

    # Load data per city
    data = {}
    for city in cities:
        model_ds, obs_ds, clim_ds = load_model_obs_clim(city)
        data[city] = {"model": model_ds, "obs": obs_ds, "clim": clim_ds}

    # Panel (b): skill
    skill_by_city = {}
    for city in cities:
        model_ds = data[city]["model"]
        obs_ds   = data[city]["obs"]
        clim_ds  = data[city]["clim"]

        msess = compute_msess(model_ds, obs_ds, clim_ds, lead_min_skill, lead_max_skill,
                              use_doy=use_doy_climatology, eps_denom=eps_denom)
        crpss = compute_crpss(model_ds, obs_ds, clim_ds, lead_min_skill, lead_max_skill,
                              use_doy=use_doy_climatology, eps_denom=eps_denom)
        skill_by_city[city] = {"msess": msess, "crpss": crpss}

    # Panel (c): reliability fits
    fits_by_city = {}
    for city in cities:
        model_ds = data[city]["model"]
        obs_ds   = data[city]["obs"]

        fits_by_thr = {}
        for thr in thresholds_mm_day:
            prob_fcst, obs_event, dims_sample = compute_prob_forecast_and_obs_event(
                model_ds, obs_ds, lead_days_rel, threshold=thr
            )
            p, rel, counts = compute_reliability_and_counts(
                prob_fcst, obs_event, dims_sample, n_bins=n_bins_rel
            )
            a, b = weighted_linear_fit(p, rel, counts)
            fits_by_thr[thr] = (a, b)
        fits_by_city[city] = fits_by_thr

    # Panel (d): QQ
    q_levels = quantile_levels(n=n_quantiles, clip=clip_endpoints)
    qq_by_city = {}
    for city in cities:
        model_ds = data[city]["model"]
        obs_ds   = data[city]["obs"]
        obs_q, model_q = compute_aggregated_quantiles(model_ds, obs_ds, lead_min_qq, lead_max_qq, q_levels)
        qq_by_city[city] = (obs_q, model_q)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    for ax in [ax_a, ax_b, ax_c, ax_d]:
        ax.set_box_aspect(1)

    plot_panel_example_forecast(
        ax_a,
        data[example_city]["model"],
        data[example_city]["clim"],
        init_date=example_init_date,
        obs_start=example_obs_start,
        obs_end=example_obs_end,
    )
    plot_panel_skill(ax_b, skill_by_city, lead_min_skill, lead_max_skill, vline_day=vline_day)
    plot_panel_reliability_envelope(ax_c, fits_by_city, thresholds_mm_day, lead_days_rel)
    plot_panel_qq(ax_d, qq_by_city)

    if write2file:
        os.makedirs(path_out, exist_ok=True)
        outpath = os.path.join(path_out, filename_out)
        fig.savefig(outpath)
        print(f"Saved: {outpath}")

    plt.show()
