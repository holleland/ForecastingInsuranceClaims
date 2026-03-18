"""
Calculates the expected daily cost of different claim forecasts in the paper.
Plots the potential economic value (PEV) for each forecast relative to the unconditional forecast.
Figure 06 of the paper.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# =============================================================================
# User input
# =============================================================================
data_type        = "toy"  # "real" or "toy"
areas            = ["bergen", "oslo"]
path_in          = '/your_path_for_toy_claim_predictions/'
path_out_fig     = '/your_path_for_output_figures/'
cost_loss_ratios = np.arange(0.0, 1.01, 0.01)
delta_x          = 0.0 # Panel (c) horizontal separation between Bergen (circles) and Oslo (triangles) 
figname_out      = f'{path_out_fig}fig_06.pdf'
write2file       = True


# =============================================================================
# I/O helpers
# =============================================================================
def load_data(path_in, area, data_type):
    """
    Load probabilistic claim forecasts + truth for one area from NetCDF.
    Keeps only selected forecast models. Adds a dummy 'seasonal' equal to 'unconditional'.
    """
    keep_models = [
        "observed", "seasonal",
        "observed-forecast", "observed-forecast-gam",
        "saturated", "stepwise", "lasso",
        "xgboost", "cnn",
    ]

    filename_in = f"{path_in}{data_type}.claim.predictions.{area}.2020-2021.nc"
    da = xr.open_dataset(filename_in)[area]  # saved as DataArray with name=area

    # Add seasonal model if missing (dummy = unconditional)
    if "seasonal" not in da.model.values:
        seasonal = da.sel(model="unconditional").copy().expand_dims(model=["seasonal"])
        da = xr.concat([da, seasonal], dim="model")

    # Forecast: keep chosen models, in the desired order
    forecast_models = [m for m in keep_models if m in da.model.values]
    forecast = da.sel(model=forecast_models)

    # Observation + reference
    observation = da.sel(model="truth")
    reference   = da.sel(model="unconditional")
    
    return forecast, observation, reference


# =============================================================================
# Core computations
# =============================================================================
def compute_cost_loss_value(forecast, observation, reference, cost_loss_ratios):
    """
    Compute expected cost per day as a function of cost/loss ratio c, and PEV.

    Decision rule:
      - Act if p(event) >= c
      - Cost per action = c
      - Loss for missed event = 1
    """
    clr = cost_loss_ratios
    models = forecast["model"].values

    forecast_cost = np.zeros((models.size, clr.size))
    reference_cost = np.zeros(clr.size)

    for i, c in enumerate(clr):
        # Reference (unconditional)
        act_ref = (reference >= c)
        reference_cost[i] = c * act_ref.mean().item() + ((observation == 1) & ~act_ref).mean().item()

        # Each model
        for j, model in enumerate(models):
            act = (forecast.sel(model=model) >= c)
            forecast_cost[j, i] = c * act.mean().item() + ((observation == 1) & ~act).mean().item()

    # Convert to xarray
    forecast_cost_da = xr.DataArray(
        forecast_cost,
        dims=["model", "costlossratio"],
        coords={"model": models, "costlossratio": clr},
        name="forecast_cost",
    )

    reference_cost_da = xr.DataArray(
        reference_cost,
        dims=["costlossratio"],
        coords={"costlossratio": clr},
        name="reference_cost",
    )

    # PEV relative to unconditional
    pov = 1.0 - (forecast_cost_da / reference_cost_da)
    pov = pov.where(reference_cost_da > 0)  # avoid division by zero
    pov.name = "potential_value"

    return forecast_cost_da, reference_cost_da, pov


# =============================================================================
# Plotting
# =============================================================================
def plot_pev_figure(
    pov_bergen, pov_oslo,
    forecast_cost_bergen, forecast_cost_oslo,
    areas, delta_x,
    write2file, figname_out
):
    """
    Make a 3-panel figure:
      a) PEV curves for Bergen
      b) PEV curves for Oslo
      c) Max PEV vs cost/loss ratio at max PEV (Bergen circles, Oslo triangles)
    """
    import matplotlib as mpl

    # Style: ggplot2 theme_classic-ish
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
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    }

    with mpl.rc_context(classic_rc):
        fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(14, 4))

        # Plot only up to 0.5 on x-axis in panels (a) and (b)
        clr_all = forecast_cost_bergen["costlossratio"]
        clr_mask = clr_all <= 0.5
        clr = clr_all.where(clr_mask, drop=True)

        # Use consistent model list + consistent colors
        model_names = [str(m) for m in forecast_cost_bergen["model"].values]
        palette = plt.get_cmap("tab10").colors
        model_colors = {name: palette[i % len(palette)] for i, name in enumerate(model_names)}

        # ---- Panel (a): Bergen curves
        legend_lines, legend_labels = [], []
        for name in model_names:
            ln, = ax[0].plot(
                clr,
                pov_bergen.sel(model=name).where(clr_mask, drop=True),
                color=model_colors[name],
                linewidth=1.6,
                label=name,
            )
            legend_lines.append(ln)
            legend_labels.append(name)

        ax[0].axhline(0, color="black", linewidth=1.0)
        ax[0].set_title("a) Bergen")
        ax[0].set_xlabel("Cost–loss ratio")
        ax[0].set_ylabel("Potential economic value")
        ax[0].set_xlim(0, 0.5)
        ax[0].set_ylim(-0.3, 0.6)
        ax[0].legend(
            handles=legend_lines,
            labels=legend_labels,
            loc="best",
            ncol=2,
            frameon=False,
            handlelength=1.6,
            columnspacing=1.0,
            borderaxespad=0.3,
            prop={"family": "monospace"},
        )

        # ---- Panel (b): Oslo curves
        for name in model_names:
            ax[1].plot(
                clr,
                pov_oslo.sel(model=name).where(clr_mask, drop=True),
                color=model_colors[name],
                linewidth=1.6,
            )

        ax[1].axhline(0, color="black", linewidth=1.0)
        ax[1].set_title("b) Oslo")
        ax[1].set_xlabel("Cost–loss ratio")
        ax[1].set_ylabel("Potential economic value")
        ax[1].set_xlim(0, 0.5)
        ax[1].set_ylim(-0.3, 0.6)

        # ---- Panel (c): max PEV points per model (Bergen vs Oslo)
        for name in model_names:
            color = model_colors[name]

            pev_b = pov_bergen.sel(model=name).where(clr_mask, drop=True)
            max_pev_b = float(pev_b.max().item())
            clr_b = float(pev_b["costlossratio"].sel(costlossratio=pev_b.idxmax()).item())

            pev_o = pov_oslo.sel(model=name).where(clr_mask, drop=True)
            max_pev_o = float(pev_o.max().item())
            clr_o = float(pev_o["costlossratio"].sel(costlossratio=pev_o.idxmax()).item())

            # Offset x slightly to separate markers visually
            ax[2].scatter(clr_b - delta_x, max_pev_b, marker="o", s=45,
                          facecolors="none", edgecolors=color, linewidths=1.5)
            ax[2].scatter(clr_o + delta_x, max_pev_o, marker="^", s=45,
                          facecolors="none", edgecolors=color, linewidths=1.5)

        ax[2].axhline(0, color="black", linewidth=1.0)
        ax[2].set_title("c) Bergen (circles) & Oslo (triangles)")
        ax[2].set_xlabel("Cost–loss ratio at maximum potential economic value")
        ax[2].set_ylabel("Maximum potential economic value")
        ax[2].set_xlim(0, 0.08)
        ax[2].set_ylim(-0.3, 0.6)

        fig.tight_layout()

        if write2file:
            plt.savefig(figname_out, bbox_inches="tight")

        plt.show()


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":

    # 1) Load forecasts, truth, and reference
    forecast_b, obs_b, ref_b = load_data(path_in, areas[0], data_type)
    forecast_o, obs_o, ref_o = load_data(path_in, areas[1], data_type)

    # 2) Compute cost/loss curves + potential value
    fc_b, rc_b, pov_b = compute_cost_loss_value(forecast_b, obs_b, ref_b, cost_loss_ratios)
    fc_o, rc_o, pov_o = compute_cost_loss_value(forecast_o, obs_o, ref_o, cost_loss_ratios)

    # 3) Plot figure
    plot_pev_figure(
        pov_b, pov_o,
        fc_b, fc_o,
        areas, delta_x,
        write2file, figname_out
    )
