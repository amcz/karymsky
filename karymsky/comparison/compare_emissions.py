import numpy as np
import xarray as xr
from scipy.stats import wasserstein_distance

def feature_differences(features: xr.Dataset, ref: str):
    """
    Compute differences relative to a reference estimate.
    """

    diffs = features - features.sel(dim_0=ref)
    return diffs




def compare_emissions(emissions: dict):
    """
    emissions: dict[str, xr.DataArray]
        e.g. {"method_A": E1, "method_B": E2}
    """

    features = {name: emission_features(E)
                for name, E in emissions.items()}

    return xr.Dataset.from_dict(features)




def emission_features(E: xr.DataArray):
    """
    Extract physically meaningful features from E(time, height).

    Returns a dict of scalar features.
    """

    # Total emitted mass
    total_mass = E.sum(dim=("time", "height")).item()

    if total_mass <= 0:
        raise ValueError("Total emitted mass is zero or negative.")

    # Normalize to probability distribution
    P = E / total_mass

    # --- Time features ---
    t = E["time"]
    Et = E.sum(dim="height")

    t_mean = (Et * t).sum() / Et.sum()
    t_var = (Et * (t - t_mean) ** 2).sum() / Et.sum()
    t_std = np.sqrt(t_var)

    # cumulative mass fraction in time
    Et_cum = Et.cumsum("time") / Et.sum()

    t_10 = t.where(Et_cum >= 0.10, drop=True)[0]
    t_90 = t.where(Et_cum >= 0.90, drop=True)[0]

    # --- Height features ---
    z = E["height"]
    Ez = E.sum(dim="time")

    z_mean = (Ez * z).sum() / Ez.sum()
    z_var = (Ez * (z - z_mean) ** 2).sum() / Ez.sum()
    z_std = np.sqrt(z_var)

    Ez_cum = Ez.cumsum("height") / Ez.sum()

    z_10 = z.where(Ez_cum >= 0.10, drop=True)[0]
    z_90 = z.where(Ez_cum >= 0.90, drop=True)[0]

    # --- Peak features ---
    peak_idx = E.argmax(dim=("time", "height"))
    peak_time = E["time"].isel(time=peak_idx["time"])
    peak_height = E["height"].isel(height=peak_idx["height"])

    return {
        "total_mass": total_mass,

        "time_mean": t_mean.item(),
        "time_std": t_std.item(),
        "time_10": t_10.item(),
        "time_90": t_90.item(),
        "time_duration_10_90": (t_90 - t_10).item(),

        "height_mean": z_mean.item(),
        "height_std": z_std.item(),
        "height_10": z_10.item(),
        "height_90": z_90.item(),
        "height_IQR_10_90": (z_90 - z_10).item(),

        "peak_time": peak_time.item(),
        "peak_height": peak_height.item(),
    }


def _normalized_1d(dist: xr.DataArray):
    total = dist.sum()
    if total <= 0:
        raise ValueError("Distribution has zero total mass.")
    return dist / total



def wasserstein_time_height(E1: xr.DataArray, E2: xr.DataArray):
    """
    Compute 1D Wasserstein distances between two emission estimates
    in time and height.

    Returns distances with physical units.
    """

    # --- Time marginal ---
    Et1 = _normalized_1d(E1.sum(dim="height"))
    Et2 = _normalized_1d(E2.sum(dim="height"))

    t = E1["time"].values.astype("datetime64[s]").astype(float)

    W_time = wasserstein_distance(
        t, t,
        u_weights=Et1.values,
        v_weights=Et2.values,
    )

    # --- Height marginal ---
    Ez1 = _normalized_1d(E1.sum(dim="time"))
    Ez2 = _normalized_1d(E2.sum(dim="time"))

    z = E1["height"].values.astype(float)

    W_height = wasserstein_distance(
        z, z,
        u_weights=Ez1.values,
        v_weights=Ez2.values,
    )

    return {
        "wasserstein_time": W_time,      # seconds
        "wasserstein_height": W_height,  # height units
    }




def pairwise_wasserstein(emissions: dict):
    """
    Compute pairwise Wasserstein distances between multiple estimates.

    Returns an xarray.Dataset.
    """

    names = list(emissions.keys())
    n = len(names)

    Wt = np.zeros((n, n))
    Wz = np.zeros((n, n))

    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i >= j:
                continue

            W = wasserstein_time_height(
                emissions[ni],
                emissions[nj]
            )

            Wt[i, j] = Wt[j, i] = W["wasserstein_time"]
            Wz[i, j] = Wz[j, i] = W["wasserstein_height"]

    return xr.Dataset(
        data_vars=dict(
            wasserstein_time=(("source", "target"), Wt),
            wasserstein_height=(("source", "target"), Wz),
        ),
        coords=dict(
            source=names,
            target=names,
        ),
    )



