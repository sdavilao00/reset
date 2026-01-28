# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 18:08:22 2026

@author: sdavilao

Integrated:
- Soil transport + production (Landlab TNLD)
- Failure detection every timestep for a RANGE of buffer radii
- FS computed from MEAN soil depth in each buffer (single FS per buffer)
- When failure occurs: choose the SMALLEST buffer that down-crosses FS=1
- Reset soil depth to 0 inside that triggering buffer mask
- Continue simulation to get multiple failures (recurrence intervals)

Notes:
- This runs for ONE hollow center point (first point in POINTS_SHP).
- Slope is treated as a constant scalar (avg_slope_deg). You can:
  (A) set it manually, or
  (B) provide SLOPE_RASTER and DOWNSLOPE_LINES and it will compute it (like your other code).
"""

#%% ---------------- PACKAGES ----------------
import os
import re
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import rasterio
from rasterio.features import geometry_mask
from rasterio.mask import mask as rio_mask
from osgeo import gdal

from landlab import imshowhs_grid
from landlab.components import TaylorNonLinearDiffuser
from landlab.io import read_esri_ascii, write_esri_ascii

# Optional (only used if you want auto slope from downslope line)
try:
    from rasterstats import zonal_stats
    HAS_RASTERSTATS = True
except Exception:
    HAS_RASTERSTATS = False


#%% ---------------- PATHS + CONSTANTS ----------------
BASE_DIR = os.path.join(os.getcwd())

INPUT_TIFF = 'ext9.tif'
POINTS_SHP = os.path.join(BASE_DIR, 'ext9_1.shp')       # center-of-hollow points
BUFFER_SHP = os.path.join(BASE_DIR, 'ext9_buff.shp')  # written by create_buffer_from_points()

# OPTIONAL: if you want slope computed like your other code
# (provide these paths, otherwise set avg_slope_deg manually below)
SLOPE_RASTER = r"C:\Users\sdavilao\Documents\newcodesoil\slope_smooth_m_warp.tif"
DOWNSLOPE_LINES = r"C:\Users\sdavilao\Documents\newcodesoil\polylines\ext9_lines.shp"
TARGET_POINT_ID_FOR_SLOPE = None  # if your line shp has multiple points; set integer Point_ID or leave None (uses first)

# Output dirs
OUT_DIR    = os.path.join(BASE_DIR, r"simulation_results\new\test")
OUT_DIRpng = os.path.join(OUT_DIR, "PNGs")
OUT_DIRtiff= os.path.join(OUT_DIR, "GeoTIFFs")
OUT_DIRasc = os.path.join(OUT_DIR, "ASCs")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIRpng, exist_ok=True)
os.makedirs(OUT_DIRtiff, exist_ok=True)
os.makedirs(OUT_DIRasc, exist_ok=True)

# Unit conversions
ft2mUS  = 1200 / 3937
ft2mInt = 0.3048


#%% ---------------- USER RUN PARAMETERS ----------------
# Soil transport / production
K  = 0.0042   # Diffusion coefficient
Sc = 1.25     # Critical slope gradient
pr = 2000     # Ratio of production (example value)
ps = 1600     # Ratio of soil loss (example value)
P0 = 0.0003   # initial soil production rate
h0 = 0.5      # soil production depth scale
dt = 50
target_time = 10000

# Failure model constants (your values)
pw = 1000
g  = 9.81
phi_deg = 41
m_sat = 1.0
l = 10
w = 6.7
C0 = 1920
j  = 0.8

# Buffer radii to test EACH timestep (meters)
buffer_sizes = [3, 4, 5, 6, 7, 8, 9]

# Re-arm threshold to avoid double-counting if FS jitters
REARM_FS = 1.05

# Slope handling:
# If you don't provide SLOPE_RASTER + DOWNSLOPE_LINES, set this manually:
avg_slope_deg_manual = 35.0  # <-- change me if you want


#%% ---------------- HELPERS ----------------
def create_buffer_from_points(input_points_path, output_buffer_path, buffer_distance, target_crs=None):
    gdf = gpd.read_file(input_points_path)

    if gdf.crs is None:
        raise ValueError(f"Shapefile {input_points_path} has no CRS. Define it first.")

    if target_crs:
        if gdf.crs != target_crs:
            print(f"Reprojecting points from {gdf.crs} -> {target_crs}")
            gdf = gdf.to_crs(target_crs)
    else:
        print("⚠️ No target CRS provided; using original shapefile CRS.")

    buffer_geom = gdf.buffer(buffer_distance)
    buffered = gdf.copy()
    buffered["geometry"] = buffer_geom

    buffered.set_crs(gdf.crs, inplace=True)
    buffered.to_file(output_buffer_path)

    print(f"Buffer created at {output_buffer_path}")
    print("Buffer bounds:", buffered.total_bounds)
    return output_buffer_path


def tiff_to_asc(in_path, out_path):
    with rasterio.open(in_path) as src:
        XYZunit = src.crs.linear_units if src.crs else "meters"
        mean_res = float(np.mean(src.res))
    gdal.Translate(out_path, in_path, format='AAIGrid', xRes=mean_res, yRes=mean_res)
    print(f"Converted GeoTIFF -> ASC with spacing {mean_res} ({XYZunit})")
    return mean_res, XYZunit


def asc_to_tiff(asc_path, tiff_path, meta):
    data = np.loadtxt(asc_path, skiprows=10)
    meta2 = meta.copy()
    meta2.update(dtype=rasterio.float32, count=1, compress='deflate')
    with rasterio.open(tiff_path, 'w', **meta2) as dst:
        dst.write(data.astype(rasterio.float32), 1)
    print(f"Saved GeoTIFF: {tiff_path}")


def save_as_tiff(data, filename, meta, grid_shape):
    if data.ndim == 1:
        data = data.reshape(grid_shape)

    meta2 = meta.copy()
    meta2.update(dtype=rasterio.float32, count=1, compress='deflate')

    with rasterio.open(filename, 'w', **meta2) as dst:
        dst.write(np.flipud(data.astype(rasterio.float32)), 1)

    print(f"Saved TIFF: {filename}")


def plot_save(grid, z, basefilename, time, K, mean_res, XYZunit):
    plt.figure(figsize=(6, 5.25))
    imshowhs_grid(grid, z, plot_type="Hillshade")
    plt.title(f"{basefilename} Time {time} yrs (K={K})")
    plt.tight_layout()
    png_path = os.path.join(OUT_DIRpng, f"{basefilename}_{time}yrs_K{K}.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    asc_path = os.path.join(OUT_DIRasc, f"{basefilename}_{time}yrs_K{K}.asc")
    write_esri_ascii(asc_path, grid, names=['topographic__elevation'], clobber=True)
    return asc_path


def init_simulation(asc_file, K, Sc, XYZunit=None):
    grid, _ = read_esri_ascii(asc_file, name='topographic__elevation')
    grid.set_closed_boundaries_at_grid_edges(False, False, False, False)

    # initialize soil depth everywhere (will be modified by failure resets)
    soil_depth = np.full(grid.number_of_nodes, 0.5)
    grid.add_field('soil__depth', soil_depth, at='node')

    # unit-aware diffusivity
    if XYZunit is None or 'meter' in str(XYZunit).lower() or 'metre' in str(XYZunit).lower():
        Kc = K
    elif 'foot' in str(XYZunit).lower():
        # choose US survey feet conversion by default
        Kc = K / (ft2mUS ** 2) if "US" in str(XYZunit) else K / (ft2mInt ** 2)
    else:
        raise RuntimeError("Unsupported unit type for K conversion.")

    tnld = TaylorNonLinearDiffuser(
        grid,
        linear_diffusivity=Kc,
        slope_crit=Sc,
        dynamic_dt=True,
        nterms=2,
        if_unstable="pass"
    )
    return grid, tnld


def buffer_geom_to_node_mask(buffer_geom, dem_path, grid_shape):
    """Convert a shapely geometry to a boolean node mask aligned with the DEM/grid."""
    with rasterio.open(dem_path) as src:
        transform = src.transform
        out_shape = src.read(1).shape

    mask_r = geometry_mask([buffer_geom], transform=transform, invert=True, out_shape=out_shape)

    node_mask = np.flipud(mask_r).flatten()

    if node_mask.size != (grid_shape[0] * grid_shape[1]):
        raise ValueError("Mask size != grid size. Check DEM vs grid alignment.")
    return node_mask


def fs_from_mean_depth(z, slope_deg,
                       C0=C0, j=j, phi_deg=phi_deg,
                       m=m_sat, pw=pw, ps=ps, g=g,
                       l=l, w=w):
    """Your FS equation, but computed from mean depth (z) and constant slope (deg)."""
    if z is None or np.isnan(z) or z <= 0 or np.isnan(slope_deg):
        return np.nan

    phi = np.deg2rad(phi_deg)
    theta = np.deg2rad(slope_deg)

    yw = g * pw
    ys = g * ps

    Crb = C0 * np.exp(-z * j)
    Crl = (C0 / (j * z)) * (1 - np.exp(-z * j))

    K0 = 1 - np.sin(theta)
    Kp = np.tan(np.deg2rad(45) + phi/2)**2
    Ka = np.tan(np.deg2rad(45) - phi/2)**2

    Frb   = (Crb + (np.cos(theta)**2) * z * (ys - yw*m) * np.tan(phi)) * l * w
    Frc   = (Crl + (K0 * 0.5 * z * (ys - yw*(m**2)) * np.tan(phi))) * (np.cos(theta) * z * l * 2)
    Frddu = (Kp - Ka) * 0.5 * (z**2) * (ys - yw*(m**2)) * w
    Fdc   = (np.sin(theta)) * (np.cos(theta)) * z * ys * l * w

    return (Frb + Frc + Frddu) / Fdc if Fdc != 0 else np.nan


def compute_avg_slope_from_line_bulletproof(
    slope_raster_path,
    downslope_lines_path,
    dem_crs,
    point_geom_for_fallback=None,
    target_point_id=None,
    buffer_try_m=(0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0),
    fallback_point_buffer_m=8.0,
    verbose=True
):
    """
    Bulletproof mean slope from a slope raster using a downslope LineString.

    Strategy:
    1) Reproject line to dem_crs
    2) Buffer the line by progressively larger widths until zonal_stats returns pixels
    3) Use raster's true nodata
    4) If still no pixels, fall back to point-buffer mean slope (requires point_geom_for_fallback)

    Returns: avg_slope_deg (float)
    """
    if not HAS_RASTERSTATS:
        raise RuntimeError("rasterstats not available; run: pip install rasterstats")

    # Read slope raster metadata
    with rasterio.open(slope_raster_path) as src:
        nodata = src.nodata
        rb = src.bounds
        rcrs = src.crs

    # Read downslope lines
    line_gdf = gpd.read_file(downslope_lines_path)

    # Reproject to raster CRS (best practice: use slope raster CRS)
    # If slope raster CRS differs from dem_crs, trust the raster CRS for sampling.
    target_crs = rcrs if rcrs is not None else dem_crs
    if line_gdf.crs != target_crs:
        line_gdf = line_gdf.to_crs(target_crs)

    # Rename id -> Point_ID if needed
    if 'id' in line_gdf.columns and 'Point_ID' not in line_gdf.columns:
        line_gdf = line_gdf.rename(columns={'id': 'Point_ID'})

    # Choose geometry
    if target_point_id is not None and 'Point_ID' in line_gdf.columns:
        sel = line_gdf[line_gdf['Point_ID'] == target_point_id]
        if len(sel) == 0:
            raise ValueError(f"No downslope line found for Point_ID={target_point_id}")
        geom_line = sel.geometry.iloc[0]
    else:
        geom_line = line_gdf.geometry.iloc[0]

    # Debug bounds
    if verbose:
        lb = geom_line.bounds
        print(f"[SLOPE] slope raster CRS: {target_crs}")
        print(f"[SLOPE] slope raster bounds: {(rb.left, rb.bottom, rb.right, rb.top)}")
        print(f"[SLOPE] line bounds (unbuffered): {lb}")
        print(f"[SLOPE] raster nodata: {nodata}")

    # Try buffered line widths
    for bw in buffer_try_m:
        geom = geom_line.buffer(float(bw))

        zs = zonal_stats(
            [geom],
            slope_raster_path,
            stats=["count", "mean"],
            nodata=nodata
        )

        avg = zs[0].get("mean", None)
        cnt = zs[0].get("count", 0)

        if verbose:
            print(f"[SLOPE] try buffer={bw:>4} m -> count={cnt}, mean={avg}")

        if avg is not None and cnt and cnt > 0 and np.isfinite(avg):
            if verbose:
                print(f"[SLOPE] ✅ Using buffered line width {bw} m: mean slope = {float(avg):.4f} deg from {int(cnt)} px")
            return float(avg)

    # Fallback: point-buffer slope if line sampling failed
    if point_geom_for_fallback is not None:
        # Ensure point is in target CRS
        pt = point_geom_for_fallback
        # If point is in dem_crs but target_crs differs, reproject point via GeoSeries
        if dem_crs is not None and target_crs is not None and dem_crs != target_crs:
            pt = gpd.GeoSeries([pt], crs=dem_crs).to_crs(target_crs).iloc[0]

        geom_pt = pt.buffer(float(fallback_point_buffer_m))
        zs = zonal_stats([geom_pt], slope_raster_path, stats=["count", "mean"], nodata=nodata)
        avg = zs[0].get("mean", None)
        cnt = zs[0].get("count", 0)

        if avg is not None and cnt and cnt > 0 and np.isfinite(avg):
            if verbose:
                print(f"[SLOPE] ⚠️ Line failed; fallback point-buffer {fallback_point_buffer_m} m worked: mean slope={float(avg):.4f} deg from {int(cnt)} px")
            return float(avg)

    raise RuntimeError(
        "Slope extraction failed for all buffered-line widths AND fallback.\n"
        "Likely causes:\n"
        " - geometry does not overlap slope raster extent\n"
        " - CRS mismatch (line/point vs slope raster)\n"
        " - slope raster nodata masks everything\n"
    )

#%% ---------------- MAIN SIMULATION ----------------
def run_simulation(in_tiff, K, Sc, dt, target_time, point_shapefile,
                   buffer_sizes,
                   avg_slope_deg=None,
                   slope_raster_path=None,
                   downslope_lines_path=None,
                   target_point_id_for_slope=None):

    dem_path = os.path.join(BASE_DIR, in_tiff)
    basefilename = os.path.splitext(in_tiff)[0]

    # Open DEM (CRS anchor)
    with rasterio.open(dem_path) as src:
        print(f"✅ DEM loaded: {dem_path}")
        print("   CRS:", src.crs)
        dem_crs = src.crs

    # Convert DEM to ASC and init Landlab grid
    in_asc = os.path.join(BASE_DIR, f"{basefilename}.asc")
    mean_res, XYZunit = tiff_to_asc(dem_path, in_asc)

    grid, tnld = init_simulation(in_asc, K, Sc, XYZunit)
    z_old = grid.at_node['topographic__elevation']

    # Raster meta for writing outputs
    with rasterio.open(dem_path) as src:
        meta = src.meta.copy()

    # Load point geometry (first point)
    pt_gdf = gpd.read_file(point_shapefile)
    if pt_gdf.crs != dem_crs:
        pt_gdf = pt_gdf.to_crs(dem_crs)

    if 'id' in pt_gdf.columns and 'Point_ID' not in pt_gdf.columns:
        pt_gdf = pt_gdf.rename(columns={'id': 'Point_ID'})

    point_geom = pt_gdf.geometry.iloc[0]

    # --------------------------------------------------
    # SLOPE (constant scalar for this run)
    # --------------------------------------------------
    if avg_slope_deg is None:
        if slope_raster_path and downslope_lines_path:
            avg_slope_deg = compute_avg_slope_from_line_bulletproof(
                slope_raster_path=slope_raster_path,
                downslope_lines_path=downslope_lines_path,
                dem_crs=dem_crs,
                point_geom_for_fallback=point_geom,
                target_point_id=target_point_id_for_slope,
                buffer_try_m=(0.5, 1.0, 2.0, 3.0, 5.0, 8.0),
                fallback_point_buffer_m=8.0,
                verbose=True
            )
        else:
            raise ValueError(
                "avg_slope_deg is None. Provide slope_raster_path + downslope_lines_path "
                "or pass avg_slope_deg_manual."
            )

    print(f"[SLOPE] Using constant slope = {avg_slope_deg:.3f} deg")

    # --------------------------------------------------
    # PRECOMPUTE BUFFER NODE MASKS (once)
    # --------------------------------------------------
    buffer_node_masks = {}
    for b in buffer_sizes:
        buf_geom = point_geom.buffer(b)
        buffer_node_masks[b] = buffer_geom_to_node_mask(buf_geom, dem_path, grid.shape)

    # --------------------------------------------------
    # Soil depth state + INITIAL CORE MASK
    # --------------------------------------------------
    total_soil_depth = grid.at_node['soil__depth'].copy()
    initial_soil_depth = total_soil_depth.copy()

    INITIAL_CORE_RADIUS = 8  # meters (your choice)
    if INITIAL_CORE_RADIUS not in buffer_node_masks:
        raise ValueError(
            f"INITIAL_CORE_RADIUS={INITIAL_CORE_RADIUS} must be included in buffer_sizes={buffer_sizes}"
        )

    core_mask = buffer_node_masks[INITIAL_CORE_RADIUS]
    total_soil_depth[core_mask] = 0.0
    initial_soil_depth[core_mask] = 0.0
    grid.at_node['soil__depth'] = total_soil_depth

    print(f"[INIT] Set initial soil depth = 0 inside {INITIAL_CORE_RADIUS} m hollow core")

    # Save initial hillshade/asc
    asc_path = plot_save(grid, z_old, basefilename, 0, K, mean_res, XYZunit)

    # Failure tracking
    FS_prev = {b: None for b in buffer_sizes}
    armed = True
    failure_events = []

    # Time series storage (per-buffer rows only, easiest to plot)
    ts_rows = []

    num_steps = int(target_time / dt)

    for i in range(num_steps):
        # -------- LANDLAB DIFFUSION STEP --------
        tnld.run_one_step(dt)
        time = (i + 1) * dt

        # -------- SOIL PRODUCTION --------
        production_rate = (pr / ps) * (P0 * np.exp(-total_soil_depth / h0)) * dt

        z_new = grid.at_node['topographic__elevation']
        elevation_change = z_new - z_old

        if 'foot' in str(XYZunit).lower():
            elevation_change = elevation_change * ft2mUS

        # Your erosion limiter logic
        nonzero_soil_mask = initial_soil_depth > 0.0
        erosion_exceeds = (np.abs(elevation_change) > initial_soil_depth) & nonzero_soil_mask
        if np.any(erosion_exceeds):
            z_new[erosion_exceeds] = z_old[erosion_exceeds] - total_soil_depth[erosion_exceeds]
            total_soil_depth[erosion_exceeds] = production_rate[erosion_exceeds]

        change_in_soil_depth = production_rate.copy()
        change_in_soil_depth[~erosion_exceeds] = elevation_change[~erosion_exceeds] + production_rate[~erosion_exceeds]
        total_soil_depth = np.where(erosion_exceeds, production_rate, total_soil_depth + change_in_soil_depth)

        grid.at_node['soil__depth'] = total_soil_depth
        z_old = z_new.copy()

        # ==========================
        # FAILURE CHECK (EVERY STEP)
        # ==========================
        FS_now = {}
        hbar_now = {}

        for b in buffer_sizes:
            mask_b = buffer_node_masks[b]

            # ✅ Match your original workflow: mean of positive-only depths
            vals = total_soil_depth[mask_b]
            vals = vals[np.isfinite(vals) & (vals > 0)]
            hbar = float(np.mean(vals)) if vals.size > 0 else np.nan

            hbar_now[b] = hbar
            FS_now[b] = fs_from_mean_depth(hbar, avg_slope_deg)

            # store per-buffer time series (every timestep)
            ts_rows.append({
                "Year": time,
                "Buffer_Size": b,
                "MeanDepth_m": hbar,
                "FS": FS_now[b]
            })

        # init prev on first evaluation
        if FS_prev[buffer_sizes[0]] is None:
            for b in buffer_sizes:
                FS_prev[b] = FS_now[b]
        else:
            crossed = [
                b for b in buffer_sizes
                if np.isfinite(FS_prev[b]) and np.isfinite(FS_now[b]) and (FS_prev[b] > 1.0) and (FS_now[b] <= 1.0)
            ]

            if armed and crossed:
                candidates = []
                t0 = time - dt  # previous sample time
        
                for b in crossed:
                    f0 = FS_prev[b]
                    f1 = FS_now[b]

                    # linear interpolation fraction to FS=1
                    frac = (1.0 - f0) / (f1 - f0) if (f1 != f0) else 0.0
                    frac = np.clip(frac, 0.0, 1.0)

                    t_cross = t0 + frac * dt
                    candidates.append((t_cross, b))

                t_fail, b_trig = min(candidates, key=lambda x: x[0])


                failure_events.append({
                    "Failure_Year": t_fail,
                    "Trigger_Buffer_m": b_trig,
                    "FS_Trigger": FS_now[b_trig],
                    "MeanDepth_Trigger_m": hbar_now[b_trig],
                    "MinFS_AllBuffers": float(np.nanmin(list(FS_now.values()))),
                    "Slope_deg_used": float(avg_slope_deg)
                })

                print(f"\n🔥 FAILURE at t={time} yrs | trigger buffer={b_trig} m | "
                      f"FS={FS_now[b_trig]:.3f} | hbar={hbar_now[b_trig]:.3f}")

                # RESET: reset soil depth in triggering footprint
                trig_mask = buffer_node_masks[b_trig]
                total_soil_depth[trig_mask] = 0.0
                grid.at_node['soil__depth'] = total_soil_depth

                # IMPORTANT: also reset initial_soil_depth for your erosion limiter logic
                initial_soil_depth[trig_mask] = 0.0

                armed = True
                for b in buffer_sizes:
                    FS_prev[b] = None   # force a clean restart of crossing logic next step

                # reset prev to prevent immediate re-trigger
                for b in buffer_sizes:
                    FS_prev[b] =  None


        # Optional debug every 200 yrs
        if time % 200 == 0:
            min_fs = float(np.nanmin(list(FS_now.values())))
            print(f"[DEBUG] t={time} yrs | minFS={min_fs:.3f} | hbar(8m)={hbar_now.get(8, np.nan):.3f}")

        # ==========================
        # OUTPUTS (adjust as you like)
        # ==========================
        if time % 1000 == 0:
            save_as_tiff(elevation_change, os.path.join(OUT_DIRtiff, f"{basefilename}_change_in_elevation_{time}yrs.tif"), meta, grid.shape)
            save_as_tiff(change_in_soil_depth, os.path.join(OUT_DIRtiff, f"{basefilename}_change_in_soil_depth_{time}yrs.tif"), meta, grid.shape)
            save_as_tiff(total_soil_depth, os.path.join(OUT_DIRtiff, f"{basefilename}_total_soil_depth_{time}yrs.tif"), meta, grid.shape)
            save_as_tiff(production_rate, os.path.join(OUT_DIRtiff, f"{basefilename}_production_rate_{time}yrs.tif"), meta, grid.shape)

            asc_path = plot_save(grid, z_new, basefilename, time, K, mean_res, XYZunit)
            tiff_path = os.path.join(OUT_DIRtiff, f"{basefilename}_{time}yrs_K{K}.tif")
            asc_to_tiff(asc_path, tiff_path, meta)

    # Cleanup ASC artifacts
    try:
        os.remove(in_asc)
        os.remove(in_asc.replace(".asc", ".prj"))
    except Exception:
        pass

    # Save failures + recurrence intervals
    fe = pd.DataFrame()
    if failure_events:
        fe = pd.DataFrame(failure_events).sort_values("Failure_Year").reset_index(drop=True)
        fe["RecurrenceInterval_yr"] = fe["Failure_Year"].diff()

        out_fail_csv = os.path.join(OUT_DIR, f"{basefilename}_failure_events_C{C0}_m{m_sat}.csv")
        fe.to_csv(out_fail_csv, index=False)

        print("\n✅ Failure events saved:", out_fail_csv)
        print(fe)
    else:
        print("\n⚠️ No failures detected in the simulation window.")

    # Save time series to CSV too (handy)
    ts_df = pd.DataFrame(ts_rows)
    out_ts_csv = os.path.join(OUT_DIR, f"{basefilename}_timeseries_C{C0}_m{m_sat}.csv")
    ts_df.to_csv(out_ts_csv, index=False)
    print("✅ Time series saved:", out_ts_csv)

    print("Simulation complete.")

    # Return both time series and failure events for plotting outside this function
    return ts_rows, fe

#%% ---------------- RUN ----------------
if __name__ == "__main__":

    # Choose slope source
    if SLOPE_RASTER and DOWNSLOPE_LINES:
        avg_slope_deg = None  # auto-compute from downslope line
    else:
        avg_slope_deg = avg_slope_deg_manual

    ts_rows, fe = run_simulation(
        in_tiff=INPUT_TIFF,
        K=K,
        Sc=Sc,
        dt=dt,
        target_time=target_time,
        point_shapefile=POINTS_SHP,
        buffer_sizes=buffer_sizes,
        avg_slope_deg=avg_slope_deg,
        slope_raster_path=SLOPE_RASTER,
        downslope_lines_path=DOWNSLOPE_LINES,
        target_point_id_for_slope=TARGET_POINT_ID_FOR_SLOPE
    )

#%%
ts = pd.DataFrame(ts_rows)

# Keep only rows that have Buffer_Size (per-buffer series)
tsb = ts.dropna(subset=["Buffer_Size"]).copy()
tsb["Buffer_Size"] = tsb["Buffer_Size"].astype(int)

# Failure years (if any)
fail_years = fe["Failure_Year"].values if 'fe' in locals() and not fe.empty else []

# ---- PLOT 1: Mean depth vs time ----
fig, ax = plt.subplots(figsize=(10, 5))
for b in sorted(tsb["Buffer_Size"].unique()):
    sub = tsb[tsb["Buffer_Size"] == b].sort_values("Year")
    ax.plot(sub["Year"], sub["MeanDepth_m"], lw=1.5, label=f"{b} m")

for fy in fail_years:
    ax.axvline(fy, lw=1.0, linestyle="--")

ax.set_xlabel("Time (years)")
ax.set_ylabel("Mean soil depth (m) in buffer")
ax.set_title("Soil depth through time (dashed lines = failures)")
ax.legend(title="Buffer radius", ncols=2, fontsize=8)
plt.tight_layout()
#plt.savefig(os.path.join(OUT_DIRpng, f"{basefilename}_MeanDepth_vs_Time.png"), dpi=200)
plt.show()

# ---- PLOT 2: FS vs time (with FS=1 line) ----
fig, ax = plt.subplots(figsize=(10, 5))
for b in sorted(tsb["Buffer_Size"].unique()):
    sub = tsb[tsb["Buffer_Size"] == b].sort_values("Year")
    ax.plot(sub["Year"], sub["FS"], lw=1.5, label=f"{b} m")

ax.axhline(1.0, lw=1.5, linestyle=":")  # FS=1 threshold
for fy in fail_years:
    ax.axvline(fy, lw=1.0, linestyle="--")

ax.set_xlabel("Time (years)")
ax.set_ylabel("Factor of Safety (FS)")
# ax.set_title("FS through time (dashed = failures, dotted = FS=1)")
ax.legend(title="Landslide length", ncols=2, fontsize=8)
plt.tight_layout()
#plt.savefig(os.path.join(OUT_DIRpng, f"{basefilename}_FS_vs_Time.png"), dpi=200)
plt.show()



