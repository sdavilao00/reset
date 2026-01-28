# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 23:56:29 2026

@author: sdavilao
"""

import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import glob
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from rasterstats import zonal_stats
from rasterio.plot import show

#%% ---------------- LOAD & PROCESS DATA ----------------
# Define file paths
shapefile_path = r"C:\Users\sdavilao\Documents\newcodesoil\reproj_shp\ext26_32610.shp"
dem_path = r"C:\Users\sdavilao\Documents\newcodesoil\dem_smooth_m_warp.tif"
slope_path = r"C:\Users\sdavilao\Documents\newcodesoil\slope_smooth_m_warp.tif"
downslope_lines =r"C:\Users\sdavilao\Documents\newcodesoil\polylines\ext26_lines.shp" 

# Define soil depth raster directory and naming pattern
soil_depth_dir = r"C:\Users\sdavilao\Documents\newcodesoil\simulation_results\new\GeoTIFFs\reproj_tif"
basename = "ext26"
soil_depth_pattern = os.path.join(soil_depth_dir, f"{basename}_total_soil_depth_*yrs_32610.tif")

# Load the point shapefile
gdf = gpd.read_file(shapefile_path)

line_gdf = gpd.read_file(downslope_lines)
line_gdf = line_gdf.to_crs(gdf.crs)  # Ensure same CRS

if 'id' in line_gdf.columns:
    line_gdf = line_gdf.rename(columns={'id': 'Point_ID'})
    
with rasterio.open(dem_path) as dem:
    dem_crs = dem.crs
    if gdf.crs != dem_crs:
        gdf = gdf.to_crs(dem_crs)
    if line_gdf.crs != dem_crs:
        line_gdf = line_gdf.to_crs(dem_crs)


# Rename 'id' column to 'Point_ID' if it exists
if 'id' in gdf.columns:
    gdf = gdf.rename(columns={'id': 'Point_ID'})

# Ensure the shapefile and raster CRS match
with rasterio.open(dem_path) as dem:
    dem_crs = dem.crs  
    if gdf.crs != dem_crs:
        gdf = gdf.to_crs(dem_crs)  

# Get all soil depth .tif files and sort them by year
soil_depth_files = sorted(glob.glob(soil_depth_pattern))

# Define buffer sizes to test
buffer_sizes = [3, 4, 5, 6, 7, 8, 9] ## possible landslide areas (called bufferes here)

# Create a list to store results
results = []


# Pick a soil depth raster to display (e.g., final year)
soil_depth_map = [f for f in soil_depth_files if '100yrs' in f or '200yrs' in f]
if len(soil_depth_map) == 0:
    print("No final-year soil depth raster found for visualization.")
else:
    soil_depth_path = soil_depth_map[0]  # Use first match

    # Create buffer geometries
    buffer_geoms = []
    for _, row in gdf.iterrows():
        point_id = row['Point_ID']
        point_geom = row.geometry
        for buffer_distance in buffer_sizes:
            buffer = point_geom.buffer(buffer_distance)
            buffer_geoms.append({'geometry': buffer, 'Point_ID': point_id, 'Buffer_Size': buffer_distance})

    buffer_gdf = gpd.GeoDataFrame(buffer_geoms, crs=gdf.crs)

    # Plot raster and buffers
    with rasterio.open(soil_depth_path) as src:
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the soil depth raster
        show(src, ax=ax, cmap='terrain', title='Soil Depth and Buffer Zones')

        # Overlay buffers (colored by buffer size)
        buffer_gdf.plot(ax=ax, column='Buffer_Size', cmap='viridis', alpha=0.3, legend=True)

        # Plot point centers
        gdf.plot(ax=ax, color='black', markersize=5)

        # Add Point_ID labels
        for x, y, pid in zip(gdf.geometry.x, gdf.geometry.y, gdf['Point_ID']):
            ax.text(x, y, str(pid), fontsize=7, ha='center', va='center')

        ax.set_title("Buffer Radii (m) Overlaid on Soil Depth")
        plt.tight_layout()
        plt.show()

point_slope_dict = {}
point_slope_count = {}

# Loop through each point in the shapefile
for _, line_row in line_gdf.iterrows():
    pid = line_row["Point_ID"]
    try:
        zs = zonal_stats([line_row["geometry"]], slope_path, stats=["count", "mean"], nodata=-9999)
        avg = zs[0]["mean"]
        count = zs[0]["count"]
        point_slope_dict[pid] = avg
        point_slope_count[pid] = count
        print(f"[OK] Slope for Point {pid}: {avg:.4f} from {count} pixels")
    except Exception as e:
        point_slope_dict[pid] = np.nan
        point_slope_count[pid] = 0
        print(f"[FAIL LINE SLOPE] Point {pid}: {e}")

# Now begin main loop for each point and buffer
for _, row in gdf.iterrows():
    point_geom = row.geometry
    point_id = row['Point_ID']

    for buffer_distance in buffer_sizes:
        buffer_geom = point_geom.buffer(buffer_distance)
        buffer_json = [buffer_geom.__geo_interface__]
        print(f"\n[CHECK] Point {point_id}, Buffer {buffer_distance}m")

        # Use precomputed slope
        avg_slope = point_slope_dict.get(point_id, np.nan)
        slope_count = point_slope_count.get(point_id, 0)
        print(f"    → Constant slope used: {avg_slope}, pixel count: {slope_count}")

        ### Step 2: Extract Soil Depth for Each Year ###
        for soil_depth_tif in soil_depth_files:
            match = re.search(r'_(\d+)yrs.*\.tif$', soil_depth_tif)
            if match:
                year = int(match.group(1))
            else:
                continue

            try:
                print(f"Opening soil raster: {os.path.basename(soil_depth_tif)}")
                with rasterio.open(soil_depth_tif) as soil_depth_raster:
                    soil_depth_image, _ = mask(soil_depth_raster, buffer_json, crop=True)
                    soil_depth_image = soil_depth_image[0]
                    nodata_val = soil_depth_raster.nodata
                    print(f"    → Raster bounds: {soil_depth_raster.bounds}")
                    print(f"    → Buffer bounds: {buffer_geom.bounds}")
                    print(f"    → Nodata value: {nodata_val}")
                    print(f"    → Min/Max in masked soil: {soil_depth_image.min()}, {soil_depth_image.max()}")

                    valid_soil_depth_values = soil_depth_image[
                        (soil_depth_image != nodata_val) & (soil_depth_image > 0)]
                    print(f"    → Valid values count: {valid_soil_depth_values.size}")

                    if valid_soil_depth_values.size == 0:
                        print(f"No valid soil depth at Year {year}, Point {point_id}, Buffer {buffer_distance}")
                        avg_soil_depth = np.nan
                    else:
                        avg_soil_depth = np.mean(valid_soil_depth_values)
                        print(f"    → Avg soil depth: {avg_soil_depth:.4f}")

            except Exception as e:
                print(f"[FAIL SOIL] Year {year}, Point {point_id}, Buffer {buffer_distance}: {e}")
                avg_soil_depth = np.nan
                continue  # skip bad year

            # Store result regardless of validity for debugging
            results.append({
                'Point_ID': point_id,
                'Year': year,
                'Buffer_Size': buffer_distance,
                'Avg_Slope': avg_slope,
                'Avg_Soil_Depth': avg_soil_depth
            })
            print(f"Stored: Point {point_id}, Year {year}, Buffer {buffer_distance}")

# Final reporting and sorting
print(f"\n Total results collected: {len(results)}")
df = pd.DataFrame(results)

if not df.empty and 'Point_ID' in df.columns:
    df = df.sort_values(by=['Point_ID', 'Year', 'Buffer_Size']).reset_index(drop=True)
else:
    print("No data to sort — check for skipped buffers or missing raster overlap.")
#%% ---------------- CALCULATE FACTOR OF SAFETY and RI ----------------
# Constants 
Sc = 1.25  
pw = 1000  
ps = 1600  
g = 9.81  
yw = g * pw
ys = g * ps
phi = np.deg2rad(41)  
m = 1 # saturation value (unitless)
l = 10  # length (m)
w = 6.7  # width (m)
C0 = 760 # cohesion value (Pa)
j = 0.8  # cohesion depth decay value (m)

# Define functions

def calculate_fs(row):
    hollow_rad = np.radians(row['Avg_Slope'])
    z = row['Avg_Soil_Depth']
    
    if np.isnan(z) or np.isnan(hollow_rad) or z <= 0:
        return np.nan
  
    
    Crb = C0 * np.exp(-z * j)
    Crl = (C0 / (j * z)) * (1 - np.exp(-z * j))

    K0 = 1 - np.sin(hollow_rad)
    
    Kp = np.tan((np.deg2rad(45))+(phi/2))**2
    Ka = np.tan((np.deg2rad(45))-(phi/2))**2


    Frb = (Crb + ((np.cos(hollow_rad)) ** 2) * z * (ys - yw * m) * np.tan(phi)) * l * w
    Frc = (Crl + (K0 * 0.5 * z * (ys - yw * m ** 2) * np.tan(phi))) * (np.cos(hollow_rad) * z * l * 2)
    Frddu = (Kp - Ka) * 0.5 * (z ** 2) * (ys - yw * (m ** 2)) * w
    Fdc = (np.sin(hollow_rad)) * (np.cos(hollow_rad)) * z * ys * l * w

    return (Frb + Frc + Frddu) / Fdc if Fdc != 0 else np.nan

df['FS'] = df.apply(calculate_fs, axis=1)

# ---------------- INTERPOLATE FS TO FIND EXACT FS = 1 YEAR ----------------
optimal_buffers = {}
FS0_MIN = 1

for point_id in df["Point_ID"].unique():
    point_data = df[df["Point_ID"] == point_id].copy()

    buffer_first_crossings = {}

    # ---- Loop over buffers for this point ----
    for buffer_size in sorted(point_data["Buffer_Size"].unique()):
        buffer_data = (
            point_data[point_data["Buffer_Size"] == buffer_size]
            .sort_values(by="Year")
        )

        years     = buffer_data["Year"].values
        fs_values = buffer_data["FS"].values

        # Need at least two FS values and some finite ones
        if len(years) < 2 or np.all(~np.isfinite(fs_values)):
            continue
        
        # 🔴 Skip buffers that are already failed at the first time step
        if fs_values[0] <= FS0_MIN:
            print(f"[SKIP] Point {point_id}, Buffer {buffer_size}: "
                  f"FS(Year={years[0]})={fs_values[0]:.3f} ≤ 1 at start")
            continue

        # Look for sign changes of (FS - 1) between consecutive samples
        diff = fs_values - 1.0
        idx = np.where(
            np.isfinite(diff[:-1]) &
            np.isfinite(diff[1:]) &
            (diff[:-1] * diff[1:] <= 0)
        )[0]

        if idx.size == 0:
            # never crosses 1 (stays >1 or <1 but not from the start)
            continue

        # Use the first crossing segment
        i = idx[0]
        fs_i, fs_ip1 = fs_values[i],   fs_values[i+1]
        t_i,  t_ip1  = years[i],       years[i+1]

        if np.isclose(fs_ip1, fs_i):
            frac = 0.0
        else:
            frac = (1.0 - fs_i) / (fs_ip1 - fs_i)

        crossing_year = t_i + frac * (t_ip1 - t_i)

        if years.min() < crossing_year < years.max():
            buffer_first_crossings[buffer_size] = crossing_year
            print(f"[FS=1] Point {point_id}, Buffer {buffer_size} → "
                  f"Year ≈ {crossing_year:.2f}")

    # ---- Pick earliest crossing buffer for this point ----
    if not buffer_first_crossings:
        print(f"Warning: No valid FS = 1 crossing for Point_ID {point_id} "
              f"(ignoring already-failed buffers). Skipping.")
        continue

    optimal_buffer_size = min(buffer_first_crossings, key=buffer_first_crossings.get)
    optimal_year = buffer_first_crossings[optimal_buffer_size]

    # ---- Interpolate Soil Depth and Slope at this year ----
    selected_buffer_data = (
        point_data[point_data["Buffer_Size"] == optimal_buffer_size]
        .sort_values(by="Year")
    )

    soil_depth_interp = interp1d(
        selected_buffer_data["Year"].values,
        selected_buffer_data["Avg_Soil_Depth"].values,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    slope_interp = interp1d(
        selected_buffer_data["Year"].values,
        selected_buffer_data["Avg_Slope"].values,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )

    estimated_soil_depth = float(soil_depth_interp(optimal_year))
    estimated_slope      = float(slope_interp(optimal_year))

    optimal_buffers[point_id] = {
        "Optimal_Buffer_m":   optimal_buffer_size,
        "Year":               optimal_year,
        "FS":                 1.0,
        "Avg_Soil_Depth_m":   estimated_soil_depth,
        "Avg_Slope_deg":      estimated_slope,
    }

# ---- Convert to DataFrame & save ----
df_optimal_interpolated = pd.DataFrame.from_dict(optimal_buffers, orient="index")
df_optimal_interpolated.reset_index(inplace=True)
df_optimal_interpolated.rename(columns={"index": "Point_ID"}, inplace=True)

output_path = r"C:\Users\sdavilao\Documents\newcodesoil\results\new\optimal_buffer_results_interpolated_ext26_760.csv" #alter for correct cohesion/saturation value
df_optimal_interpolated.to_csv(output_path, index=False)

print(df_optimal_interpolated)