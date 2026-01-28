# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 15:39:24 2025

@author: sdavilao
"""

#%%

import geopandas as gpd
import os
import glob
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

#%% reproject resulting tifs

def reproject_all_tifs(input_folder, output_folder, target_crs="EPSG:32610"): # target crs
    os.makedirs(output_folder, exist_ok=True)
    tif_files = glob.glob(os.path.join(input_folder, "*.tif"))

    for tif_path in tif_files:
        with rasterio.open(tif_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            out_name = os.path.basename(tif_path).replace(".tif", "_32610.tif") # alter for desired crs
            out_path = os.path.join(output_folder, out_name)

            with rasterio.open(out_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest
                    )
            print(f"✅ Reprojected: {out_name}")

# ==== Usage ====
input_folder = r"C:\Users\sdavilao\Documents\newcodesoil\simulation_results\new\new\GeoTIFFs"
output_folder = r"C:\Users\sdavilao\Documents\newcodesoil\simulation_results\new\GeoTIFFs\reproj_tif"


reproject_all_tifs(input_folder, output_folder)

#%% reproject shp files for landslide area selection

# === Set your folders ===
input_folder = r"C:\Users\sdavilao\Documents\newcodesoil"       # Folder with .shp files in EPSG:X
output_folder = r"C:\Users\sdavilao\Documents\newcodesoil\reproj_shp"   # Where to save reprojected files

# Create output folder if needed
os.makedirs(output_folder, exist_ok=True)

# === Loop through all .shp files ===
shapefiles = glob.glob(os.path.join(input_folder, "*.shp"))

for shp_path in shapefiles:
    try:
        # Load and assign correct CRS
        gdf = gpd.read_file(shp_path)
        gdf.set_crs(epsg=6557, inplace=True)  # alter espf to input shp CRS

        # Reproject to EPSG:32610
        gdf_utm = gdf.to_crs(epsg=32610) # desired output crs

        # Save to output folder
        base_name = os.path.splitext(os.path.basename(shp_path))[0]
        out_path = os.path.join(output_folder, f"{base_name}_32610.shp")
        gdf_utm.to_file(out_path)

        print(f"Reprojected: {base_name}.shp → EPSG:32610")

    except Exception as e:
        print(f"Failed: {os.path.basename(shp_path)} → {e}")
