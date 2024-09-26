import rasterio
import rasterio.features
import numpy as np
import geopandas as gpd
import pandas as pd
import os

from tof.utils import (
    classify_tif_tof,
    compute_ndvi,
    forest_definition,
    mask_lidar_chm,
    shape_filtering,
    subtract_and_save,
    merge_and_filter_tif_tof,
    get_file_list,
    argparse_TOF,
    kmeans_clustering,
    compute_si,
)


if __name__ == "__main__":

    args = argparse_TOF()
    state = args.state
    epsg = args.epsg
    create_masks = args.create_masks

    list_path_RGBI = get_file_list(f"Sites/{state}/TOP")
    list_path_shapes = get_file_list(f"Sites/{state}/TOP", ".shp", "SHP")
    list_path_bDOM = get_file_list(f"Sites/{state}/DGM", ".tif", "bDOM")
    list_path_DGM = get_file_list(f"Sites/{state}/bDOM", ".tif", "DGM")
    list_path_nDOM = get_file_list(f"Sites/{state}/TOP", ".tif", "nDOM")
    path_final_shape = f"Sites/{state}/SHP/{state}_result_merged.shp"

    if create_masks:
        gdf = gpd.read_file(path_final_shape)

        # Check if mask folder exists
        if not os.path.exists(f"Sites/{state}/Masks"):
            os.makedirs(f"Sites/{state}/Masks")

        for i, path_RGBI in enumerate(list_path_RGBI):

            name = path_RGBI.split("/")[-1].replace("TOP", "mask")
            mask_path = f"Sites/{state}/Masks/{name}"

            with rasterio.open(path_RGBI[:-4] + ".tif") as ref_raster:
                with rasterio.open(
                    mask_path,
                    "w",
                    driver="GTiff",
                    height=ref_raster.height,
                    width=ref_raster.width,
                    count=1,
                    dtype=rasterio.uint16,
                    crs=ref_raster.crs,
                    transform=ref_raster.transform,
                ) as mask_raster:
                    burned = rasterio.features.rasterize(
                        [
                            (geometry, value)
                            for geometry, value in zip(gdf.geometry, gdf["classvalue"])
                        ],
                        out_shape=ref_raster.shape,
                        all_touched=False,
                        transform=ref_raster.transform,
                        fill=0,
                        dtype=rasterio.uint16,
                        default_value=1,
                    )

                    mask_raster.write(burned, 1)

            print(f"{i+1}/{len(list_path_RGBI)}: {name} created at {mask_path}")

    else:
        # Parameter to dermine after what a area is classified as forest
        area_threshold = 5000  # in m²
        min_width = 20  # in m

        # Parameter to mask non forested areas
        height_threshold = 3  # in m
        ndvi_threshold = 0.4
        si_threshold = 150

        tif_tof_gdf = gpd.GeoDataFrame()

        # Loop over all images
        for i in range(len(list_path_RGBI)):

            ## Step 0: Create nDOM
            if not os.path.exists(list_path_nDOM[i]):
                subtract_and_save(
                    list_path_bDOM[i], list_path_DGM[i], list_path_nDOM[i]
                )

            # Open images
            with rasterio.open(list_path_RGBI[i][:-3] + "tif") as top, rasterio.open(
                list_path_nDOM[i]
            ) as height:
                print(f"starting with {list_path_RGBI[i]}")
                ## Step 1: Mask all objects smaller than height threshold
                lidar_chm_masked = mask_lidar_chm(height, height_threshold)

                ## Step 2: Masking non Vegetated Objects: Apply K-means clustering on NDVI
                # Compute NDVI and SI
                ndvi = compute_ndvi(top.read(1), top.read(4))
                si = compute_si(top.read(2), top.read(3))

                # Mask NDVI with height information
                masked_ndvi = np.where(lidar_chm_masked < 1, 0, ndvi)

                masked_ndvi = np.nan_to_num(masked_ndvi, nan=0)

                # Compute Clusters on the masked NDVI
                cluster_labels, cluster_centers = kmeans_clustering(masked_ndvi)

                # Choose the higher cluster as vegetation
                if cluster_centers[0] > cluster_centers[1]:
                    vegetation_mask = np.where(cluster_labels == 0, 1, 0)
                else:
                    vegetation_mask = np.where(cluster_labels == 1, 1, 0)

                # Update mask with SI
                vegetation_mask = np.where(
                    (vegetation_mask == 1)
                    & (si > si_threshold)
                    & (masked_ndvi < ndvi_threshold),
                    0,
                    vegetation_mask,
                )

                ## Step 3: Apply forest definition
                tif_tof = forest_definition(
                    vegetation_mask, top, area_threshold, min_width
                )
                tif_tof = tif_tof.set_crs(epsg)

                ## Step 4: Filter TOF and TIF
                tif_tof_filtered = shape_filtering(tif_tof, 3, 0.2)

                # Append the current tif_tof_filtered DataFrame to the cumulative tif_tof_gdf DataFrame
                tif_tof_gdf = pd.concat(
                    [tif_tof_gdf, tif_tof_filtered], ignore_index=True
                )
                tif_tof_filtered.to_file(list_path_shapes[i])

                print(f"Finished with {i+1} of {len(list_path_RGBI)}")

        print("Merging all shapes together.")

        ## Step 5: Put them together and check again for the forest defnition
        tif_tof_merged_gdf = merge_and_filter_tif_tof(
            tif_tof_gdf, area_threshold, min_width
        )

        ## Step 6: Classifiy TOF
        tif_tof_classified_gdf = classify_tif_tof(tif_tof_merged_gdf)
        tif_tof_classified_gdf = tif_tof_classified_gdf.set_crs(epsg)
        tif_tof_classified_gdf.to_file(path_final_shape)

        # Delete all temporary files
        extensions = [".shp", ".cpg", ".dbf", ".prj", ".shx"]
        for path in list_path_shapes:
            for ext in extensions:
                file_path = path.replace(".shp", ext)
                if os.path.exists(file_path):
                    os.remove(file_path)
