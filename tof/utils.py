import os
import math
import argparse
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from rasterio.enums import Resampling
from shapely.geometry import shape
from shapely.ops import unary_union
from skimage.morphology import closing, rectangle
from sklearn.cluster import KMeans


def argparse_TOF():
    """Parse the command line arguments for the TOF script"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "state",
        type=str,
        help="Set it to the foldernames of your sites as string",
    )
    parser.add_argument(
        "--create_masks",
        action="store_true",
        help="Set it to True if you want to create the masks",
    )
    parser.add_argument(
        "--epsg",
        type=str,
        default="EPSG:25832",
        help="Set the EPSG code of your data",
    )
    return parser.parse_args()


def get_file_list(directory, extension=".tif", type=None):
    """Get a list of files with a given extension in a directory and optionally rename them with a prefix"""
    files = os.listdir(directory)
    file_list = []
    for file in files:
        if type:
            if file.endswith(".tif"):
                filename, _ = os.path.splitext(file)
                fileid = filename[4:]
                if fileid.startswith("_"):
                    fileid = fileid[1:]
                new_filename = f"{type}_{fileid}{extension}"
                new_directory = directory.replace("TOP", type)
                new_file_path = os.path.join(new_directory, new_filename)
                # Create the directory if it doesn't exist
                os.makedirs(new_directory, exist_ok=True)
                file_list.append(new_file_path)
        else:
            if file.endswith(extension):
                file_list.append(f"{directory}/{file}")

    return sorted(file_list)


def calculate_line_length(x1, y1, x2, y2):
    """Calculate the length of the line segment using the distance formula"""
    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return length


def calculate_width_length(polygon):
    """Calculate the width and length a polygon"""
    min_rotated_rect = polygon.minimum_rotated_rectangle

    x, y = min_rotated_rect.exterior.coords.xy

    width = min(
        calculate_line_length(x[0], y[0], x[1], y[1]),
        calculate_line_length(x[1], y[1], x[2], y[2]),
    )
    length = max(
        calculate_line_length(x[0], y[0], x[1], y[1]),
        calculate_line_length(x[1], y[1], x[2], y[2]),
    )

    return width, length


def mask_lidar_chm(raster, height_threshold):
    """Mask CHM values below a given threshold"""
    lidar_chm = raster.read(1)
    lidar_chm = lidar_chm.astype(float)
    lidar_chm_masked = np.where(lidar_chm < height_threshold, 0, 1)
    return lidar_chm_masked


def kmeans_clustering(ndvi, num_clusters=2):
    """Apply K-means clustering to the NDVI values to separate vegetation from non-vegetation pixels"""
    flat_ndvi = ndvi.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(flat_ndvi)
    cluster_labels = kmeans.labels_.reshape(ndvi.shape)
    return cluster_labels, kmeans.cluster_centers_


def simplify_geometry(geometry, tolerance):
    """apply Douglas-Peucker simplification to a geometry"""
    return geometry.simplify(tolerance, preserve_topology=True)


def shape_filtering(gdf, area_threshold, filter_threshold):
    """Filter polygons based on area and width criteria and simplify the polygons using the Douglas-Peucker algorithm"""
    # Filter for small polygons
    filtered_gdf = gdf[gdf["geometry"].area > area_threshold]
    # Use Peuken-Douglas algorithm to simplify the polygons
    filtered_gdf.loc[:, "geometry"] = filtered_gdf["geometry"].apply(
        lambda x: simplify_geometry(x, filter_threshold)
    )

    return filtered_gdf


def tif_tof_filtering(gdf, area_threshold, min_width):
    """Filter polygons based on width and area criteria and fill holes in polygons below a certain area threshold"""

    # Unpack the polygons from the GeoDataFrame to a list for iteration
    list_polygons = list(gdf.geometry)

    # Filter polygons based on width and area criteria
    smaller20m_polygons = []
    bigger20m_polygons = []

    for polygon in list_polygons:
        shp_polygon = shape(polygon)
        # Check if the width is smaller than 20m
        width, _ = calculate_width_length(polygon)

        if width < min_width:
            smaller20m_polygons.append(shp_polygon)
        else:
            bigger20m_polygons.append(shp_polygon)

    # Check the area of the polygons
    tof = []
    tif = []
    for polygon in bigger20m_polygons:
        if polygon.area > area_threshold:
            # TODO while the area is bigger than 5000m², check for the minimum width is less than 10m and cut there and check again
            tif.append(polygon)
        else:
            tof.append(polygon)

    tof.extend(smaller20m_polygons)

    # Create GeoDataFrame from filtered polygons
    tif_gdf = gpd.GeoDataFrame(geometry=tif)
    tof_gdf = gpd.GeoDataFrame(geometry=tof)
    tif_gdf["class"] = "TIF"
    tof_gdf["class"] = "TOF"

    return pd.concat([tif_gdf, tof_gdf], ignore_index=True)


def forest_definition(vegetation_mask, raster, min_size, min_width):
    """Apply morphological closing to the vegetation mask and polygonize the result"""
    footprint1 = rectangle(5, 5)  # five pixels (0.2m) are 1m
    # Apply morphological closing
    vegetation_mask_closed = closing(vegetation_mask, footprint1)

    # Polygonize the raster
    raster_polygons = rasterio.features.shapes(
        vegetation_mask_closed.astype(np.uint8),
        transform=raster.transform,
        mask=(vegetation_mask_closed == 1),
        connectivity=4,
    )

    list_polygons = []
    for polygon, value in raster_polygons:
        shp_polygon = shape(polygon)
        list_polygons.append(shp_polygon)
    raster_gdf = gpd.GeoDataFrame(geometry=list_polygons)
    # raster_gdf = raster_gdf.set_crs("epsg:25833")

    return tif_tof_filtering(raster_gdf, min_size, min_width)


def compute_si(green_band, blue_band):
    """Compute the Shadow Index (SI) from green, and blue bands"""
    # Ensure the input bands have the same shape
    assert (
        green_band.shape == blue_band.shape
    ), "Green, and blue bands must have the same shape"

    # Convert to float to avoid integer division
    green_band = green_band.astype(float)
    blue_band = blue_band.astype(float)

    # Avoid division by zero
    np.seterr(divide="ignore", invalid="ignore")

    # Compute SI
    si = np.sqrt((255 - blue_band) * (255 - green_band))

    # Reset division by zero warning to default
    np.seterr(divide="warn", invalid="warn")

    return si


def compute_ndvi(red_band, nir_band):
    """Compute the Normalized Difference Vegetation Index (NDVI) from red and near-infrared bands"""
    # Ensure the input bands have the same shape
    assert (
        red_band.shape == nir_band.shape
    ), "Red and NIR bands must have the same shape"

    # Convert to float to avoid integer division
    red_band = red_band.astype(float)
    nir_band = nir_band.astype(float)

    # Avoid division by zero
    np.seterr(divide="ignore", invalid="ignore")

    # Compute NDVI
    ndvi = (nir_band - red_band) / (nir_band + red_band)

    # Reset division by zero warning to default
    np.seterr(divide="warn", invalid="warn")

    return ndvi


def subtract_and_save(bDOM, DGM, TOP, output_file):
    """Subtract DGM from bDOM and save the result to a new raster file"""
    with rasterio.open(bDOM) as bDOM_src, rasterio.open(DGM) as DGM_src, rasterio.open(
        TOP
    ) as TOP_src:
        # Resample DGM to match bDOM resolution
        DGM_data = DGM_src.read(
            out_shape=(DGM_src.count, bDOM_src.height, bDOM_src.width),
            resampling=Resampling.nearest,
        )

        # Read bDOM
        bDOM_data = bDOM_src.read(1)

        # Subtract resampled DGM from bDOM
        result = DGM_data - bDOM_data

        # Resample the result to match TOP resolution
        result_resampled = np.empty((TOP_src.height, TOP_src.width), dtype=np.float32)
        rasterio.warp.reproject(
            source=result,
            destination=result_resampled,
            src_transform=bDOM_src.transform,
            src_crs=bDOM_src.crs,
            dst_transform=TOP_src.transform,
            dst_crs=TOP_src.crs,
            resampling=Resampling.nearest,
        )

        # Save the result to a new raster file
        profile = TOP_src.profile
        profile.update(dtype=rasterio.float32, count=1)
        with rasterio.open(output_file, "w", **profile) as dst:
            dst.write(result_resampled, 1)


def calculate_width_length_ratio(polygon):
    """Calculate the width-length ratio for a given polygon"""
    width, length = calculate_width_length(polygon)

    return length / width if length != 0 else 0


def classify_tif_tof(gdf):
    """Classify the polygons as Forest, Patch, Linear, or Tree based on their elongation and area"""
    if gdf.empty:
        return gdf

    gdf["Elongation"] = gdf.apply(
        lambda row: calculate_width_length_ratio(row["geometry"]), axis=1
    )
    gdf["Area"] = gdf["geometry"].area

    gdf["tof_class"] = "Forest"
    gdf["classvalue"] = int(1)
    gdf.loc[gdf["class"] == "TOF", "tof_class"] = "Patch"
    gdf.loc[gdf["class"] == "TOF", "classvalue"] = 2
    gdf.loc[
        (gdf["Elongation"] > 3) & (gdf["area"] < 5000) & (gdf["area"] > 10), "tof_class"
    ] = "Linear"
    gdf.loc[
        (gdf["Elongation"] > 3) & (gdf["area"] < 5000) & (gdf["area"] > 10),
        "classvalue",
    ] = 3
    gdf.loc[(gdf["Elongation"] > 5), "tof_class"] = "Linear"
    gdf.loc[(gdf["Elongation"] > 5), "classvalue"] = 3
    gdf.loc[
        (gdf["class"] == "TOF")
        & (gdf["area"] < 500)
        & (gdf["Elongation"] < 3),  # & (gdf["Circularity"] > 0.75)
        "tof_class",
    ] = "Tree"
    gdf.loc[
        (gdf["class"] == "TOF")
        & (gdf["area"] < 500)
        & (gdf["Elongation"] < 3),  # & (gdf["Circularity"] > 0.75)
        "classvalue",
    ] = 4
    gdf["classvalue"] = gdf["classvalue"].astype("int32")

    return gdf


def merge_and_filter_tif_tof(tif_tof_gdf, area_threshold, min_width):
    """Merge the TIF and TOF polygons and filter them based on the forest definition criteria"""
    merged_tif_tof_geometry = unary_union(tif_tof_gdf.geometry)
    merged_gdf = gpd.GeoDataFrame(geometry=[merged_tif_tof_geometry])
    merged_exploded_gdf = merged_gdf.explode(index_parts=True)

    # Check again for the forest definition
    tif_tof_merged_gdf = tif_tof_filtering(
        merged_exploded_gdf, area_threshold, min_width
    )
    tif_tof_merged_gdf.loc[:, "area"] = tif_tof_merged_gdf["geometry"].area

    return tif_tof_merged_gdf
