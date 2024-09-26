# Manual TOF Detection

## Introduction

This project will take a aerial image and an nDSM to classify four Trees Outside Forest Classes:
1. Class 1: Forest
2. Class 2: Patch
3. Class 3: Linear
4. Class 4: Tree

If no nDSM is available it can take a picture based DSM and a DEM to automatically create it.

## Background

An automated approach with manual refinement is used to create the reference data for training and validation. First, a mask is created for all values below a threshold for the nDSM of 3m. The threshold for the nDSM was set to include shrubs but exclude high growing crops. This was then used to mask the NDVI, which was then clustered into two classes using K-means. This results in one class containing all buildings and the other class containing the trees. As there was an issue with some roofs having shadows, the tree mask was updated using the Shadow Index (SI) with values lower than 150. The SI is calculated as follows:

```
SI = sqrt((255 - Blue) * (255 - Green))
```
Afterwards, the remaining tree raster cells were filtered by a morphological closing (5×5) and polygonized. To clean the results smaller objects than 3m² were delted and the edges we smoothed using To distinguish between forest and TOF, we follow the FAO definition. Forest is defined as areas larger than 0.5ha and with a width larger than 20m. The width of the polygons is calculated using smaller site of the smallest rotated enclosing rectangle. All polygons with a smaller width than 20m and smaller area than 0.5ha are classified as TOF.
These TOF polygons are then classified into the following classes:
- **Patch**: Areas smaller than 0.5ha and larger than 500m² and elongation (length divided by width) lower than 3.
- **Linear**: Elongation is higher than 3.
- **Tree:** Smaller than 500m²
\end{itemize}
Finally, a manual refinement step was necessary to ensure the quality of the reference data. In particular, linear objects that were part of a forest polygon had to be cut and reclassified.

## Installation

### Using Conda

To create and activate a conda environment with the necessary dependencies, follow these steps:

1. Create a new conda environment:
    ```bash
    conda create --name tof-detection python=3.x
    ```

2. Activate the environment:
    ```bash
    conda activate tof-detection
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Using Pip

1. To install the necessary dependencies, run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To start the process you need the following folder structure:

```
main.py
├── Sites
|    ├── Site_Name
|    |    ├── DGM
|    |    ├── TOP
|    |    ├── bDOM
|    |    ├── nDOM
```

The script does have two functionalities:
1. Creating the TOF classes from the input data with a shapifile as result

Now it's time for the manual refinement!

2. Create the masks to then use them for network training with tif masks as result

Start the Script: 
```bash
python main.py "Site_Name" --create_masks --epsg "EPSG:Number"
```

## Configuration
Ensure that the EPSG code provided matches the coordinate system of your data.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any changes or suggestions.

## License
This project is licensed under the MIT License.

## Contact Information
For support or questions, please contact [moritz.lucas@uos.de].