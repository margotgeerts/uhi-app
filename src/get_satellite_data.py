import argparse
import yaml
import geopandas as gpd
import pandas as pd
import planetary_computer
from pystac_client import Client
from odc.stac import stac_load
import rioxarray as rxr
import xarray as xr
import rasterio

from tqdm import tqdm

# 🔹 Load Configurations
with open("./config.yml", "r") as f:
    config = yaml.safe_load(f)

# 🔹 Define Constants
BUFFER_RADIUS = config["buffer_radius"]
START_DATE = config["start_date"]
END_DATE = config["end_date"]
CLOUD_THRESHOLD = config["cloud_threshold"]
OUTPUT_DIR = "data/raw/"
INPUT_FILE = config["input_data"]
bounds = config["bounds"]
SENT_BANDS = config["sentinel_bands"]
LANDSAT_BANDS = config["landsat_bands"]
resolution = config["resolution"]

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🔹 Connect to Planetary Computer
stac_client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")


parser = argparse.ArgumentParser(description="Extract and save satellite data")
parser.add_argument(
    "--collection", 
    type=str, 
    default="sentinel", 
    choices=["sentinel", "landsat"],
    help="Collection to extract data from"
    )
    

def get_landsat_data(collection="landsat-c2-l2"):
    """
    Queries Sentinel-2 or Landsat images for the AOI and time range.
    Filters based on cloud cover.
    """
    search = stac_client.search(
        collections=[collection],
        bbox=bounds,
        datetime=f"{START_DATE}/{END_DATE}",
    query={"eo:cloud_cover": {"lt": CLOUD_THRESHOLD},"platform": {"in": ["landsat-8"]}},
    )
    items = list(search.get_items())
    print(f"Found {len(items)} items")
    return items

def get_sentinel_data(collection="sentinel-2-l2a"):
    """
    Queries Sentinel-2 images.
    Filters based on cloud cover.
    """
    search = stac_client.search(
        collections=[collection],
        bbox=bounds,
        datetime=f"{START_DATE}/{END_DATE}",
        query={"eo:cloud_cover": {"lt": CLOUD_THRESHOLD}},
    )
    items = list(search.get_items())
    print(f"Found {len(items)} items")
    return items

def extract_save_data(collection):
    """
    Extracts and saves data from Sentinel-2 or Landsat images.
    To calculate NDVI, NDBI, MNDWI, LST.
    """
    scale = resolution/111320.
    items = get_sentinel_data() if "sentinel" in collection else get_landsat_data()
    bands = SENT_BANDS if "sentinel" in collection else LANDSAT_BANDS
    print("Loading data...")
    data = stac_load(
        [planetary_computer.sign(item) for item in items],
        bands = bands,
        resolution=scale,
        chunks={"x": 1024, "y": 1024},
        dtype="uint16",
        crs = "EPSG:4326",
        patch_url=planetary_computer.sign,
        bbox=bounds
    ).persist()

    print("Data loaded!")

    # Save all images of indices and timestamps to raw data directory
    height = data.dims["latitude"]
    width = data.dims["longitude"]
    # Define the Coordinate Reference System (CRS) to be common Lat-Lon coordinates
    # Define the tranformation using our bounding box so the Lat-Lon information is written to the GeoTIFF
    gt = rasterio.transform.from_bounds(*bounds, width, height)
    for i in tqdm(range(len(data.time))):
        data_slice = data.isel(time=i)
        data_slice = data_slice.rio.write_crs("epsg:4326", inplace=True)
        data_slice.rio.write_transform(transform=gt, inplace=True)
        filename = os.path.join(OUTPUT_DIR, f"{collection}_{data_slice.time.values}.tif")
        with rasterio.open(filename,'w',driver='GTiff',width=width,height=height,
                   crs='epsg:4326',transform=gt,count=len(bands),compress='lzw',dtype='float64') as dst:
            # Write each band to the file
            for i, band in enumerate(bands):
                dst.write(data_slice[band], i+1)
            dst.close()
        print(f"Saved {filename}")

if __name__ == "__main__":
    args = parser.parse_args()
    extract_save_data(args.collection)
