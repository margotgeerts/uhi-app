import os
repo_dir = "/Users/margotgeerts/Library/CloudStorage/OneDrive-KULeuven/UrbanHeatIslands_NYC"

os.chdir(repo_dir)
import sys
sys.path.append(repo_dir)
print(os.getcwd())

# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data Science
import numpy as np
import pandas as pd

# Geospatial data analysis
import geopandas as gpd


# Feature Engineering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Planetary Computer Tools
import pystac_client
import planetary_computer as pc

# Others
import os
from tqdm import tqdm
import joblib
tqdm.pandas()
from src.get_satellite_features import *
from src.get_geo_features import *

# 🔹 Load Configurations
with open("config.yml", "r") as f:
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


print(os.getcwd())

if os.path.exists("data/processed/processed_data_attempt2.csv"):
    print('True')
    data = pd.read_csv("data/processed/processed_data_attempt2.csv")
else:
    data = gpd.read_file(INPUT_FILE)
    data['geometry'] = gpd.points_from_xy(data.Longitude, data.Latitude, crs="EPSG:4326")
    data = get_building_features(data)
    data = get_weather_features(data)
    print(data.head())
    data = get_satellite_features(data)

    print(data.head())

    data = data.replace([np.inf, -np.inf], np.nan)
    print(data.columns)

    # Remove duplicate rows from the DataFrame based on specified columns and keep the first occurrence
    columns_to_check = [col for col in data.columns if (col not in ['UHI Index', 'geometry', 'datetime'])]
    for col in columns_to_check:
        # Check if the value is a numpy array and has more than one dimension
        data[col] = data[col].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) and x.ndim > 0 else x)

    # Now remove duplicates
    data = data.drop_duplicates(subset=columns_to_check, keep='first')

    data=data.reset_index(drop=True)
    data.to_csv("data/processed/processed_data_attempt2.csv", index=False)
data = data.drop(['datetime', 'geometry', 'Longitude', 'Latitude'], axis=1)
# Split the data into features (X) and target (y), and then into training and testing sets
X = data.drop(columns=['UHI Index']).values
print(X.shape)
y = data['UHI Index'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=123)

# Scale the training and test data using standardscaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

print("Training model...")
model = RandomForestRegressor(random_state=42)
model.fit(X_train,y_train)
joblib.dump(model, "models/rf.pkl")

insample_predictions = model.predict(X_train)
Y_train = y_train.tolist()
print(r2_score(Y_train, insample_predictions))

outofsample_predictions = model.predict(X_test)
Y_test = y_test.tolist()
print(r2_score(Y_test, outofsample_predictions))

print("Preparing validation set...")
validation_set = pd.read_csv("data/raw/Submission_template_UHI2025-v2.csv")
if os.path.exists("data/processed/validation_data_attempt2.csv"):
    validation_gdf = pd.read_csv("data/processed/validation_data_attempt2.csv")
else:
    validation_gdf = gpd.GeoDataFrame(validation_set, geometry=gpd.points_from_xy(validation_set.Longitude, validation_set.Latitude), crs="EPSG:4326")
    validation_gdf = get_building_features(validation_gdf)
    validation_gdf = get_weather_features(validation_gdf)
    validation_gdf = get_satellite_features(validation_gdf)
    validation_gdf.to_csv("data/processed/validation_data_attempt2.csv", index=False)
validation_gdf = validation_gdf.replace([np.inf, -np.inf], np.nan)
validation_gdf = validation_gdf.drop('datetime', axis=1) if 'datetime' in validation_gdf.columns else validation_gdf
validation_gdf = validation_gdf.drop('geometry', axis=1) if 'geometry' in validation_gdf.columns else validation_gdf
validation_gdf = validation_gdf.drop('UHI Index', axis=1) if 'UHI Index' in validation_gdf.columns else validation_gdf
validation_gdf = validation_gdf.drop('Longitude', axis=1) if 'Longitude' in validation_gdf.columns else validation_gdf
validation_gdf = validation_gdf.drop('Latitude', axis=1) if 'Latitude' in validation_gdf.columns else validation_gdf
# validation_gdf = validation_gdf.drop(['datetime', 'geometry'], axis=1)

X_validation = validation_gdf.values
print(X_validation.shape)
print(validation_gdf.columns)
print([col for col in validation_gdf.columns if col not in data.columns])

print("Making predictions on the validation set...")
# # Make predictions on the validation set
validation_predictions = model.predict(X_validation)
validation_set['UHI Index'] = validation_predictions
validation_set.to_csv("data/processed/Submission_attempt2_UHI2025.csv", index=False)