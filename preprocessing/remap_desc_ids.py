from config import PROJECT_ROOT
import pandas as pd
from lib.poi_logger import POILog

tag = "ID Remapper"

venues_txt_path = f"{PROJECT_ROOT}/Dataset/Source/dataset_TIST2015_POIs.txt"
desc_path = f"{PROJECT_ROOT}/additional_data/venues_desc.csv"
desc_mapped_path = f"{PROJECT_ROOT}/additional_data/venues_desc_mapped.csv"

venues_txt_columns = ["Venue_ID", "Latitude", "Longitude", "Venue_category", "Country_code"]
id_name = "Venue_ID"
feature_names = ["Venue_description"]

POILog.i(tag, "Reading the files...")

venues_txt_df = pd.read_csv(
    venues_txt_path,
    sep="\t",
    header=None,
    names=venues_txt_columns,
)

desc_df = pd.read_csv(desc_path)

POILog.i(tag, "Files loaded.")

desc_ids = desc_df["Venue_ID"].to_numpy()
venues_txt_df = venues_txt_df[venues_txt_df["Country_code"] == "IT"]
venues_txt_df = venues_txt_df[venues_txt_df["Venue_ID"].isin(desc_ids)]

dest_df = pd.DataFrame()

mapping = {venue_id: idx for idx, venue_id in enumerate(venues_txt_df["Venue_ID"])}
dest_df[id_name] = desc_df[id_name].map(mapping)

for feature_name in feature_names:
    dest_df[feature_name] = desc_df[feature_name]

POILog.i(tag, f"Mapping entries: {len(mapping)}")
POILog.i(tag, f"Written entries: {len(dest_df)}")

POILog.i(tag, "Mapping the ids...")
dest_df.to_csv(desc_mapped_path, index=False, header=["Venue_ID", "Venue_description"])
POILog.i(tag, "Mapping completed.")
