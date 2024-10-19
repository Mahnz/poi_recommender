from config import PROJECT_ROOT
import os
import gc
import shutil
import sys
import requests
import zipfile
import pandas as pd
from pandas import DataFrame
from pandas.io.parsers import TextFileReader

from poi_logger import POILog
from preprocessing.merge_venue_data import merge_venues_data

tag = "Write CSV"

debug = {
    "Checkins chunks": True,
    "Venues-Embedding mapping": True,
}

files_to_extract = [
    "dataset_TIST2015_Checkins.txt",
    "dataset_TIST2015_POIs.txt",
    "dataset_TIST2015_Cities.txt",
]

# %% - - - - - - - - - - Dataset download

if not all([os.path.exists(f"{PROJECT_ROOT}/Dataset/Source/{file}") for file in files_to_extract]):
    # Download the dataset from the Google Drive link
    POILog.i(tag, "============= Dataset Downloader =============")

    # Remove the existing dataset folder
    POILog.i(tag, ">> Deleting the existing dataset folder...")
    shutil.rmtree(f"{PROJECT_ROOT}/Dataset", ignore_errors=True)
    POILog.i(tag, "Folder deleted.\n")

    url = (
        "https://drive.usercontent.google.com/download?id=0BwrgZ-IdrTotZ0U0ZER2ejI3VVk&export=download&authuser=0"
        "&resourcekey=0-rlHp_JcRyFAxN7v5OAGldw&confirm=t&uuid=7dcd198c-60be-42f6-9ca2-e5520b4a2930&at"
        "=APZUnTXMBotzh_ctrebQqxOL1i8M%3A1723936332783"
    )
    source_folder = f"{PROJECT_ROOT}/Dataset/Source"
    zip_path = source_folder + "/dataset_TIST2015.zip"

    if not os.path.exists(source_folder):
        os.makedirs(source_folder)

    POILog.i(tag, ">> Downloading the dataset ZIP...")
    response = requests.get(url)
    with open(zip_path, "wb") as file:
        file.write(response.content)
    POILog.i(tag, "Dataset downloaded.")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        POILog.i(tag, ">> Extracting files...")
        for file in files_to_extract:
            zip_ref.extract(file, source_folder)
            POILog.i(tag, f" - Extracted '{file}'")

    os.remove(zip_path)

# %% - - - - - - - - - - Script Settings

chunk_size = 10_000
lite_checkins: int | None = None
country_code: str | None = "IT"
checkin_debug_interval = 75 if lite_checkins is None else round(lite_checkins / 10_000)

source_folder = f"{PROJECT_ROOT}/Dataset/Source"
checkins_txt_path = f"{source_folder}/dataset_TIST2015_Checkins.txt"
venues_txt_path = f"{source_folder}/dataset_TIST2015_POIs.txt"

csv_path = f"{PROJECT_ROOT}/Dataset"
checkins_csv_path = f"{csv_path}/checkins.csv"
users_csv_path = f"{csv_path}/users.csv"
venues_csv_path = f"{csv_path}/venues.csv"
category_mapping_csv_path = f"{csv_path}/venue_categories.csv"

emb_csv_path = f"{PROJECT_ROOT}/additional_data/venues_emb.csv"

blacklist = [checkins_csv_path, users_csv_path, venues_csv_path, category_mapping_csv_path]

POILog.i(tag, "============= TXT -> CSV Converter =============")

# %% - - - - - - - - - - Old files deletion

if os.path.exists(checkins_csv_path) or os.path.exists(users_csv_path) or os.path.exists(venues_csv_path):
    while True:
        POILog.w(tag, "The existing CSV files will be deleted and replaced. ", suffix="")
        delete_request = input("Confirm? (Y/N)  ").strip().lower()
        print()
        if delete_request in ["y", "yes"]:
            break
        elif delete_request in ["n", "no"]:
            POILog.i(tag, "Operation aborted. Files will not be deleted.\nExiting...")
            exit(0)
        else:
            POILog.w(tag, "Invalid input. Please, type 'Y' or 'N'.")

POILog.i(tag, ">> Deleting existing CSV files...")
for file in blacklist:
    if os.path.exists(file):
        os.remove(file)
POILog.i(tag, "Files deleted.\n")

# %% - - - - - - - - - - Venues loading and processing

POILog.i(tag, ">> Loading venues...")
venues_columns = ["Venue_ID", "Latitude", "Longitude", "Venue_category", "Country_code"]
venues_df = pd.read_csv(
    filepath_or_buffer=venues_txt_path,
    sep="\t",
    header=None,
    names=venues_columns,
)

if country_code is not None:
    POILog.i(tag, f"Filtering the venues with country code '{country_code}'...")
    POILog.d(tag, f" - Initial venues: {len(venues_df)}")
    venues_df = venues_df[venues_df["Country_code"] == country_code]
    POILog.d(tag, f" - Filtered venues: {len(venues_df)}")

if os.path.exists(emb_csv_path):
    POILog.d(tag, "Loading the embeddings...")
    emb_df = pd.read_csv(emb_csv_path)
    POILog.d(tag, "Embeddings loaded.\n\n")

    # Add for each venue the description's embedding
    POILog.i(tag, "Merging the embeddings into the venues data...")
    venues_df = merge_venues_data(venues_df, emb_df, debug["Venues-Embedding mapping"])
    POILog.d(tag, "Merge completed.\n\n")
else:
    POILog.w(tag, "Embeddings CSV not found. This could lead to serious problems in future operations.")

print()

venues_ids = set(venues_df["Venue_ID"].values)
venue_id_mapping = {venue_id: idx for idx, venue_id in enumerate(venues_df["Venue_ID"])}
venues_df["Venue_ID"] = venues_df["Venue_ID"].map(venue_id_mapping)

# Fix the "Cafè" category, which is not correctly encoded
venues_df['Venue_category'] = venues_df['Venue_category'].replace({'Caf\x1a\x1a': 'Cafè', 'Caf\x1a': 'Cafè'})

# Map the venue categories to a numerical code
venue_categories = venues_df[['Venue_category']].drop_duplicates().reset_index(drop=True)
venue_categories['Cat_code'] = venue_categories['Venue_category'].astype('category').cat.codes
venue_categories.to_csv(f"{csv_path}/venue_categories.csv", index=False)

venues_df.to_csv(
    venues_csv_path,
    index=False,
    header=["Venue_ID", "Latitude", "Longitude", "Venue_category", "Country_code", "Venue_desc_emb"]
)

POILog.i(tag, "Venues converted and saved.\n")

# %% - - - - - - - - - - Checkins loading and processing

POILog.i(tag, ">> Loading the checkins...")
checkins_columns = ["User_ID", "Venue_ID", "UTC_time", "Timezone_offset"]
checkins_reader: TextFileReader = pd.read_csv(
    filepath_or_buffer=checkins_txt_path,
    sep="\t",
    header=None,
    names=checkins_columns,
    chunksize=chunk_size,
    nrows=lite_checkins,
)
POILog.i(tag, "Checkins loaded.\n")

POILog.i(tag, ">> Converting the checkins...")

user_ids = set()

user_checkins = {}
for idx, checkins_chunk in enumerate(checkins_reader):
    checkins_chunk: DataFrame
    debug_chunk = (debug["Checkins chunks"] and (idx % checkin_debug_interval == 0))

    if debug_chunk:
        POILog.d(tag, f" - Filtering the checkins chunk...")
        if country_code is not None: POILog.d(tag, f" - Initial checkins: {len(checkins_chunk)}")

    checkins_chunk = checkins_chunk[checkins_chunk["Venue_ID"].isin(venues_ids)]

    if debug_chunk and country_code is not None:
        POILog.d(tag, f" - Filtered checkins: {len(checkins_chunk)}")

    # Convert UTC time to datetime object
    checkins_chunk.loc[:, 'UTC_time'] = pd.to_datetime(
        checkins_chunk['UTC_time'],
        format='%a %b %d %H:%M:%S %z %Y'  # Format used in the TXT file
    )

    # Group check-ins by the "User_ID"
    for user_id, chunk in checkins_chunk.groupby("User_ID"):
        if user_id not in user_checkins:
            user_checkins[user_id] = []
        user_checkins[user_id].extend(chunk["UTC_time"])

    user_ids.update(checkins_chunk["User_ID"])

    # Conversione dei Venue_ID nei check-in utilizzando la mappatura
    checkins_chunk.loc[:, "Venue_ID"] = checkins_chunk["Venue_ID"].map(venue_id_mapping)

    checkins_chunk.to_csv(checkins_csv_path, mode="a", header=(idx == 0), index=False)
    del checkins_chunk
    gc.collect()

    if debug_chunk:
        POILog.d(tag, f" - [{(idx + 1) * chunk_size} checkins converted.]")
        sys.stdout.flush()

POILog.i(tag, "Checkins converted.\n")

# %% - - - - - - - - - - Recency scores computation

recency_scores = {}
POILog.i(tag, ">> Computing recency scores...")
for idx, (user_id, times) in enumerate(user_checkins.items()):
    if idx == 0 or (idx + 1) % 1000 == 0 or idx == len(user_checkins):
        POILog.d(tag, f" - [{idx + 1}/{len(user_checkins)}] users processed.")
        sys.stdout.flush()

    min_time = min(times)
    max_time = max(times)
    range_time = (max_time - min_time).total_seconds()

    if range_time > 0:
        recency_scores[user_id] = {
            time.strftime('%Y-%m-%d %H:%M:%S%z'): (time - min_time).total_seconds() / range_time for time in times
        }
    elif range_time == 0:
        recency_scores[user_id] = {
            time.strftime('%Y-%m-%d %H:%M:%S%z'): 1.0 for time in times
        }
    else:
        POILog.w(tag, "Time is not timing.")

POILog.i(tag, "Recency scores computed.\n")

POILog.i(tag, ">> Inserting recency scores to checkins...")
checkins = pd.read_csv(checkins_csv_path, usecols=["User_ID", "Venue_ID", "UTC_time"])
for idx, row in checkins.iterrows():
    user_id = row["User_ID"]
    timestamp = pd.to_datetime(row["UTC_time"]).strftime('%Y-%m-%d %H:%M:%S%z')

    recency_score = recency_scores[user_id][timestamp]
    checkins.at[idx, "Recency_Score"] = recency_score

checkins.to_csv(checkins_csv_path, index=False)
POILog.i(tag, "Recency scores added.\n")

# %% - - - - - - - - - - User identification

POILog.i(tag, ">> Identifying users...")
users_df = pd.DataFrame(sorted(user_ids), columns=["User_ID"])
users_df.to_csv(users_csv_path, index=False)
POILog.i(tag, f"{len(users_df)} unique users identified.\n")

del users_df
gc.collect()
