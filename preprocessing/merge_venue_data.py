from config import PROJECT_ROOT
import pandas as pd
from lib.poi_logger import POILog

tag = "DataMerge"


def merge_venues_data(venues_df, emb_df, debug=False):
    merged_df = venues_df.copy()

    venue_ids = merged_df["Venue_ID"].to_numpy()
    emb_df = emb_df[emb_df["Venue_ID"].isin(venue_ids)]

    emb_column = []
    for idx, venue_id in enumerate(venue_ids):
        embedding = emb_df[emb_df["Venue_ID"] == venue_id]
        embedding = embedding["Venue_embedding"].to_numpy()

        if embedding.size > 0:
            embedding = embedding.item()
            emb_column.append(embedding)
        else:
            embedding = None
            merged_df.drop(merged_df[merged_df["Venue_ID"] == venue_id].index, inplace=True)

        if debug and (idx == 0 or ((idx + 1) % 500 == 0) or idx + 1 == len(venue_ids)):
            emb_txt = str(embedding)
            POILog.i(tag, f"[{idx + 1}/{len(venue_ids)}] Embedding: {emb_txt[:50]}")

    if debug:
        POILog.i(tag, f"Venue_ID column: {len(venue_ids)}")
        POILog.i(tag, f"Venue_embedding column: {len(emb_column)}")

    merged_df.loc[:, "Venue_desc_emb"] = emb_column

    return merged_df


def main():
    venues_csv_path = f"{PROJECT_ROOT}/Dataset/venues.csv"
    test_csv_path = f"{PROJECT_ROOT}/additional_data/venues_final.csv"
    emb_csv_path = f"{PROJECT_ROOT}/additional_data/venues_emb.csv"

    POILog.i(tag, "Loading the venues...")
    venues_df = pd.read_csv(venues_csv_path)
    POILog.i(tag, "Venues loaded.")

    POILog.d(tag, "Loading the embeddings...")
    emb_df = pd.read_csv(emb_csv_path)
    POILog.d(tag, "Embeddings loaded.")

    venues_df = merge_venues_data(venues_df, emb_df, debug=True)

    POILog.i(tag, f"Venues: \n{venues_df.head()}", prefix="\n")

    POILog.i(tag, "Saving the updated CSV...")
    venues_df.to_csv(test_csv_path, index=False)
    POILog.i(tag, "Saved.")


if __name__ == "__main__":
    main()
