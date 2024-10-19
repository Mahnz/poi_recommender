from config import PROJECT_ROOT
import pandas as pd
from lib.poi_logger import POILog, LogLevel

"""
The purpose of this script is correcting some mistakes made by the description generator that could lead to
a decrease of the performances at training and inference time. It employs three types of corrections:
 - token replacements, where an isolated (surrounded by whitespace/commas) word is replaced (e.g. 'caf' -> 'cafe')
 - substring replacements, where a specific sequence is replaced, regardless by its surroundings (e.g. ' / ' -> '/')
 - substring ban, where the presence of a specific sequence forces us to drop the venue entirely (e.g. 'condom')
The results are applied in place on the venues description file.
"""

POILog.MAX_LOG_LEVEL = LogLevel.DEBUG
tag = "CorrectDesc"
descriptions_path = f"{PROJECT_ROOT}/additional_data/venues_desc.csv"

POILog.i(tag, "Loading the descriptions...")
venues = pd.read_csv(descriptions_path)
venues_data = venues[["Venue_ID", "Venue_description"]].values
POILog.i(tag, "Descriptions loaded.\n\n")

token_replacements = {
    ("caf", "cafe"),
}

substr_replacements = {
    (" ' ", "'"),
    (" / ", "/"),
    ("( ", "("),
    (" )", ")"),
}

banlist = [
    "condom",
    "condo / condo",
    "townhouse / townhouse",
]

n_removed = 0
n_corrections = 0

for venue_idx, (venue_id, description) in enumerate(venues_data):
    if any(banword in description for banword in banlist):
        deleting_id = venues.loc[venue_idx, "Venue_ID"]
        POILog.w(tag, f"Removing {deleting_id}: {description}")
        venues.drop(index=venue_idx, inplace=True)
        n_removed += 1
        continue

    tokens: list[str] = description.lower().split()
    comma_indexes = [idx for idx, token in enumerate(tokens) if "," in token]
    tokens = [token.replace(",", "") for token in tokens]

    modified = False

    for idx, token in enumerate(tokens):
        for old_token, new_token in token_replacements:
            if token == old_token:
                tokens[idx] = new_token
                modified = True
                break

        if idx in comma_indexes:
            tokens[idx] += ","

    has_substr = any(old_substr in description for old_substr, _ in substr_replacements)

    if has_substr or modified:
        new_desc = " ".join(tokens)

        for (old_substr, new_substr) in substr_replacements:
            if old_substr in new_desc:
                new_desc = new_desc.replace(old_substr, new_substr)

        POILog.i(tag, f"[{venue_idx+1}/{len(venues)}]   {new_desc}   [{venue_id}]")
        venues.loc[venue_idx, :] = {"Venue_ID": venue_id, "Venue_description": new_desc}
        n_corrections += 1

print()
POILog.i(tag, "Saving the corrected descriptions...")
venues.to_csv(descriptions_path, index=False)
POILog.i(tag, "Descriptions saved.")

print()
POILog.i(tag, f"{n_corrections} descriptions corrected")
POILog.i(tag, f"{n_removed} descriptions removed")
