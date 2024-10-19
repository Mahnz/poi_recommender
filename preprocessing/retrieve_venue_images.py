from config import PROJECT_ROOT
import os
import pandas as pd
import hashlib
import hmac
import base64
import urllib.parse as urlparse
import requests
from lib.poi_logger import POILog

f"""
The purpose of this script is to retrieve the Google Maps POI that is closest to our Foursquare POI in a 100m radius
(assuming the closest is the exact same venue even if the coordinates don't perfectly match), download a picture of
the venue from the Google Maps Street View Static API (SVSAPI) and save the image in the <PROJECT_ROOT>/images folder.

This requires to set the SVSAPI credentials as environment variables with keys STREET_VIEW_KEY and STREET_VIEW_SECRET
in order to link the program to the API. We recommend using an environment file (e.g. street_view.env) filled with two
entries in KEY="value" fashion and pass the .env file to the interpreter in the PyCharm run configuration.
"""

tag = "Venue Images"

api_key = os.environ.get("STREET_VIEW_KEY")
api_secret = os.environ.get("STREET_VIEW_SECRET")

if api_key is None or api_secret is None:
    POILog.e(tag, "The Google Maps API credentials are required to run this script.")
    exit(1)

def sign_url(input_url):
    decoded_key = base64.urlsafe_b64decode(api_secret)

    url = urlparse.urlparse(input_url)
    url_to_sign = url.path + "?" + url.query

    signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)
    encoded_signature = base64.urlsafe_b64encode(signature.digest())

    original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query

    return original_url + "&signature=" + encoded_signature.decode()


def get_poi_image(lat, lon, radius=100):
    location = f"{lat},{lon}"

    image_url = f"https://maps.googleapis.com/maps/api/streetview?"
    image_url += f"location={location}"
    image_url += f"&radius={radius}"
    image_url += f"&size=400x400"
    image_url += f"&return_error_code=true"
    image_url += f"&key={api_key}"

    signed_url = sign_url(image_url)
    POILog.v(tag, f"Image Request URL: {signed_url}")

    response = requests.get(signed_url)

    if response.status_code == 200:
        return response.content
    else:
        return None


def is_image_present(venue_id):
    return os.path.exists(f"../images/{venue_id}.jpg")


def save_image(image: bytes, venue_id):
    image_path = f"{PROJECT_ROOT}/images/{venue_id}.jpg"
    with open(image_path, 'wb') as file:
        file.write(image)
    POILog.i(tag, f"Image saved at path: {image_path}")


def main():
    POILog.i(tag, "Loading the venues...")
    venues = pd.read_csv(f"{PROJECT_ROOT}/Dataset/venues.csv")
    POILog.i(tag, "Venues loaded.")

    print("---------------------------------------------------------------------------------------------------------------")

    count_successes = 0
    count_failures = 0
    count_skips = 0

    for idx, (venue_id, lat, lon) in enumerate(venues[["Venue_ID", "Latitude", "Longitude"]].to_numpy()):
        if is_image_present(venue_id):
            count_skips += 1
            continue

        print()
        POILog.i(tag, f"[{idx + 1}/{len(venues)}] Retreiving the image of the venue {venue_id} with coords {lat},{lon} ...")

        image = get_poi_image(lat, lon)

        if image is not None:
            save_image(image, venue_id)
            count_successes += 1
        else:
            POILog.e(tag, f"Image download failed")
            count_failures += 1

    print("---------------------------------------------------------------------------------------------------------------")
    print()

    POILog.i(tag, f"Downloaded images: {count_successes}")
    POILog.i(tag, f"Failed downloads: {count_failures}")
    POILog.i(tag, f"Skipped venues: {count_skips}")

    print()


if __name__ == "__main__":
    main()
