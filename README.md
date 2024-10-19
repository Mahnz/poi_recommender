# POI-RECOMMENDER

## Overview
This project aims to realize a Recommender System that leverages the power of Graph Neural Networks (GNNs) to provide personalized recommendations for Points of Interest (POIs). By analyzing user preferences and historical data, the system can predict and suggest locations that a user is likely to be interested in. The core functionality revolves around a GNN architecture that captures the complex relationships and interactions between users and POIs.

During inference, users can input a query, and the system will process it alongside the user's historical data to generate tailored recommendations. This approach enhances the user experience by delivering highly relevant and personalized suggestions, making it easier for users to discover new and interesting places that align with their tastes and interests.

The project is designed to be flexible and scalable, accommodating various datasets and user inputs to provide robust and accurate recommendations.


## Installation
To set up the project locally, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/Mahnz/poi_recommender.git
   cd poi_recommender
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the required dependencies by executing the `env_setup.py` script:
   ```bash
   python ./lib/env_setup.py
   ```


## Usage
### Generating Descriptions
To generate descriptions for venue images:
1. Place the images in the `images/` directory.
2. Run the description generation script:
   ```bash
   python preprocessing/descriptions_generator.py
   ```

### Encoding Descriptions
To encode the generated descriptions:
1. Ensure the descriptions are stored in `additional_data/venues_desc.csv`.
2. Run the encoding script:
   ```bash
   python preprocessing/description_encoder.py
   ```

### Retrieving Venue Images
To retrieve venue images from Google Street View:
1. Set up your API credentials in the environment variables `STREET_VIEW_KEY` and `STREET_VIEW_SECRET`.
2. Run the image retrieval script:
   ```bash
   python preprocessing/retrieve_venue_images.py
   ```


## Acknowledgements
- [PyTorch](https://pytorch.org/)
- [Hugging Face](https://huggingface.co/)
- [Google Street View API](https://developers.google.com/maps/documentation/streetview)
