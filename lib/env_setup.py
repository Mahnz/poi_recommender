import subprocess

commands = [
    "pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118",
    "pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu118.html",
    "pip install torch-geometric",
    "pip install pandas",
    "pip install numpy",
    "pip install matplotlib",
    "pip install plotly",
    "pip install networkx",
    "pip install optuna",
    "pip install scikit-learn",
    "pip install requests",
    "pip install tqdm",
    "pip install clip",
    "pip install transformers",
    "pip install sentence-transformers",
]

for command in commands:
    process = subprocess.run(command, shell=True, check=True)
