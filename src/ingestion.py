import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_data(dataset='mlg-ulb/creditcardfraud', download_path='data/'):
    """
    Downloads the credit card fraud dataset from Kaggle and unzips it into the data directory.
    """
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    print("Authenticating with Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    print(f"Downloading dataset: {dataset}")
    api.dataset_download_files(dataset, path=download_path, unzip=True)

    print("✅ Download complete and extracted to:", download_path)

if __name__ == "__main__":
    download_data()
