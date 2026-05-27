import urllib.request
import os

# Download Telco Churn dataset from Kaggle (public dataset)
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
destination = "data/raw/telco_churn.csv"

print(f"Downloading data to {destination}...")
urllib.request.urlretrieve(url, destination)
print(f"✅ Data downloaded successfully!")

# Verify
import pandas as pd
df = pd.read_csv(destination)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
