"""
UNISON Framework - Data Cleaning & Interaction Preparation
Module for cleaning raw CSV data and preparing user-item bags.
"""

import pandas as pd
import logging
import pickle
import os
from typing import Tuple, Set, Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_and_clean_items(items_path: str) -> pd.DataFrame:
    """Loads and cleans the item catalog."""
    try:
        df = pd.read_csv(items_path, sep=";", encoding="latin1", low_memory=False)
        # Standardization
        df.columns = [col.strip() for col in df.columns]
        df["ISBN"] = df["ISBN"].astype(str).str.strip()
        df = df.dropna(subset=["ISBN", "Book-Title"])

        # Clean Year
        df["Year-Of-Publication"] = pd.to_numeric(df["Year-Of-Publication"], errors="coerce")
        df = df.dropna(subset=["Year-Of-Publication"])
        df["Year-Of-Publication"] = df["Year-Of-Publication"].astype(int)

        logger.info(f"Loaded {len(df)} valid items.")
        return df
    except Exception as e:
        logger.error(f"Error loading items: {e}")
        return pd.DataFrame()

def load_and_clean_users(users_path: str) -> pd.DataFrame:
    """Loads and cleans the user registry."""
    try:
        df = pd.read_csv(users_path, sep=";", encoding="latin1", low_memory=False)
        df.columns = [col.strip() for col in df.columns]
        df["User-ID"] = df["User-ID"].astype(str).str.strip()

        if "Age" in df.columns:
            df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

        logger.info(f"Loaded {len(df)} users.")
        return df
    except Exception as e:
        logger.error(f"Error loading users: {e}")
        return pd.DataFrame()

def load_and_clean_ratings(ratings_path: str, valid_items: Set[str], valid_users: Set[str]) -> pd.DataFrame:
    """Filters ratings based on valid users/items and explicit scores."""
    try:
        df = pd.read_csv(ratings_path, sep=";", encoding="latin1", low_memory=False)
        df.columns = [col.strip() for col in df.columns]
        df["User-ID"] = df["User-ID"].astype(str).str.strip()
        df["ISBN"] = df["ISBN"].astype(str).str.strip()

        # Referential Integrity and explicit ratings (1-10)
        df = df[df["User-ID"].isin(valid_users)]
        df = df[df["ISBN"].isin(valid_items)]
        df["Book-Rating"] = pd.to_numeric(df["Book-Rating"], errors="coerce")
        df = df[(df["Book-Rating"] >= 1) & (df["Book-Rating"] <= 10)]

        # Keep the most recent if multiple exist
        df = df.drop_duplicates(subset=["User-ID", "ISBN"], keep="last")
        logger.info(f"Cleaned ratings: {len(df)} remaining.")
        return df
    except Exception as e:
        logger.error(f"Error loading ratings: {e}")
        return pd.DataFrame()

def build_item_metadata(items_df: pd.DataFrame) -> Dict[str, str]:
    """
    Creates the mapping from item ID to its descriptive key.
    The description MUST match the keys in your pre-computed embeddings file.
    """
    mapping = {}
    for _, row in items_df.iterrows():
        isbn = str(row["ISBN"])
        title = str(row["Book-Title"]).strip()
        author = str(row.get("Book-Author", "")).strip()
        publisher = str(row.get("Publisher", "")).strip()
        year = str(row.get("Year-Of-Publication", ""))

        # Matches the descriptive format used in the framework
        parts = [title]
        if author: parts.append(f"by {author}")
        if publisher: parts.append(f"published by {publisher}")
        if year: parts[-1] += f" in {year}"

        mapping[isbn] = ", ".join(parts) + "."
    return mapping

def run_data_cleaning_pipeline(
    items_csv: str,
    users_csv: str,
    ratings_csv: str,
    output_dir: str,
    split_age: int = 32
):
    """Orchestrates the cleaning and prepares input for the Episode Generator."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load and Clean
    items_df = load_and_clean_items(items_csv)
    users_df = load_and_clean_users(users_csv)

    valid_ids = set(items_df["ISBN"].tolist())
    valid_uids = set(users_df["User-ID"].tolist())

    ratings_df = load_and_clean_ratings(ratings_csv, valid_ids, valid_uids)

    # 2. Group into Bags
    user2items = {}
    user2ratings = {}
    for uid, group in ratings_df.groupby("User-ID"):
        user2items[uid] = group["ISBN"].tolist()
        user2ratings[uid] = group["Book-Rating"].astype(float).tolist()

    # 3. Target Attribute Mapping (Age-based binary classification)
    user2attr = {}
    for _, row in users_df.iterrows():
        uid = str(row["User-ID"])
        age = row.get("Age")
        if pd.isna(age):
            user2attr[uid] = -1
        else:
            user2attr[uid] = 1 if age >= split_age else 0

    # 4. Descriptive Keys
    item_metadata = build_item_metadata(items_df)

    # 5. Export for Episode Generation
    # We save these as pickles to be consumed by the generator
    export_files = {
        "user2items.pkl": user2items,
        "user2ratings.pkl": user2ratings,
        "user2attr.pkl": user2attr,
        "item_metadata.pkl": item_metadata
    }

    for filename, data in export_files.items():
        with open(os.path.join(output_dir, filename), "wb") as f:
            pickle.dump(data, f)

    logger.info(f"Cleaning complete. Processed files saved to {output_dir}")

if __name__ == "__main__":
    # Define your local data paths
    run_data_cleaning_pipeline(
        items_csv="data/raw/Books.csv",
        users_csv="data/raw/Users.csv",
        ratings_csv="data/raw/Ratings.csv",
        output_dir="data/processed_bags"
    )