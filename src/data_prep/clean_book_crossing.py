"""
Book-Crossing Dataset Preprocessing

This module provides utilities for loading and cleaning the Book-Crossing dataset,
a widely-used benchmark for collaborative filtering and recommendation systems.

Dataset Structure:
    The Book-Crossing dataset consists of three CSV files:
    - Books.csv: Book metadata (ISBN, Title, Author, Publisher, Year)
    - Users.csv: User demographics (User-ID, Location, Age)
    - Ratings.csv: User-book interactions (User-ID, ISBN, Rating)

Preprocessing Philosophy:
    1. String IDs: All identifiers (User-ID, ISBN) are kept as strings to avoid
       integer overflow and preserve leading zeros in ISBNs.

    2. Whitespace Handling: The dataset contains inconsistent whitespace and
       encoding issues. All string fields are stripped and "N/A" is converted to NaN.

    3. Explicit vs Implicit Feedback: Ratings of 0 indicate implicit feedback
       (book views without ratings). These are filtered out by default, keeping
       only explicit ratings in [1, 10].

    4. Duplicate Handling: Some users rate the same book multiple times. We keep
       the last rating to reflect the most recent preference.

    5. Age Binarization: For bag-level classification (Stage 3 in UNISON), ages
       are binarized at the median (32 years) to balance classes.

Typical Usage:
    # Step 1: Load and clean base tables
    books_df, valid_isbns = load_and_clean_books("Books.csv")
    users_df, valid_users = load_and_clean_users("Users.csv")
    ratings_df = load_and_clean_ratings("Ratings.csv", valid_isbns, valid_users)

    # Step 2: Build user interaction dictionaries
    user2items, user2ratings = build_user_interactions(ratings_df, min_items_per_user=10)

    # Step 3: Compute bag-level attributes for Stage 3
    user2target_attr = compute_user_target_attr(users_df, drop_users_with_missing_age=True)

Notes:
    - All functions preserve deterministic ordering (via sorting) for reproducibility
    - Statistics are printed to stdout for debugging and data exploration
    - Missing embeddings are handled gracefully (converted to NaN, can be filtered later)
"""

import pandas as pd
from typing import Tuple, Set, Dict, List


# ============================================================
# 1) BOOKS PREPROCESSING
# ============================================================
def load_and_clean_books(books_csv_path: str) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Load and clean Books.csv from the Book-Crossing dataset.

    This function handles the quirks of the Book-Crossing book metadata:
    - ISBNs with leading zeros (must be strings, not integers)
    - Inconsistent whitespace and encoding issues (latin1)
    - Missing or invalid publication years

    The cleaning process ensures only valid books with complete metadata
    are retained for downstream processing.

    Args:
        books_csv_path: Path to Books.csv file

    Returns:
        Tuple containing:
        - books_df: Cleaned DataFrame with columns [ISBN, Title, Author, Publisher, Year]
        - valid_isbns: Set of valid ISBN strings for filtering ratings

    Cleaning Rules:
        * ISBN is forced to string type and whitespace is stripped
        * All string columns have whitespace stripped
        * Empty strings and "N/A" values are converted to NaN
        * Rows without ISBN or Title are dropped (required fields)
        * Year is converted to integer; non-numeric years are dropped
        * Author and Publisher may be missing (NaN) and are kept as-is

    Example:
        >>> books_df, valid_isbns = load_and_clean_books("data/Books.csv")
        >>> print(f"Loaded {len(books_df)} valid books")
        >>> print(f"Year range: {books_df['Year'].min()} - {books_df['Year'].max()}")
    """
    # Load with latin1 encoding (dataset contains special characters)
    df = pd.read_csv(
        books_csv_path,
        sep=";",
        encoding="latin1",
        dtype={"ISBN": str},  # CRITICAL: preserve leading zeros and avoid int overflow
        low_memory=False,
    )

    # Force ISBN to string and strip whitespace (handles mixed types from CSV parsing)
    df["ISBN"] = df["ISBN"].astype(str).str.strip()

    # Strip whitespace from all text columns (dataset has trailing/leading spaces)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    # Replace empty strings and "N/A" placeholders with proper NaN
    df = df.replace({"": None, "N/A": None})

    # Drop books without ISBN or Title (these are required for item identification)
    df = df.dropna(subset=["ISBN", "Title"])

    # Clean publication year: keep only valid numeric years
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df = df.dropna(subset=["Year"])
        df["Year"] = df["Year"].astype(int)

    # Build set of valid ISBNs for filtering ratings table
    valid_isbns: Set[str] = set(df["ISBN"].tolist())

    return df, valid_isbns


# ============================================================
# 2) USERS PREPROCESSING
# ============================================================
def load_and_clean_users(users_csv_path: str) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Load and clean Users.csv from the Book-Crossing dataset.

    This function handles user demographic data, with special attention to:
    - User IDs that may contain leading zeros or special characters
    - Age field with invalid entries (e.g., "canada", empty strings)
    - Location strings that may need parsing (not done here, preserved as-is)

    Args:
        users_csv_path: Path to Users.csv file

    Returns:
        Tuple containing:
        - users_df: Cleaned DataFrame with columns [User-ID, Location, Age]
        - valid_users: Set of valid User-ID strings for filtering ratings

    Cleaning Rules:
        * User-ID is forced to string type and whitespace is stripped
        * All string columns have whitespace stripped
        * Empty strings are converted to NaN
        * Age is converted to numeric; invalid values (e.g., "canada") become NaN
        * Users with invalid/missing Age are KEPT (can be filtered later based on task)

    Notes:
        Users with missing Age can still be used for item scoring tasks (Stage 2),
        but should be filtered out for age-based classification tasks (Stage 3).
        Use drop_users_with_missing_age=True in compute_user_target_attr() for this.

    Example:
        >>> users_df, valid_users = load_and_clean_users("data/Users.csv")
        >>> print(f"Loaded {len(users_df)} users")
        >>> print(f"Users with valid age: {users_df['Age'].notna().sum()}")
    """
    # Load with User-ID as string to preserve identifiers
    df = pd.read_csv(
        users_csv_path,
        sep=";",
        encoding="latin1",
        dtype={"User-ID": str},  # CRITICAL: preserve user ID format
        low_memory=False,
    )

    # Force User-ID to string and strip whitespace
    df["User-ID"] = df["User-ID"].astype(str).str.strip()

    # Strip whitespace from all text columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    # Replace empty strings with NaN
    df = df.replace({"": None})

    # Parse Age: convert to numeric, invalid entries become NaN
    # Invalid examples in the dataset: "canada", empty strings, negative numbers
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    # Build set of valid user IDs (all users present in table, regardless of age validity)
    valid_users: Set[str] = set(df["User-ID"].tolist())

    return df, valid_users


# ============================================================
# 3) RATINGS PREPROCESSING
# ============================================================
def load_and_clean_ratings(
    ratings_csv_path: str,
    valid_isbns: Set[str],
    valid_users: Set[str],
    min_rating: int = 1,
    max_rating: int = 10,
) -> pd.DataFrame:
    """
    Load and clean Ratings.csv from the Book-Crossing dataset.

    This function performs the critical join operation between users and books,
    filtering to valid (User-ID, ISBN) pairs and handling rating scale issues.

    The Book-Crossing dataset contains both explicit ratings (1-10) and implicit
    feedback (rating=0 for books that were viewed but not rated). By default,
    this function keeps only explicit ratings for clearer preference signals.

    Args:
        ratings_csv_path: Path to Ratings.csv file
        valid_isbns: Set of valid ISBNs from load_and_clean_books()
        valid_users: Set of valid User-IDs from load_and_clean_users()
        min_rating: Minimum rating to keep (default=1, filters out implicit feedback)
        max_rating: Maximum rating to keep (default=10, upper bound of rating scale)

    Returns:
        Cleaned ratings DataFrame with columns [User-ID, ISBN, Rating]

    Cleaning Steps:
        1. Force User-ID and ISBN to string type and strip whitespace
        2. Filter to valid (User-ID, ISBN) pairs based on provided sets
        3. Convert Rating to numeric (invalid ratings become NaN and are dropped)
        4. Filter ratings to [min_rating, max_rating] range
        5. Handle duplicates: if a user rated the same book multiple times,
           keep the last rating (most recent preference)

    Notes:
        - Setting min_rating=0 will include implicit feedback (rating=0)
        - Duplicate ratings are surprisingly common (~2-3% of the dataset)
        - The last-rating-wins strategy assumes temporal ordering in the CSV

    """
    # Load with both IDs as strings
    df = pd.read_csv(
        ratings_csv_path,
        sep=";",
        encoding="latin1",
        dtype={"User-ID": str, "ISBN": str},  # CRITICAL: both as strings for join
        low_memory=False,
    )

    # Force User-ID and ISBN to string and strip whitespace
    df["User-ID"] = df["User-ID"].astype(str).str.strip()
    df["ISBN"] = df["ISBN"].astype(str).str.strip()

    # Strip whitespace from all text columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    # Replace empty strings with NaN
    df = df.replace({"": None})

    # Filter to valid users and items (inner join via boolean indexing)
    df = df[df["User-ID"].isin(valid_users)]
    df = df[df["ISBN"].isin(valid_isbns)]

    # Convert Rating to numeric
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    # Drop rows with NaN rating (invalid/unparseable ratings)
    df = df.dropna(subset=["Rating"])

    # Filter by rating range (default: [1, 10] removes implicit feedback at 0)
    df = df[(df["Rating"] >= min_rating) & (df["Rating"] <= max_rating)]

    # Handle duplicate ratings: keep last occurrence for each (User-ID, ISBN) pair
    # Assumes CSV is sorted by time (last rating is most recent)
    df = df.sort_index()  # Preserve file order
    df = df.drop_duplicates(subset=["User-ID", "ISBN"], keep="last")

    return df


# ============================================================
# 4) USER INTERACTIONS AGGREGATION
# ============================================================
def build_user_interactions(
    ratings_df: pd.DataFrame,
    min_items_per_user: int = 1,
) -> Tuple[Dict[str, List[str]], Dict[str, List[float]]]:
    """
    Aggregate ratings into per-user interaction dictionaries.

    This function converts the flat ratings table into a user-centric format
    suitable for the UNISON framework, where each user (bag) is associated with
    a list of items and corresponding scores.

    The function ensures deterministic ordering by sorting ratings before grouping,
    which is critical for reproducibility when splitting into support/query sets.

    Args:
        ratings_df: Cleaned ratings DataFrame from load_and_clean_ratings()
        min_items_per_user: Minimum number of ratings required to keep a user
                            (default=1, but typically set higher like 10 for
                            meaningful cold-start evaluation)

    Returns:
        Tuple containing:
        - user2items: Dictionary mapping User-ID -> [ISBN1, ISBN2, ...]
        - user2ratings: Dictionary mapping User-ID -> [rating1, rating2, ...]
                        (aligned with user2items, same order)

    Notes:
        - Both dictionaries have the same keys (user IDs)
        - Lists within each user are deterministically ordered (sorted by ISBN)
        - Users with fewer than min_items_per_user interactions are dropped
        - Function prints summary statistics for data exploration

    Example:
        >>> user2items, user2ratings = build_user_interactions(ratings_df, min_items_per_user=10)
        >>> user_id = list(user2items.keys())[0]
        >>> print(f"User {user_id} rated {len(user2items[user_id])} books")
        >>> print(f"Average rating: {sum(user2ratings[user_id]) / len(user2ratings[user_id]):.2f}")
    """
    # Handle edge case: empty ratings table
    if ratings_df.empty:
        print("WARNING: ratings_df is empty in build_user_interactions. "
              "Check that Users.csv and Ratings.csv IDs actually overlap.")
        return {}, {}

    # Sort for deterministic ordering within each user
    # This ensures support/query splits are reproducible given a random seed
    ratings_sorted = ratings_df.sort_values(by=["User-ID", "ISBN"])

    user2items: Dict[str, List[str]] = {}
    user2ratings: Dict[str, List[float]] = {}

    # Group by user and collect items + ratings
    for user_id, grp in ratings_sorted.groupby("User-ID"):
        isbns = grp["ISBN"].tolist()
        ratings = grp["Rating"].astype(float).tolist()

        # Filter users with too few interactions
        if len(isbns) < min_items_per_user:
            continue

        user2items[user_id] = isbns
        user2ratings[user_id] = ratings

    # ========== Compute and print summary statistics ==========
    num_users = len(user2items)

    # Count users with substantial interaction history (10+ ratings)
    count_over_10 = sum(1 for v in user2items.values() if len(v) >= 10)
    percentage_over_10 = (count_over_10 / num_users * 100) if num_users > 0 else 0

    # Count unique items across all users
    all_items = {isbn for items in user2items.values() for isbn in items}
    num_items = len(all_items)

    # Compute rating distribution statistics
    lengths = [len(v) for v in user2items.values()]
    if lengths:
        min_len = min(lengths)
        max_len = max(lengths)
        avg_len = sum(lengths) / len(lengths)
    else:
        min_len = max_len = avg_len = 0

    print("=== User Interactions Summary ===")
    print(f"# users (after min_items_per_user={min_items_per_user}): {num_users}")
    print(f"% of users with 10+ ratings: {percentage_over_10:.2f}%")
    print(f"# unique items rated: {num_items}")
    print(f"min #ratings per user: {min_len}")
    print(f"max #ratings per user: {max_len}")
    print(f"avg #ratings per user: {avg_len:.2f}")

    return user2items, user2ratings


# ============================================================
# 5) BAG-LEVEL ATTRIBUTE (AGE BINARIZATION)
# ============================================================
def compute_user_target_attr(
    users_df: pd.DataFrame,
    drop_users_with_missing_age: bool = False,
) -> Dict[str, int]:
    """
    Compute bag-level target attributes for Stage 3 classification.

    This function binarizes user ages for the bag classification task in UNISON
    Stage 3 (predicting user demographics from functional embeddings). The age
    threshold of 32 years was chosen as the approximate median to balance classes.

    Age Binarization Rules:
        - Age < 32  → target_attr = 0 (younger users)
        - Age ≥ 32  → target_attr = 1 (older users)
        - Missing or invalid age:
            * If drop_users_with_missing_age=True  → user is skipped (not in output dict)
            * If drop_users_with_missing_age=False → target_attr = -1 (can be filtered later)

    Args:
        users_df: Cleaned users DataFrame from load_and_clean_users()
        drop_users_with_missing_age: Whether to exclude users with missing/invalid age
                                      from the output dictionary

    Returns:
        Dictionary mapping User-ID -> target_attr (0, 1, or -1)

    Notes:
        - Age statistics are computed only for "reasonable" ages [5, 100] to avoid
          outliers like age=0 or age=200+ which appear in the raw dataset
        - The function prints detailed statistics for debugging and verification
        - For Stage 2 (item scoring), users with target_attr=-1 can still be used
        - For Stage 3 (age classification), filter to target_attr ∈ {0, 1}

    Example:
        >>> # Include all users (for Stage 2 training)
        >>> user2target = compute_user_target_attr(users_df, drop_users_with_missing_age=False)
        >>>
        >>> # Exclude users without age (for Stage 3 evaluation)
        >>> user2target = compute_user_target_attr(users_df, drop_users_with_missing_age=True)
        >>> valid_users = [uid for uid, attr in user2target.items() if attr in [0, 1]]
    """
    user2target: Dict[str, int] = {}

    # Counters for statistics
    num_total_users = len(users_df)
    num_bin0 = 0  # Age < 32
    num_bin1 = 0  # Age >= 32
    num_minus1 = 0  # Missing/invalid age
    num_skipped = 0  # Skipped due to drop_users_with_missing_age=True

    # Process each user
    for _, row in users_df.iterrows():
        user_id = str(row["User-ID"])

        age = row.get("Age", None)

        # Handle missing age
        if pd.isna(age):
            if drop_users_with_missing_age:
                num_skipped += 1
                continue
            user2target[user_id] = -1
            num_minus1 += 1
            continue

        # Handle invalid age (non-numeric)
        try:
            age_val = float(age)
        except Exception:
            if drop_users_with_missing_age:
                num_skipped += 1
                continue
            user2target[user_id] = -1
            num_minus1 += 1
            continue

        # Binarize age at threshold of 32 years (approximate median)
        if age_val < 32.0:
            user2target[user_id] = 0
            num_bin0 += 1
        else:
            user2target[user_id] = 1
            num_bin1 += 1

    # ========== Print age distribution statistics ==========
    if "Age" in users_df.columns:
        age_stats_series = users_df["Age"].dropna()
        # Filter to "reasonable" ages for statistics (exclude obvious errors)
        age_stats_series = age_stats_series[
            (age_stats_series >= 5) & (age_stats_series <= 100)
        ]

    # ========== Print target attribute distribution ==========
    print("\n============================================================")
    print("TARGET_ATTR (Age-based Binarization) SUMMARY")
    print("============================================================")
    print(f"Total users in Users.csv:               {num_total_users}")
    print(f"Users with assigned target_attr (keys): {len(user2target)}")
    print(f"  target_attr = 0 (age < 32):           {num_bin0}")
    print(f"  target_attr = 1 (age >= 32):          {num_bin1}")
    print(f"  target_attr = -1 (missing/invalid):   {num_minus1}")
    print("============================================================\n")

    return user2target


# ============================================================
# 6) EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":
    """
    Example usage demonstrating the complete preprocessing pipeline.
    
    This example loads the raw Book-Crossing CSV files, cleans them, and
    builds the data structures needed for episode construction.
    
    Adjust the paths to match your local directory structure.
    """
    # File paths (adjust to your setup)
    books_path = "data/raw/Books.csv"
    users_path = "data/raw/Users.csv"
    ratings_path = "data/raw/Ratings.csv"

    # Step 1: Load and clean base tables
    print("Loading and cleaning Books.csv...")
    books_df, valid_isbns = load_and_clean_books(books_path)


    print("Loading and cleaning Users.csv...")
    users_df, valid_users = load_and_clean_users(users_path)


    print("Loading and cleaning Ratings.csv...")
    ratings_df = load_and_clean_ratings(
        ratings_path,
        valid_isbns,
        valid_users,
        min_rating=1,  # Filter out implicit feedback (rating=0)
        max_rating=10,
    )


    # Step 2: Build user interaction dictionaries

    user2items, user2ratings = build_user_interactions(
        ratings_df,
        min_items_per_user=1  # Keep all users for now; filter later if needed
    )

    # Step 3: Compute age-based target attributes

    user2target_attr = compute_user_target_attr(
        users_df,
        drop_users_with_missing_age=False,  # Keep all users; mark missing age as -1
    )