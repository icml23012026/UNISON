"""
UNISON Episode Construction for Book-Crossing Dataset

This module constructs training and evaluation episodes for the UNISON framework
by splitting the Book-Crossing dataset into warm/cold scenarios. Each episode
represents a bag of scored items with support/query partitioning for meta-learning.

Episode Types:
    The framework creates three episode types corresponding to the scenarios
    described in the UNISON paper:

    1. WU-WI (Warm User-Warm Item):
       - User appears in training data
       - All items in query set appeared during training
       - Support: N_sup warm items
       - Query: Remaining warm items
       - Use case: Standard collaborative filtering evaluation

    2. WU-CI (Warm User-Cold Item):
       - User appears in training data
       - All items in query set are NEW (not seen during training)
       - Support: Empty (no support needed)
       - Query: All cold items for this user
       - Use case: Recommend newly released items to existing users

    3. CU-Mixed (Cold User-Mixed Items):
       - User does NOT appear in training data
       - Query contains both warm items (CU-WI) and cold items (CU-CI)
       - Support: N_sup warm items
       - Query: Remaining warm items + all cold items
       - Use case: New user cold-start with mixed item familiarity

Warm/Cold Split Strategy:
    - Users: Random 80/20 split → warm_users / cold_users
    - Items: Random 80/20 split → warm_items / cold_items
    - WU-WI and WU-CI share the SAME warm users (no new users in WU-CI)
    - CU-Mixed uses only cold users (disjoint from WU-WI/WU-CI)

Mask Codes for Query Items:
    Each query item is tagged with a mask code indicating its warm/cold status:
    - 1 (CU-WI): Cold user, warm item
    - 2 (CU-CI): Cold user, cold item
    - 3 (WU-CI): Warm user, cold item
    - 4 (WU-WI): Warm user, warm item

    These masks enable fine-grained evaluation (e.g., separate metrics for
    CU-WI vs CU-CI within the same episode).

Episode Format:
    Each episode is a dictionary with the following structure:
    {
        "id": str,                    # User ID (bag identifier)
        "scenario": str,              # "WU-WI" | "WU-CI" | "CU-Mixed"
        "items_sup": List[str],       # Item descriptions (human-readable)
        "scores_sup": List[float],    # Ratings for support items
        "items_qry": List[str],       # Item descriptions for query
        "scores_qry": List[float],    # Ratings for query items
        "mask_qry": List[int],        # Mask codes (1-4) for each query item
        "target_attr": int,           # Bag-level attribute (0/1/-1 for age)
    }

Item Description Format:
    Items are represented as human-readable strings rather than ISBNs:
    "Title by Author, published by Publisher in Year."

    Example: "Harry Potter and the Philosopher's Stone by J.K. Rowling,
              published by Bloomsbury in 1997."

    This format enables:
    - Text embedding via language models (Stage 1)
    - Interpretable debugging and visualization
    - Easy cross-referencing with original dataset

Typical Usage:
    from clean_book_crossing import (
        load_and_clean_books,
        load_and_clean_users,
        load_and_clean_ratings,
        build_user_interactions,
        compute_user_target_attr,
    )

    # Step 1: Load preprocessed data
    books_df, valid_isbns = load_and_clean_books("Books.csv")
    users_df, valid_users = load_and_clean_users("Users.csv")
    ratings_df = load_and_clean_ratings("Ratings.csv", valid_isbns, valid_users)
    user2items, user2ratings = build_user_interactions(ratings_df)
    user2target_attr = compute_user_target_attr(users_df, drop_users_with_missing_age=True)

    # Step 2: Build ISBN → description mapping
    isbn2key = build_isbn_to_description_mapping(books_df)

    # Step 3: Generate episodes for different support set sizes
    for n_sup in [50, 60, 70, 80]:
        wu_wi, wu_ci, cu_mixed, stats = build_warm_cold_episodes(
            user2items, user2ratings, user2target_attr, isbn2key,
            n_sup=n_sup, warm_user_ratio=0.8, warm_item_ratio=0.8
        )
        save_episode_splits(wu_wi, wu_ci, cu_mixed, f"episodes_N_SUP_{n_sup}")

Design Rationale:
    - ISBN-level warm/cold split ensures items are truly unseen (not just different
      editions or metadata variants)
    - WU-CI reuses WU-WI users to isolate the effect of cold items (holding user
      familiarity constant)
    - Empty support in WU-CI tests whether learned user embeddings generalize to
      new items without additional adaptation
    - CU-Mixed combines both challenges (cold user + mixed items) for the hardest
      evaluation setting
"""

import os
import json
import random
from typing import Dict, List, Tuple, Set, Any

import pandas as pd

from src.data_prep.clean_book_crossing import (
    load_and_clean_books,
    load_and_clean_users,
    load_and_clean_ratings,
    build_user_interactions,
    compute_user_target_attr,
)

# ============================================================
# MASK CODES FOR QUERY ITEMS
# ============================================================
MASK_CODE = {
    "CU-WI": 1,  # Cold user, warm item in query
    "CU-CI": 2,  # Cold user, cold item in query
    "WU-CI": 3,  # Warm user, cold item in query
    "WU-WI": 4,  # Warm user, warm item in query
}


# ============================================================
# HELPER: SAVE EPISODES TO DISK
# ============================================================
def save_episode_splits(
        wu_wi_episodes: List[Dict[str, Any]],
        wu_ci_episodes: List[Dict[str, Any]],
        cu_mixed_episodes: List[Dict[str, Any]],
        output_dir: str,
        prefix_wu_wi: str = "wu_wi",
        prefix_wu_ci: str = "wu_ci",
        prefix_cu_mixed: str = "cu_mixed"
) -> None:
    """
    Save episode lists to disk in organized directory structure.

    Creates three subdirectories within output_dir, each containing individual
    JSON files for episodes. This structure enables efficient loading during
    training (e.g., using PyTorch DataLoader with file-based dataset).

    Directory Structure:
        output_dir/
        ├── wu_wi/
        │   ├── episode_000000.json
        │   ├── episode_000001.json
        │   └── ...
        ├── wu_ci/
        │   └── ...
        └── cu_mixed/
            └── ...

    Args:
        wu_wi_episodes: List of WU-WI episode dictionaries
        wu_ci_episodes: List of WU-CI episode dictionaries
        cu_mixed_episodes: List of CU-Mixed episode dictionaries
        output_dir: Root directory for saving episodes
        prefix_wu_wi: Subdirectory name for WU-WI episodes (default: "wu_wi")
        prefix_wu_ci: Subdirectory name for WU-CI episodes (default: "wu_ci")
        prefix_cu_mixed: Subdirectory name for CU-Mixed episodes (default: "cu_mixed")

    Notes:
        - Files are named with zero-padded indices (6 digits) for consistent sorting
        - JSON files use UTF-8 encoding and 2-space indentation for readability
        - Existing files in target directories will be overwritten
    """
    os.makedirs(output_dir, exist_ok=True)

    def _save_split(split_episodes: List[Dict[str, Any]], folder_name: str) -> None:
        """Helper function to save one episode split to disk."""
        folder = os.path.join(output_dir, folder_name)
        os.makedirs(folder, exist_ok=True)

        for i, episode in enumerate(split_episodes):
            filename = f"episode_{i:06d}.json"
            filepath = os.path.join(folder, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(episode, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(split_episodes)} episodes → {folder}")

    # Save all three splits
    _save_split(wu_wi_episodes, prefix_wu_wi)
    _save_split(wu_ci_episodes, prefix_wu_ci)
    _save_split(cu_mixed_episodes, prefix_cu_mixed)

    print("\n=== All episode splits saved successfully ===\n")


# ============================================================
# HELPER: SPLIT SUPPORT/QUERY FROM ITEM POOL
# ============================================================
def split_support_query_from_pool(
        item_list: List[str],
        score_list: List[float],
        n_sup: int,
        rng: random.Random,
) -> Tuple[List[str], List[float], List[str], List[float]]:
    """
    Split a list of items into support and query sets.

    This function implements the support/query partitioning for a single bag,
    using deterministic shuffling (given rng seed) to ensure reproducibility.

    The split strategy:
    - If bag has ≤ n_sup items: all go to support, query is empty
    - If bag has > n_sup items: randomly select n_sup for support, rest for query
    - Indices are sorted after selection to preserve original ordering within sets

    Args:
        item_list: List of item identifiers (ISBNs)
        score_list: List of scores (aligned with item_list)
        n_sup: Number of items to allocate to support set
        rng: Random number generator (for reproducible shuffling)

    Returns:
        Tuple containing:
        - items_sup: Support set items
        - scores_sup: Support set scores
        - items_qry: Query set items
        - scores_qry: Query set scores

    Notes:
        - If n_sup ≤ 0 or item_list is empty, returns empty support and all items as query
        - Indices are sorted after random selection to maintain deterministic ordering
        - This enables consistent support/query splits across multiple runs with same seed
    """
    # Handle edge cases
    if n_sup <= 0 or len(item_list) == 0:
        return [], [], list(item_list), list(score_list)

    if len(item_list) <= n_sup:
        # All items go to support; empty query
        return list(item_list), list(score_list), [], []

    # Randomly select n_sup items for support
    indices = list(range(len(item_list)))
    rng.shuffle(indices)

    # Sort selected indices to preserve original ordering
    sup_indices = sorted(indices[:n_sup])
    qry_indices = sorted(indices[n_sup:])

    # Extract items and scores
    items_sup = [item_list[i] for i in sup_indices]
    scores_sup = [score_list[i] for i in sup_indices]
    items_qry = [item_list[i] for i in qry_indices]
    scores_qry = [score_list[i] for i in qry_indices]

    return items_sup, scores_sup, items_qry, scores_qry


# ============================================================
# HELPER: MAP ISBN LIST TO DESCRIPTION KEYS
# ============================================================
def _isbn_list_to_keys(isbn_list: List[str], isbn2key: Dict[str, str]) -> List[str]:
    """
    Convert a list of ISBNs to human-readable item descriptions.

    This mapping enables text-based embeddings (Stage 1) while maintaining
    a connection to the original ISBN identifiers.

    Args:
        isbn_list: List of ISBN strings
        isbn2key: Mapping from ISBN to description string

    Returns:
        List of description strings (same length as isbn_list)

    Notes:
        - If an ISBN is missing from isbn2key, the raw ISBN is used as fallback
        - This should rarely happen if isbn2key was built from the same books_df
    """
    output: List[str] = []
    for isbn in isbn_list:
        key = isbn2key.get(isbn, isbn)  # Fallback to ISBN if mapping missing
        output.append(key)
    return output


# ============================================================
# HELPER: BUILD ISBN → DESCRIPTION MAPPING
# ============================================================
def build_isbn_to_description_mapping(books_df: pd.DataFrame) -> Dict[str, str]:
    """
    Create human-readable descriptions for each book.

    This function converts raw book metadata into natural language descriptions
    suitable for text embedding via language models (e.g., BERT, GPT).

    Description Format:
        "Title by Author, published by Publisher in Year."

    Examples:
        - "1984 by George Orwell, published by Penguin in 1949."
        - "The Great Gatsby by F. Scott Fitzgerald, in 1925."  (no publisher)
        - "Unknown Title."  (minimal metadata)

    Args:
        books_df: Cleaned books DataFrame from load_and_clean_books()

    Returns:
        Dictionary mapping ISBN → description string

    Notes:
        - Missing metadata (Author, Publisher, Year) is handled gracefully
        - Title is always included (required field from preprocessing)
        - Year is converted to string and validated (skipped if invalid)
    """
    isbn2key: Dict[str, str] = {}

    for _, row in books_df.iterrows():
        isbn = str(row["ISBN"])
        title = str(row.get("Title", "")).strip()

        # Parse optional metadata
        author = row.get("Author", None)
        publisher = row.get("Publisher", None)
        year_raw = row.get("Year", None)

        # Clean author and publisher (convert empty strings to None)
        author = None if pd.isna(author) or str(author).strip() == "" else str(author).strip()
        publisher = None if pd.isna(publisher) or str(publisher).strip() == "" else str(publisher).strip()

        # Convert year to string (validate numeric)
        year_str = None
        if not pd.isna(year_raw):
            try:
                year_str = str(int(year_raw))
            except:
                year_str = None

        # Build natural language description
        parts = [title]

        if author:
            parts.append(f"by {author}")

        if publisher:
            parts.append(f"published by {publisher}")

        if year_str:
            # Append year to publisher clause if it exists, otherwise separate
            if publisher:
                parts[-1] += f" in {year_str}"
            else:
                parts.append(f"in {year_str}")

        # Join with commas and add period
        description = ", ".join(parts) + "."

        # Store in mapping
        isbn2key[isbn] = description

    return isbn2key


# ============================================================
# MAIN FUNCTION: BUILD WARM/COLD EPISODES
# ============================================================
def build_warm_cold_episodes(
        user2items: Dict[str, List[str]],
        user2ratings: Dict[str, List[float]],
        user2target_attr: Dict[str, int],
        isbn2key: Dict[str, str],
        warm_user_ratio: float = 0.8,
        warm_item_ratio: float = 0.8,
        n_sup: int = 50,
        rng_seed: int = 12345,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate WU-WI, WU-CI, and CU-Mixed episodes with warm/cold splits.

    This is the core function that implements the episode construction strategy
    described in the UNISON paper. It performs a two-level random split (users
    and items) and generates episodes according to the rules for each scenario.

    Episode Generation Rules:
        WU-WI (Warm User-Warm Item):
            - User ∈ warm_users
            - User must have ≥ n_sup warm items
            - Support: n_sup randomly selected warm items
            - Query: Remaining warm items
            - Mask: All query items tagged as WU-WI (code 4)

        WU-CI (Warm User-Cold Item):
            - Same users as WU-WI (no new users added)
            - User must have ≥ 1 cold item
            - Support: Empty (tests generalization without adaptation)
            - Query: All cold items for that user
            - Mask: All query items tagged as WU-CI (code 3)

        CU-Mixed (Cold User-Mixed Items):
            - User ∈ cold_users (disjoint from WU-WI/WU-CI)
            - User must have ≥ n_sup warm items (for support)
            - Support: n_sup randomly selected warm items
            - Query: Remaining warm items + all cold items
            - Mask: Warm query items tagged as CU-WI (code 1),
                    cold query items tagged as CU-CI (code 2)

    Args:
        user2items: Dictionary mapping User-ID → [ISBN, ISBN, ...]
        user2ratings: Dictionary mapping User-ID → [rating, rating, ...]
                      (aligned with user2items)
        user2target_attr: Dictionary mapping User-ID → target_attr (0/1/-1)
        isbn2key: Dictionary mapping ISBN → human-readable description
        warm_user_ratio: Fraction of users to allocate to warm set (default: 0.8)
        warm_item_ratio: Fraction of items to allocate to warm set (default: 0.8)
        n_sup: Support set size for WU-WI and CU-Mixed episodes
        rng_seed: Random seed for reproducible splits

    Returns:
        Tuple containing:
        - wu_wi_episodes: List of WU-WI episode dictionaries
        - wu_ci_episodes: List of WU-CI episode dictionaries
        - cu_mixed_episodes: List of CU-Mixed episode dictionaries
        - stats: Dictionary with split statistics and episode counts

    Notes:
        - Users with target_attr=-1 (missing age) are skipped entirely
        - Episodes are created only for users meeting the minimum item requirements
        - The same random seed produces identical splits (reproducibility)
        - Statistics include drop counts to track filtering effects

    Example:
        >>> wu_wi, wu_ci, cu_mixed, stats = build_warm_cold_episodes(
        ...     user2items, user2ratings, user2target_attr, isbn2key,
        ...     n_sup=50, warm_user_ratio=0.8, warm_item_ratio=0.8
        ... )
        >>> print(f"Created {len(wu_wi)} WU-WI episodes")
        >>> print(f"Warm users: {stats['num_warm_users_initial']}")
        >>> print(f"Cold items: {stats['num_cold_items']}")
    """
    rng = random.Random(rng_seed)

    # ========== Step 1: Prepare user and item universes ==========
    all_users: List[str] = sorted(user2items.keys())
    all_items: Set[str] = {isbn for items in user2items.values() for isbn in items}
    all_items_list = sorted(list(all_items))

    # ========== Step 2: Random warm/cold split for users ==========
    n_users_total = len(all_users)
    n_warm_users = int(round(warm_user_ratio * n_users_total))
    n_warm_users = max(0, min(n_warm_users, n_users_total))

    warm_users_set = set(rng.sample(all_users, n_warm_users))
    cold_users_set = set(u for u in all_users if u not in warm_users_set)

    # ========== Step 3: Random warm/cold split for items (ISBNs) ==========
    """
    n_items_total = len(all_items_list)
    n_warm_items = int(round(warm_item_ratio * n_items_total))
    n_warm_items = max(0, min(n_warm_items, n_items_total))

    warm_items_set = set(rng.sample(all_items_list, n_warm_items))
    cold_items_set = set(i for i in all_items_list if i not in warm_items_set)
    """
    isbn_to_year = dict(zip(books_df['ISBN'], books_df['Year']))

    warm_items_set: Set[str] = {isbn for isbn in all_items_list if isbn_to_year.get(isbn, 9999) <= 1997}
    cold_items_set: Set[str] = {isbn for isbn in all_items_list if isbn_to_year.get(isbn, 0) >= 1998}

    # ========== Step 4: Initialize episode containers ==========
    wu_wi_episodes: List[Dict[str, Any]] = []
    wu_ci_episodes: List[Dict[str, Any]] = []
    cu_mixed_episodes: List[Dict[str, Any]] = []

    # Track which users successfully contributed episodes (after filtering)
    wu_wi_users: Set[str] = set()
    wu_ci_users: Set[str] = set()
    cu_mixed_users: Set[str] = set()

    # ========== Step 5: Build WU-WI and WU-CI episodes ==========
    for user_id in all_users:
        # Skip users with missing/invalid age (target_attr = -1)
        target_attr = user2target_attr.get(user_id, -1)
        if target_attr == -1:
            continue

        # Get user's items and ratings (in ISBN space)
        items = user2items[user_id]
        scores = user2ratings[user_id]

        # Partition user's items into warm/cold based on global warm_items_set
        user_warm_items: List[str] = []
        user_warm_scores: List[float] = []
        user_cold_items: List[str] = []
        user_cold_scores: List[float] = []

        for isbn, rating in zip(items, scores):
            if isbn in warm_items_set:
                user_warm_items.append(isbn)
                user_warm_scores.append(rating)
            else:
                user_cold_items.append(isbn)
                user_cold_scores.append(rating)

        # ---------- WU-WI Episode ----------
        # Requirements: user in warm set AND has enough warm items for support
        if user_id in warm_users_set and len(user_warm_items) >= n_sup:
            # Split warm items into support and query
            items_sup_isbn, scores_sup, items_qry_isbn, scores_qry = split_support_query_from_pool(
                user_warm_items, user_warm_scores, n_sup, rng
            )

            # Map ISBNs to human-readable descriptions
            items_sup_keys = _isbn_list_to_keys(items_sup_isbn, isbn2key)
            items_qry_keys = _isbn_list_to_keys(items_qry_isbn, isbn2key)

            episode = {
                "id": user_id,
                "scenario": "WU-WI",
                "items_sup": items_sup_keys,
                "scores_sup": scores_sup,
                "items_qry": items_qry_keys,
                "scores_qry": scores_qry,
                "mask_qry": [MASK_CODE["WU-WI"]] * len(items_qry_keys),
                "target_attr": target_attr,
            }
            wu_wi_episodes.append(episode)
            wu_wi_users.add(user_id)

        # ---------- WU-CI Episode ----------
        # Requirements: SAME users as WU-WI (already contributed WU-WI) AND has cold items
        # This design isolates the effect of cold items by holding user familiarity constant
        if user_id in wu_wi_users and len(user_cold_items) > 0:
            # Map cold ISBNs to descriptions
            items_qry_keys = _isbn_list_to_keys(user_cold_items, isbn2key)

            episode = {
                "id": user_id,
                "scenario": "WU-CI",
                "items_sup": [],  # Empty support tests generalization without adaptation
                "scores_sup": [],
                "items_qry": items_qry_keys,
                "scores_qry": list(user_cold_scores),
                "mask_qry": [MASK_CODE["WU-CI"]] * len(items_qry_keys),
                "target_attr": target_attr,
            }
            wu_ci_episodes.append(episode)
            wu_ci_users.add(user_id)

    # ========== Step 6: Build CU-Mixed episodes ==========
    for user_id in all_users:
        # Only process cold users (disjoint from WU-WI/WU-CI)
        if user_id not in cold_users_set:
            continue

        # Skip users with missing/invalid age
        target_attr = user2target_attr.get(user_id, -1)
        if target_attr == -1:
            continue

        items = user2items[user_id]
        scores = user2ratings[user_id]

        # Partition cold user's items into warm/cold
        user_warm_items: List[str] = []
        user_warm_scores: List[float] = []
        user_cold_items: List[str] = []
        user_cold_scores: List[float] = []

        for isbn, rating in zip(items, scores):
            if isbn in warm_items_set:
                user_warm_items.append(isbn)
                user_warm_scores.append(rating)
            else:
                user_cold_items.append(isbn)
                user_cold_scores.append(rating)

        # Requirement: need at least n_sup warm items for support
        if len(user_warm_items) < n_sup:
            continue

        # Split warm items into support and remaining
        items_sup_isbn, scores_sup, rem_warm_items_isbn, rem_warm_scores = split_support_query_from_pool(
            user_warm_items, user_warm_scores, n_sup, rng
        )

        # Query = remaining warm items (CU-WI) + all cold items (CU-CI)
        items_qry_isbn = rem_warm_items_isbn + user_cold_items
        scores_qry = rem_warm_scores + user_cold_scores

        # Tag query items with appropriate mask codes
        mask_qry = (
                [MASK_CODE["CU-WI"]] * len(rem_warm_items_isbn)
                + [MASK_CODE["CU-CI"]] * len(user_cold_items)
        )

        # Map ISBNs to descriptions
        items_sup_keys = _isbn_list_to_keys(items_sup_isbn, isbn2key)
        items_qry_keys = _isbn_list_to_keys(items_qry_isbn, isbn2key)

        episode = {
            "id": user_id,
            "scenario": "CU-Mixed",
            "items_sup": items_sup_keys,
            "scores_sup": scores_sup,
            "items_qry": items_qry_keys,
            "scores_qry": scores_qry,
            "mask_qry": mask_qry,
            "target_attr": target_attr,
        }
        cu_mixed_episodes.append(episode)
        cu_mixed_users.add(user_id)

    # ========== Step 7: Compile statistics ==========
    stats: Dict[str, Any] = {
        # Basic counts
        "num_users_total": len(all_users),
        "num_items_total": len(all_items),

        # Warm/cold splits (in ISBN space, before filtering)
        "warm_users_initial": sorted(list(warm_users_set)),
        "cold_users_initial": sorted(list(cold_users_set)),
        "warm_items": sorted(list(warm_items_set)),
        "cold_items": sorted(list(cold_items_set)),

        "num_warm_users_initial": len(warm_users_set),
        "num_cold_users_initial": len(cold_users_set),
        "num_warm_items": len(warm_items_set),
        "num_cold_items": len(cold_items_set),

        # After filtering for episode requirements
        "num_wu_wi_users": len(wu_wi_users),
        "num_wu_ci_users": len(wu_ci_users),
        "num_cu_mixed_users": len(cu_mixed_users),

        # Drops due to insufficient items or missing age
        "warm_users_dropped": len(warm_users_set) - len(wu_wi_users),
        "cold_users_dropped_for_cu_mixed": len(cold_users_set) - len(cu_mixed_users),

        # Episode counts
        "num_wu_wi_episodes": len(wu_wi_episodes),
        "num_wu_ci_episodes": len(wu_ci_episodes),
        "num_cu_mixed_episodes": len(cu_mixed_episodes),
    }

    return wu_wi_episodes, wu_ci_episodes, cu_mixed_episodes, stats


# ============================================================
# EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":
    """
    Example usage demonstrating the complete episode construction pipeline.

    This example:
    1. Loads and cleans the Book-Crossing dataset
    2. Builds ISBN → description mapping for text embeddings
    3. Generates episodes for multiple support set sizes (ablation study)
    4. Saves episodes to disk in organized directory structure

    Adjust file paths to match your local directory structure.
    """

    # ========== Step 1: Define file paths ==========
    books_path = "data/raw/Books.csv"
    users_path = "data/raw/Users.csv"
    ratings_path = "data/raw/Ratings.csv"

    # ========== Step 2: Load and clean base tables ==========
    print("Loading and cleaning Book-Crossing dataset...")
    books_df, valid_isbns = load_and_clean_books(books_path)
    users_df, valid_users = load_and_clean_users(users_path)
    ratings_df = load_and_clean_ratings(ratings_path, valid_isbns, valid_users)

    # ========== Step 3: Build user interaction dictionaries ==========
    user2items, user2ratings = build_user_interactions(
        ratings_df,
        min_items_per_user=1  # Keep all users; filtering happens during episode construction
    )

    # ========== Step 4: Compute age-based target attributes ==========
    DROP_USERS_WITH_MISSING_AGE = True  # Set to True for Stage 3 evaluation
    user2target_attr = compute_user_target_attr(
        users_df,
        drop_users_with_missing_age=DROP_USERS_WITH_MISSING_AGE,
    )

    # ========== Step 5: Build ISBN → description mapping ==========
    isbn2key = build_isbn_to_description_mapping(books_df)

    # Example description
    example_isbn = list(isbn2key.keys())[0]

    # ========== Step 6: Generate episodes for multiple N_SUP values ==========
    # This enables ablation studies on support set size
    N_SUP_OPTIONS = [5]

    for N_SUP in N_SUP_OPTIONS:
        print(f"\n{'=' * 60}")
        print(f"Building episodes with N_SUP = {N_SUP}")
        print(f"{'=' * 60}\n")

        # Build episodes
        wu_wi_eps, wu_ci_eps, cu_mixed_eps, stats = build_warm_cold_episodes(
            user2items=user2items,
            user2ratings=user2ratings,
            user2target_attr=user2target_attr,
            isbn2key=isbn2key,
            warm_user_ratio=0.8,
            warm_item_ratio=0.8,
            n_sup=N_SUP,
            rng_seed=12345,
        )

        # Save to disk
        output_dir = f"episodes_books/N_SUP_{N_SUP}"
        save_episode_splits(wu_wi_eps, wu_ci_eps, cu_mixed_eps, output_dir)

    print("\n" + "=" * 60)
    print("Episode construction complete!")
    print("=" * 60)