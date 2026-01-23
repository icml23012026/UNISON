"""
UNISON Episode Dataset and DataLoader for Book-Crossing

This module provides PyTorch Dataset and DataLoader utilities for loading
UNISON episodes with pre-computed item embeddings from Stage 1 encoders.

Dataset Design:
    The dataset maps human-readable item descriptions to dense embeddings
    computed offline by text encoders (e.g., BERT, GPT, Qwen). This two-stage
    approach (offline embedding + online loading) enables:

    1. Flexibility: Swap encoders without regenerating episodes
    2. Efficiency: Avoid repeated encoding during training
    3. Reproducibility: Fixed embeddings ensure consistent experiments

    Each episode is loaded as a dictionary containing:
    - items_sup: Support set item embeddings [N_sup, emb_dim]
    - scores_sup: Support set ratings [N_sup]
    - items_qry: Query set item embeddings [Q, emb_dim] (variable length)
    - scores_qry: Query set ratings [Q]
    - mask_qry: Query item mask codes [Q] (1=CU-WI, 2=CU-CI, 3=WU-CI, 4=WU-WI)
    - target_attr: Bag-level attribute (0/1 for age, -1 if unknown)

Episode Storage Formats:
    The dataset supports three storage formats for flexibility:

    1. Directory of JSON files (recommended for large datasets):
       episodes/wu_wi/
       ├── episode_000000.json
       ├── episode_000001.json
       └── ...

    2. Single JSONL file (all episodes concatenated):
       episodes/wu_wi.jsonl

    3. Single JSON file (for small datasets):
       episodes/wu_wi.json

Embedding Storage:
    Item embeddings are stored in a pickle file as a dictionary:
        {
            "Title by Author, published by Publisher in Year.": np.ndarray([...]),
            ...
        }

    The keys must exactly match the "items_sup" and "items_qry" strings in
    the episode JSON files.

Missing Embeddings:
    If an item description is not found in the embedding dictionary:
    - A warning is printed (once per episode)
    - A zero vector is inserted as a placeholder
    - This preserves alignment between items and scores
    - Training code can optionally filter episodes with missing embeddings

Padding Strategy:
    - Support sets: Fixed size N_sup (or 0 for WU-CI episodes)
    - Query sets: Variable length → padded to max length in batch
    - Padding value: 0.0 for embeddings, 0 for masks

    This enables efficient batching while accommodating different episode types.

Typical Usage:
    # Create DataLoader for each episode type
    train_loader = create_dataloader(
        path="episodes_books/N_SUP_50/wu_wi",
        embedding_pkl_path="embeddings/item2vec_qwen.pkl",
        batch_size=16,
        shuffle=True,
        num_workers=4
    )

    val_wu_loader = create_dataloader(
        path="episodes_books/N_SUP_50/wu_ci",
        embedding_pkl_path="embeddings/item2vec_qwen.pkl",
        batch_size=16,
        shuffle=False
    )

    val_cu_loader = create_dataloader(
        path="episodes_books/N_SUP_50/cu_mixed",
        embedding_pkl_path="embeddings/item2vec_qwen.pkl",
        batch_size=16,
        shuffle=False
    )

    # Training loop
    for batch in train_loader:
        items_sup = batch["items_sup"]      # [B, N_sup, emb_dim]
        scores_sup = batch["scores_sup"]    # [B, N_sup]
        items_qry = batch["items_qry"]      # [B, max_Q, emb_dim]
        scores_qry = batch["scores_qry"]    # [B, max_Q]
        mask_qry = batch["mask_qry"]        # [B, max_Q]
        target_attr = batch["target_attr"]  # [B]

        # Forward pass through UNISON model
        outputs = model.train_step_warm_user_warm_item(
            optimizer, batch["id"], items_sup, scores_sup
        )

Performance Considerations:
    - Use num_workers > 0 for parallel data loading (speeds up I/O)
    - Set pin_memory=True when using GPU (faster host→device transfer)
    - Consider caching embeddings in RAM for small datasets
    - For very large datasets, use memory-mapped arrays instead of pickle

Notes:
    - The term "user" in variable names is a convention from the recommendation
      domain but applies generally to any "bag" in the UNISON framework
    - Mask codes enable fine-grained evaluation (e.g., separate CU-WI vs CU-CI metrics)
    - Target attributes with value -1 should be filtered out for Stage 3 evaluation
"""

import os
import json
import glob
import pickle
from typing import List, Dict, Union

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ============================================================
# HELPER: LOAD EPISODE PATHS OR JSONL DATA
# ============================================================
def _load_episode_paths_or_jsonl(path: str):
    """
    Load episode file paths or JSONL data based on input path type.

    This function provides a unified interface for three episode storage formats:
    1. Directory: Returns list of episode_*.json file paths
    2. JSONL file: Returns list of parsed episode dictionaries (in-memory)
    3. Single JSON: Returns single-item list with that file path

    Args:
        path: Directory path, .jsonl file path, or .json file path

    Returns:
        Tuple of (episode_paths, jsonl_data):
        - If directory/JSON: (List[str] paths, None)
        - If JSONL: (None, List[Dict] episodes)

    Raises:
        FileNotFoundError: If directory contains no episode_*.json files
        ValueError: If JSONL is empty or path format is unsupported
    """
    if os.path.isdir(path):
        # Directory: collect all episode_*.json files
        files = sorted(glob.glob(os.path.join(path, "episode_*.json")))
        if not files:
            raise FileNotFoundError(f"No episode_*.json files in directory: {path}")
        return files, None  # (paths, jsonl_data)

    if path.endswith(".jsonl"):
        # JSONL: load all episodes into memory
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        if not data:
            raise ValueError(f"Empty JSONL file: {path}")
        return None, data  # (paths, jsonl_data)

    if path.endswith(".json"):
        # Single JSON file
        return [path], None

    raise ValueError(
        f"Unsupported path: {path}. "
        f"Provide a directory, a .jsonl file, or a single .json file."
    )


# ============================================================
# DATASET CLASS
# ============================================================
class EpisodesDataset(Dataset):
    """
    PyTorch Dataset for UNISON episodes with pre-computed item embeddings.

    This dataset loads episode JSON files and maps item description strings to
    dense embeddings from a pre-computed dictionary. It handles three storage
    formats (directory, JSONL, single JSON) and gracefully handles missing
    embeddings with zero vectors.

    Parameters
    ----------
    episodes_path : str
        Path to episodes (directory with episode_*.json, .jsonl file, or .json file)
    embedding_pkl_path : str
        Path to pickle file containing item_description → embedding mapping

    Attributes
    ----------
    item2vec : Dict[str, torch.Tensor]
        Mapping from item description strings to embedding tensors
    emb_dim : int
        Embedding dimensionality (inferred from first embedding)

    Raises
    ------
    FileNotFoundError
        If episodes_path doesn't exist or contains no valid files
    ValueError
        If embedding dictionary is empty

    Example
    -------
    >>> dataset = EpisodesDataset(
    ...     episodes_path="episodes_books/N_SUP_50/wu_wi",
    ...     embedding_pkl_path="embeddings/item2vec_qwen.pkl"
    ... )
    >>> print(f"Dataset size: {len(dataset)}")
    >>> episode = dataset[0]
    >>> print(f"Support set shape: {episode['items_sup'].shape}")
    """

    def __init__(self, episodes_path: str, embedding_pkl_path: str):
        # Load episode sources (file paths or in-memory JSONL data)
        self._episode_paths, self._jsonl_data = _load_episode_paths_or_jsonl(episodes_path)

        # Load item embedding dictionary (description → numpy array)
        with open(embedding_pkl_path, "rb") as f:
            emb_dict_np: Dict[str, "np.ndarray"] = pickle.load(f)

        if not emb_dict_np:
            raise ValueError(f"Empty embedding dictionary loaded from: {embedding_pkl_path}")

        # Convert numpy arrays to PyTorch tensors (once, at initialization)
        # This avoids repeated conversion during training
        self.item2vec: Dict[str, torch.Tensor] = {
            key: torch.tensor(vec, dtype=torch.float32)
            for key, vec in emb_dict_np.items()
        }

        # Infer embedding dimension from first entry
        first_vec = next(iter(self.item2vec.values()))
        self.emb_dim = int(first_vec.shape[0])

    def __len__(self) -> int:
        """Return number of episodes in dataset."""
        if self._jsonl_data is not None:
            return len(self._jsonl_data)
        return len(self._episode_paths)

    def _load_episode_obj(self, idx: int) -> Dict:
        """
        Load a single episode dictionary from disk or memory.

        Args:
            idx: Episode index

        Returns:
            Episode dictionary with keys: id, scenario, items_sup, scores_sup,
            items_qry, scores_qry, mask_qry, target_attr
        """
        if self._jsonl_data is not None:
            # JSONL: already in memory
            return self._jsonl_data[idx]

        # JSON file: load from disk
        path = self._episode_paths[idx]
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _keys_to_embeddings(
        self,
        keys: List[str],
        episode_id: Union[int, str],
        split_name: str
    ) -> torch.Tensor:
        """
        Map a list of item description keys to stacked embedding tensor.

        This function handles missing embeddings by inserting zero vectors and
        printing a warning (once per episode/split) to alert the user.

        Args:
            keys: List of item description strings
            episode_id: Episode identifier (for warning messages)
            split_name: "support" or "query" (for warning messages)

        Returns:
            Stacked embedding tensor [len(keys), emb_dim]
            If keys is empty, returns [0, emb_dim] tensor

        Notes:
            - Zero vectors are used as placeholders to preserve alignment
            - Multiple missing embeddings in one episode trigger a single warning
            - Consider filtering episodes with missing embeddings in training code
        """
        if len(keys) == 0:
            # Empty split (e.g., WU-CI episodes have no support) → [0, emb_dim]
            return torch.empty(0, self.emb_dim, dtype=torch.float32)

        vecs = []
        missing = []

        for key in keys:
            vec = self.item2vec.get(key)
            if vec is None:
                missing.append(key)
                # Insert zero embedding to preserve alignment with scores
                vecs.append(torch.zeros(self.emb_dim, dtype=torch.float32))
            else:
                vecs.append(vec)

        # Print warning if any embeddings were missing (once per episode/split)
        if missing:
            print(
                f"[WARN] Episode {episode_id}: {len(missing)} missing {split_name} embeddings. "
                f"Example: {missing[0]}"
            )

        return torch.stack(vecs, dim=0)  # [len(keys), emb_dim]

    def __getitem__(self, idx: int) -> Dict:
        """
        Load and process a single episode.

        Args:
            idx: Episode index

        Returns:
            Dictionary containing:
            - id: Episode identifier (user ID)
            - emb_dim: Embedding dimension (for collate safety)
            - items_sup_keys: Raw support item descriptions (for debugging)
            - items_qry_keys: Raw query item descriptions (for debugging)
            - items_sup: Support embeddings [N_sup, emb_dim] or [0, emb_dim]
            - scores_sup: Support ratings [N_sup] or [0]
            - items_qry: Query embeddings [Q, emb_dim] or [0, emb_dim]
            - scores_qry: Query ratings [Q] or [0]
            - mask_qry: Query mask codes [Q] or [0]
            - target_attr: Bag-level attribute (scalar)
        """
        # Load episode JSON
        episode = self._load_episode_obj(idx)
        episode_id = episode["id"]

        # Extract raw data from episode
        sup_keys: List[str] = episode.get("items_sup", [])
        qry_keys: List[str] = episode.get("items_qry", [])
        sup_scores = episode.get("scores_sup", [])
        qry_scores = episode.get("scores_qry", [])
        mask_qry = episode.get("mask_qry", [])

        # Convert item description keys to embedding tensors
        items_sup = self._keys_to_embeddings(sup_keys, episode_id, "support")
        items_qry = self._keys_to_embeddings(qry_keys, episode_id, "query")

        # Convert scores and masks to tensors
        scores_sup_t = (
            torch.tensor(sup_scores, dtype=torch.float32)
            if len(sup_scores) > 0
            else torch.empty(0, dtype=torch.float32)
        )
        scores_qry_t = (
            torch.tensor(qry_scores, dtype=torch.float32)
            if len(qry_scores) > 0
            else torch.empty(0, dtype=torch.float32)
        )
        mask_qry_t = (
            torch.tensor(mask_qry, dtype=torch.long)
            if len(mask_qry) > 0
            else torch.empty(0, dtype=torch.long)
        )

        # Convert target attribute (age bin or -1 if unknown)
        target_attr = episode.get("target_attr", None)
        if target_attr is None:
            target_attr_t = torch.tensor(-1, dtype=torch.long)
        else:
            target_attr_t = torch.tensor(int(target_attr), dtype=torch.long)

        return {
            "id": episode_id,
            "emb_dim": self.emb_dim,          # Expose for collate safety check
            "items_sup_keys": sup_keys,       # Keep raw keys for debugging
            "items_qry_keys": qry_keys,
            "items_sup": items_sup,           # [N_sup, emb_dim] or [0, emb_dim]
            "scores_sup": scores_sup_t,       # [N_sup] or [0]
            "items_qry": items_qry,           # [Q, emb_dim] or [0, emb_dim]
            "scores_qry": scores_qry_t,       # [Q] or [0]
            "mask_qry": mask_qry_t,           # [Q] or [0]
            "target_attr": target_attr_t,     # Scalar (age bin 0/1 or -1)
        }


# ============================================================
# COLLATE FUNCTION
# ============================================================
def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for batching variable-length episodes.

    This function handles the complexity of batching episodes with:
    - Fixed-size support sets (N_sup items per episode, or 0 for WU-CI)
    - Variable-length query sets (different Q per episode)
    - Padding query sets to max length in batch

    Padding Strategy:
        - Support sets: Stacked directly (assumed same size across batch)
        - Query sets: Padded with zeros to max length in batch
        - Scores/masks: Padded with zeros (score=0, mask=0 indicate padding)

    Args:
        batch: List of episode dictionaries from __getitem__

    Returns:
        Batched dictionary containing:
        - id: List of episode IDs [B]
        - items_sup_keys: List of support key lists [B x [N_sup]]
        - items_qry_keys: List of query key lists [B x [Q_i]]
        - items_sup: Batched support embeddings [B, N_sup, emb_dim] or [B, 0, emb_dim]
        - scores_sup: Batched support scores [B, N_sup] or [B, 0]
        - items_qry: Padded query embeddings [B, max_Q, emb_dim]
        - scores_qry: Padded query scores [B, max_Q]
        - mask_qry: Padded query masks [B, max_Q]
        - target_attr: Batched attributes [B]

    Notes:
        - Empty support (WU-CI) results in [B, 0, emb_dim] tensor (no padding needed)
        - Empty query (degenerate case) results in [B, 0, emb_dim] tensor
        - Padding value 0 is safe since mask=0 indicates padding and can be ignored

    Example:
        >>> loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
        >>> batch = next(iter(loader))
        >>> print(batch["items_sup"].shape)  # [4, 50, 768] for N_sup=50
        >>> print(batch["items_qry"].shape)  # [4, max_Q, 768] where max_Q varies
    """
    B = len(batch)
    emb_dim = batch[0]["emb_dim"]

    # Extract metadata (not batched)
    ids = [b["id"] for b in batch]
    items_sup_keys = [b["items_sup_keys"] for b in batch]
    items_qry_keys = [b["items_qry_keys"] for b in batch]

    # Batch target attributes (fixed-size scalar per episode)
    target_attr = torch.stack([b["target_attr"] for b in batch], dim=0)  # [B]

    # ========== SUPPORT SET BATCHING ==========
    # Support sets are assumed fixed-size (N_sup) per episode type, or empty (WU-CI)
    sup_list = [b["items_sup"] for b in batch]          # List of [N_sup, emb_dim] or [0, emb_dim]
    sup_scores_list = [b["scores_sup"] for b in batch]  # List of [N_sup] or [0]

    # Check if all support sets are empty (e.g., WU-CI episodes)
    all_empty_sup = all(x.numel() == 0 for x in sup_list)

    if all_empty_sup:
        # All support sets empty → create [B, 0, emb_dim] tensor
        items_sup = torch.empty(B, 0, emb_dim, dtype=torch.float32)
        scores_sup = torch.empty(B, 0, dtype=torch.float32)
    else:
        # Stack support sets (assumes consistent N_sup across batch)
        items_sup = torch.stack(sup_list, dim=0)         # [B, N_sup, emb_dim]
        scores_sup = torch.stack(sup_scores_list, dim=0) # [B, N_sup]

    # ========== QUERY SET BATCHING (with padding) ==========
    # Query sets have variable length → pad to max length in batch
    qry_list = [b["items_qry"] for b in batch]          # List of [Q_i, emb_dim]
    qry_scores_list = [b["scores_qry"] for b in batch]  # List of [Q_i]
    qry_mask_list = [b["mask_qry"] for b in batch]      # List of [Q_i]

    if any(x.numel() > 0 for x in qry_list):
        # Pad query sequences to max length in batch
        items_qry = pad_sequence(
            qry_list,
            batch_first=True,
            padding_value=0.0
        )  # [B, max_Q, emb_dim]

        scores_qry = pad_sequence(
            qry_scores_list,
            batch_first=True,
            padding_value=0.0
        )  # [B, max_Q]

        mask_qry = pad_sequence(
            qry_mask_list,
            batch_first=True,
            padding_value=0
        )  # [B, max_Q]
    else:
        # All query sets empty (degenerate case) → create [B, 0, emb_dim] tensor
        items_qry = torch.empty(B, 0, emb_dim, dtype=torch.float32)
        scores_qry = torch.empty(B, 0, dtype=torch.float32)
        mask_qry = torch.empty(B, 0, dtype=torch.long)

    return {
        "id": ids,
        "items_sup_keys": items_sup_keys,
        "items_qry_keys": items_qry_keys,
        "items_sup": items_sup,       # [B, N_sup, emb_dim] or [B, 0, emb_dim]
        "scores_sup": scores_sup,     # [B, N_sup] or [B, 0]
        "items_qry": items_qry,       # [B, max_Q, emb_dim] or [B, 0, emb_dim]
        "scores_qry": scores_qry,     # [B, max_Q] or [B, 0]
        "mask_qry": mask_qry,         # [B, max_Q] or [B, 0]
        "target_attr": target_attr,   # [B]
    }


# ============================================================
# CONVENIENCE DATALOADER FACTORY
# ============================================================
def create_dataloader(
    path: str,
    embedding_pkl_path: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False
) -> DataLoader:
    """
    Create a PyTorch DataLoader for UNISON episodes.

    This convenience function wraps dataset creation and DataLoader initialization
    with sensible defaults for the UNISON framework.

    Args:
        path: Path to episodes (directory, .jsonl, or .json)
        embedding_pkl_path: Path to item embedding pickle file
        batch_size: Number of episodes per batch
        shuffle: Whether to shuffle episodes (True for training, False for eval)
        num_workers: Number of parallel data loading workers (default: 0 = main process)
        pin_memory: Whether to pin memory for faster GPU transfer (default: False)

    Returns:
        PyTorch DataLoader ready for iteration

    Performance Tips:
        - Set num_workers > 0 (e.g., 4-8) to parallelize disk I/O
        - Set pin_memory=True when using GPU to speed up host→device transfer
        - Avoid num_workers > 0 on Windows (multiprocessing issues)
        - For small datasets that fit in RAM, num_workers=0 may be faster

    Example:
        >>> train_loader = create_dataloader(
        ...     path="episodes_books/N_SUP_50/wu_wi",
        ...     embedding_pkl_path="embeddings/item2vec_qwen.pkl",
        ...     batch_size=16,
        ...     shuffle=True,
        ...     num_workers=4,
        ...     pin_memory=True  # If using GPU
        ... )
        >>> for batch in train_loader:
        ...     # Training loop
        ...     pass
    """
    dataset = EpisodesDataset(path, embedding_pkl_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )


# ============================================================
# EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":
    """
    Example usage demonstrating dataset initialization and data exploration.
    
    This example:
    1. Creates datasets for all three episode types (WU-WI, WU-CI, CU-Mixed)
    2. Prints dataset statistics
    3. Analyzes user→num_books distribution
    4. Creates DataLoaders and inspects a sample batch
    
    Adjust paths to match your local directory structure.
    """

    # ========== Configuration ==========
    config = {
        "train_data_path": "episodes_books/N_SUP_10/wu_wi",      # WU-WI episodes
        "val_wu_data_path": "episodes_books/N_SUP_40/wu_ci",     # WU-CI episodes
        "val_cu_data_path": "episodes_books/N_SUP_40/cu_mixed",  # CU-Mixed episodes
        "embedding_path": "embeddings/item2vec_qwen.pkl",
        "batch_size": 16,
        "num_workers": 0,
    }

    # ========== Step 1: Create datasets ==========
    print("Creating datasets...")
    train_dataset = EpisodesDataset(config["train_data_path"], config["embedding_path"])
    val_wu_dataset = EpisodesDataset(config["val_wu_data_path"], config["embedding_path"])
    val_cu_dataset = EpisodesDataset(config["val_cu_data_path"], config["embedding_path"])

    print(f"WU-WI episodes:    {len(train_dataset)}")
    print(f"WU-CI episodes:    {len(val_wu_dataset)}")
    print(f"CU-mixed episodes: {len(val_cu_dataset)}")
    print()

    # ========== Step 2: Analyze user→num_books distribution ==========
    print("Analyzing user interaction statistics...")
    user_num_books = {}

    for ds in [train_dataset, val_wu_dataset, val_cu_dataset]:
        for i in range(len(ds)):
            episode = ds._load_episode_obj(i)
            user_id = str(episode["id"])

            # Count unique books across support and query
            all_items = episode.get("items_sup", []) + episode.get("items_qry", [])
            num_books = len(set(all_items))

            # Keep maximum count per user (across all episodes)
            prev = user_num_books.get(user_id, 0)
            if num_books > prev:
                user_num_books[user_id] = num_books

    # Save statistics to JSON for plotting
    output_json_path = "user_num_books.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(user_num_books, f, ensure_ascii=False, indent=2)

    print(f"Saved user→num_books statistics to: {output_json_path}")
    print()

    # ========== Step 3: Create DataLoaders ==========
    print("=" * 60)
    print("CREATING DATALOADERS")
    print("=" * 60)

    train_loader = create_dataloader(
        config["train_data_path"],
        config["embedding_path"],
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )

    val_wu_loader = create_dataloader(
        config["val_wu_data_path"],
        config["embedding_path"],
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    val_cu_loader = create_dataloader(
        config["val_cu_data_path"],
        config["embedding_path"],
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    # ========== Step 4: Inspect a sample batch ==========
    print("\nInspecting first batch from training loader...")
    batch = next(iter(train_loader))

    print(f"\nBatch keys: {list(batch.keys())}")
    print(f"\nExample user IDs: {batch['id'][:3]}")
    print(f"Example item keys (first user, support): {batch['items_sup_keys'][0][:3]}")
    print(f"\nBatch shapes:")
    print(f"  items_sup:  {batch['items_sup'].shape}")   # [B, N_sup, emb_dim]
    print(f"  scores_sup: {batch['scores_sup'].shape}")  # [B, N_sup]
    print(f"  items_qry:  {batch['items_qry'].shape}")   # [B, max_Q, emb_dim]
    print(f"  scores_qry: {batch['scores_qry'].shape}")  # [B, max_Q]
    print(f"  mask_qry:   {batch['mask_qry'].shape}")    # [B, max_Q]
    print(f"  target_attr: {batch['target_attr'].shape}") # [B]

    print(f"\nExample target attributes: {batch['target_attr'][:5]}")
    print(f"Example query masks (first user): {batch['mask_qry'][0][:10]}")

    print("\n" + "=" * 60)
    print("Dataset and DataLoader setup complete!")
    print("=" * 60)