"""
UNISON Framework - BookCrossing Bag Dataset
Handles loading user interaction bags and mapping items to pre-computed embeddings.
Supports variable query lengths through padding.
"""

import os
import json
import glob
import pickle
import torch
import logging
from typing import List, Dict, Union, Tuple
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BookCrossingBagDataset(Dataset):
    """
    Dataset class specifically for BookCrossing interaction bags.
    Maps book description keys to embeddings from a pickle dictionary.
    """
    def __init__(self, bags_path: str, embedding_pkl_path: str):
        # 1. Load bag sources (directory, jsonl, or single json)
        self._bag_paths, self._jsonl_data = self._load_bag_sources(bags_path)

        # 2. Load item2vec embeddings
        with open(embedding_pkl_path, "rb") as f:
            emb_dict_np = pickle.load(f)

        if not emb_dict_np:
            raise ValueError(f"Empty embedding dict: {embedding_pkl_path}")

        # Convert to torch tensors once for performance
        self.item2vec = {
            k: torch.tensor(v, dtype=torch.float32) for k, v in emb_dict_np.items()
        }

        # Infer embedding dimension
        first_vec = next(iter(self.item2vec.values()))
        self.emb_dim = int(first_vec.shape[0])

    def _load_bag_sources(self, path: str):
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "bag_*.json")))
            if not files:
                files = sorted(glob.glob(os.path.join(path, "*.json")))
            return files, None

        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f if line.strip()]
            return None, data

        return [path], None

    def __len__(self):
        return len(self._jsonl_data) if self._jsonl_data else len(self._bag_paths)

    def _load_bag_obj(self, idx: int) -> Dict:
        if self._jsonl_data is not None:
            return self._jsonl_data[idx]
        with open(self._bag_paths[idx], "r", encoding="utf-8") as f:
            return json.load(f)

    def _keys_to_embeddings(self, keys: List[str], bag_id: Union[int, str], split_name: str) -> torch.Tensor:
        if not keys:
            return torch.empty(0, self.emb_dim, dtype=torch.float32)

        vecs, missing = [], []
        for k in keys:
            v = self.item2vec.get(k)
            if v is None:
                missing.append(k)
                vecs.append(torch.zeros(self.emb_dim, dtype=torch.float32))
            else:
                vecs.append(v)

        if missing:
            logger.warning(f"Bag {bag_id}: {len(missing)} missing {split_name} embeddings.")

        return torch.stack(vecs, dim=0)

    def __getitem__(self, idx):
        bag = self._load_bag_obj(idx)
        bag_id = bag["id"]

        # Support & Query Keys
        sup_keys = bag.get("items_sup", [])
        qry_keys = bag.get("items_qry", [])

        # Mapping to Tensors
        items_sup = self._keys_to_embeddings(sup_keys, bag_id, "support")
        items_qry = self._keys_to_embeddings(qry_keys, bag_id, "query")

        # Scores & Masks
        scores_sup_t = torch.tensor(bag.get("scores_sup", []), dtype=torch.float32)
        scores_qry_t = torch.tensor(bag.get("scores_qry", []), dtype=torch.float32)
        mask_qry_t = torch.tensor(bag.get("mask_qry", []), dtype=torch.long)

        # Target Attribute (Age Bin)
        target_attr = bag.get("target_attr")
        target_attr_t = torch.tensor(int(target_attr) if target_attr is not None else -1, dtype=torch.long)

        return {
            "id": bag_id,
            "emb_dim": self.emb_dim,
            "items_sup": items_sup,
            "scores_sup": scores_sup_t,
            "items_qry": items_qry,
            "scores_qry": scores_qry_t,
            "mask_qry": mask_qry_t,
            "target_attr": target_attr_t,
        }

def collate_bags(batch: List[Dict]):
    """
    Collate function for BookCrossing bags.
    Pads variable-length query sequences and handles empty support sets.
    """
    B = len(batch)
    emb_dim = batch[0]["emb_dim"]

    # 1. Support Processing
    sup_list = [b["items_sup"] for b in batch]
    sup_scores_list = [b["scores_sup"] for b in batch]

    if all(x.numel() == 0 for x in sup_list):
        items_sup = torch.empty(B, 0, emb_dim, dtype=torch.float32)
        scores_sup = torch.empty(B, 0, dtype=torch.float32)
    else:
        items_sup = torch.stack(sup_list, dim=0)
        scores_sup = torch.stack(sup_scores_list, dim=0)

    # 2. Query Processing (Padded)
    qry_list = [b["items_qry"] for b in batch]
    if any(x.numel() > 0 for x in qry_list):
        items_qry = pad_sequence(qry_list, batch_first=True, padding_value=0.0)
        scores_qry = pad_sequence([b["scores_qry"] for b in batch], batch_first=True, padding_value=0.0)
        mask_qry = pad_sequence([b["mask_qry"] for b in batch], batch_first=True, padding_value=0)
    else:
        items_qry = torch.empty(B, 0, emb_dim, dtype=torch.float32)
        scores_qry = torch.empty(B, 0, dtype=torch.float32)
        mask_qry = torch.empty(B, 0, dtype=torch.long)

    return {
        "id": [b["id"] for b in batch],
        "items_sup": items_sup,
        "scores_sup": scores_sup,
        "items_qry": items_qry,
        "scores_qry": scores_qry,
        "mask_qry": mask_qry,
        "target_attr": torch.stack([b["target_attr"] for b in batch], dim=0),
    }

def create_bookcrossing_loader(path, emb_path, batch_size, shuffle=True):
    ds = BookCrossingBagDataset(path, emb_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_bags)