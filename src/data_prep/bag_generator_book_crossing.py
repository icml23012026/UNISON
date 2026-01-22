"""
UNISON Framework - Bag Generation
Logic:
1. Split items/users into Warm and Cold pools.
2. WU-WI: Support (Warm) -> Query (Remaining Warm).
3. WU-CI: Support (Empty) -> Query (Cold items).
4. CU-Mixed: Support (Warm) -> Query (Mixed Warm/Cold).
"""

import os
import json
import random
import pickle
import logging
import argparse

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MASK_CODE = {"CU-WI": 1, "CU-CI": 2, "WU-CI": 3, "WU-WI": 4}

def split_support_query_from_pool(item_list, score_list, n_sup, rng):
    if n_sup <= 0 or len(item_list) <= n_sup:
        return item_list, score_list, [], []

    idxs = list(range(len(item_list)))
    rng.shuffle(idxs)

    sup_idx = sorted(idxs[:n_sup])
    qry_idx = sorted(idxs[n_sup:])

    return ([item_list[i] for i in sup_idx], [score_list[i] for i in sup_idx],
            [item_list[i] for i in qry_idx], [score_list[i] for i in qry_idx])

def build_bags(u2i, u2r, u2attr, i_meta, n_sup, w_u_ratio=0.8, w_i_ratio=0.8, seed=12345):
    rng = random.Random(seed)
    all_users = sorted(u2i.keys())
    all_items = sorted(list({i for items in u2i.values() for i in items}))

    warm_users = set(rng.sample(all_users, int(round(w_u_ratio * len(all_users)))))
    warm_items = set(rng.sample(all_items, int(round(w_i_ratio * len(all_items)))))

    splits = {"wu_wi": [], "wu_ci": [], "cu_mixed": []}
    wu_wi_ids = set()

    for uid in all_users:
        attr = u2attr.get(uid, -1)
        if attr == -1: continue

        u_warm_i, u_warm_s, u_cold_i, u_cold_s = [], [], [], []
        for i, s in zip(u2i[uid], u2r[uid]):
            if i in warm_items:
                u_warm_i.append(i); u_warm_s.append(s)
            else:
                u_cold_i.append(i); u_cold_s.append(s)

        # WU-WI & WU-CI
        if uid in warm_users and len(u_warm_i) >= n_sup:
            i_s, s_s, i_q, s_q = split_support_query_from_pool(u_warm_i, u_warm_s, n_sup, rng)
            splits["wu_wi"].append({
                "id": uid, "scenario": "WU-WI",
                "items_sup": [i_meta.get(x, x) for x in i_s], "scores_sup": s_s,
                "items_qry": [i_meta.get(x, x) for x in i_q], "scores_qry": s_q,
                "mask_qry": [MASK_CODE["WU-WI"]] * len(i_q), "target_attr": attr
            })
            wu_wi_ids.add(uid)

            if u_cold_i:
                splits["wu_ci"].append({
                    "id": uid, "scenario": "WU-CI",
                    "items_sup": [], "scores_sup": [], # Empty Support per your logic
                    "items_qry": [i_meta.get(x, x) for x in u_cold_i], "scores_qry": u_cold_s,
                    "mask_qry": [MASK_CODE["WU-CI"]] * len(u_cold_i), "target_attr": attr
                })

        # CU-Mixed
        elif uid not in warm_users and len(u_warm_i) >= n_sup:
            i_s, s_s, i_rem_w, s_rem_w = split_support_query_from_pool(u_warm_i, u_warm_s, n_sup, rng)
            i_q = i_rem_w + u_cold_i
            s_q = s_rem_w + u_cold_s
            mask = ([MASK_CODE["CU-WI"]] * len(i_rem_w)) + ([MASK_CODE["CU-CI"]] * len(u_cold_i))

            splits["cu_mixed"].append({
                "id": uid, "scenario": "CU-Mixed",
                "items_sup": [i_meta.get(x, x) for x in i_s], "scores_sup": s_s,
                "items_qry": [i_meta.get(x, x) for x in i_q], "scores_qry": s_q,
                "mask_qry": mask, "target_attr": attr
            })

    return splits

def parse_args():
    parser = argparse.ArgumentParser(description="Build episodes for UNISON")
    parser.add_argument("--n_sup", type=int, default=10, help="Number of support items")
    parser.add_argument("--data_in", type=str, default="data/processed_bags", help="Input directory")
    return parser.parse_args()


if __name__ == "__main__":
    # Load your pickles
    data_in = "data/processed_bags"
    with open(f"{data_in}/user2items.pkl", "rb") as f: u2i = pickle.load(f)
    with open(f"{data_in}/user2ratings.pkl", "rb") as f: u2r = pickle.load(f)
    with open(f"{data_in}/user2attr.pkl", "rb") as f: u2attr = pickle.load(f)
    with open(f"{data_in}/item_metadata.pkl", "rb") as f: i_meta = pickle.load(f)

    # Generate
    N_SUP = 10
    out_dir = f"data/episodes/N_SUP_{N_SUP}"
    logger.info(f"Building episodes (N={N_SUP})...")

    all_bags = build_bags(u2i, u2r, u2attr, i_meta, n_sup=N_SUP)

    # Save
    for name, data in all_bags.items():
        path = os.path.join(out_dir, name)
        os.makedirs(path, exist_ok=True)
        for i, ep in enumerate(data):
            with open(os.path.join(path, f"episode_{i:06d}.json"), "w") as f:
                json.dump(ep, f, indent=2)
    logger.info("Done.")