# UNISON

## 1. Setup

Before running the scripts, ensure your data is organized as follows:

### Raw Dataset
Place the original Book-Crossing CSV files in `data/raw/`:
* `data/raw/Books.csv`
* `data/raw/Users.csv`
* `data/raw/Ratings.csv`

### Pre-computed Embeddings
Place the item embeddings file in the `data/embeddings/` directory:
* `data/embeddings/item2vec_books_qwen2_5_7b.pkl`

## 2. Run Flow
Execute these commands in order from the project root:

```bash
# Step 1: Clean raw data
python -m src.data_prep.clean_book_crossing

# Step 2: Generate interaction bags (N_SUP=40 and N_SUP=10, needed both)
python -m src.data_prep.bag_generator_book_crossing --n_sup 40
python -m src.data_prep.bag_generator_book_crossing --n_sup 10


# Step 3: Start training
python -m src.scripts.train_book_crossing