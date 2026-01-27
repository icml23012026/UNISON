# UNISON: Unified Framework for Learning from Scored Bags

Code for the paper "UNISON: A Unified Framework for Learning from Scored Bags".

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Data

The toy dataset includes:
```
data/
├── raw/
│   ├── Books.csv       (20 books)
│   ├── Users.csv       (15 users)
│   └── Ratings.csv     (165 ratings)
└── embeddings/
    └── item2vec_books_qwen.pkl
```

**Note:** This is a minimal toy dataset.

## Usage
Run these commands **in order** from the project root:
```bash
# Step 1: Generate interaction bags
python -m src.data_prep.episode_generator_book_crossing

# Step 2: Start training
python -m src.scripts.train_book_crossing