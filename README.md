# Recommender_System_AutoEncoder


### Datasets
- Movielens-100K `https://grouplens.org/datasets/movielens/`
- Movielens-1Million `https://grouplens.org/datasets/movielens/`
- Jester `http://eigentaste.berkeley.edu/dataset/`
- FlickScore `https://arxiv.org/abs/1801.02203`
- Hetrec2011-movielens-2k `https://grouplens.org/datasets/hetrec-2011/`
- Hetrec2011-delicious-2k `https://grouplens.org/datasets/hetrec-2011/`

### Requirements
- `numpy`
- `pandas`
- `tensorflow`
- `matplotlib`
- `scipy`

### How to Run
1. Get the proper dataset in the directory first with name `ratings.csv`
2. run `CUDA_VISIBLE_DEVICES=0 python <filename>` to run `<filename>` on GPU 0

### Change for new dataset

1. Take the existing code from any of the folder for instance `imdb/warm.py`
2. Change parameters like `NUM_USERS`, `NUM_ITEMS`, `ENCODED_SIZE`, `NUM_RATINGS` etc according to the new dataset
