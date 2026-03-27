from preprocessing.scaler import MinMaxScaler, StandardScaler

datasets_config = [
    {"dataset_name": "musk1",             "best_scaler": MinMaxScaler,  "best_distance": "hausdorff",      "best_eps": 2.1673,   "best_min_pts": 2},
    {"dataset_name": "musk2",             "best_scaler": MinMaxScaler,  "best_distance": "cauchy_schwarz", "best_eps": 0.02026,  "best_min_pts": 3},
    {"dataset_name": "ImageElephant",     "best_scaler": MinMaxScaler,  "best_distance": "cauchy_schwarz", "best_eps": 0.11840,  "best_min_pts": 2},
    {"dataset_name": "BirdsChestnut",     "best_scaler": StandardScaler,"best_distance": "cauchy_schwarz", "best_eps": 0.2988,   "best_min_pts": 10},
    {"dataset_name": "BirdsHammonds",     "best_scaler": MinMaxScaler,  "best_distance": "cauchy_schwarz", "best_eps": 0.00565,  "best_min_pts": 2},
    {"dataset_name": "Harddrive1",        "best_scaler": MinMaxScaler,  "best_distance": "cauchy_schwarz", "best_eps": 0.003467, "best_min_pts": 3},
    {"dataset_name": "mutagenesis_atoms", "best_scaler": StandardScaler,"best_distance": "hausdorff",      "best_eps": 0.4748,   "best_min_pts": 3},
    {"dataset_name": "mutagenesis_chains","best_scaler": MinMaxScaler,  "best_distance": "cauchy_schwarz", "best_eps": 0.006638, "best_min_pts": 3},
    {"dataset_name": "Newsgroups1",       "best_scaler": StandardScaler,"best_distance": "hausdorff",      "best_eps": 50.434,   "best_min_pts": 2},
    {"dataset_name": "simple_dummy",      "best_scaler": StandardScaler,"best_distance": "hausdorff",      "best_eps": 0.1303,   "best_min_pts": 2},
    {"dataset_name": "Thioredoxin",       "best_scaler": MinMaxScaler,  "best_distance": "cauchy_schwarz", "best_eps": 0.001185, "best_min_pts": 2},
]