# milvus
import os
import numpy as np
import polars as pl
from pathlib import Path

milvus_uri = "http://localhost:19530"
collection_name = "union_pay_test"
pk_field = "pk"
vector_field = "vector"
num_insert_batch = 500
vector_index_name = "vector_idx"
dim = 1024

# index config
M = 16
efConstruction = 200
ef_list = [10, 20, 40, 60, 80, 100, 120, 160, 200, 300, 400, 500]

# conc test config
k = 10
conc_list = [1, 5, 10, 15, 20, 40, 60, 80]
conc_duration = 30

# dataset config
train_dir = ""
train_files = os.listdir(train_dir)
train_file_paths = [Path(train_dir, file) for file in train_files]
train_files.sort()
train_vector_col_name = "emb"
query_vectors_file = ""
query_vector_col_name = "emb"


def get_query_vectors() -> list[list[float]]:
    return np.random.rand(1000, 1024).tolist()
    df = pl.read_parquet(query_vectors_file)
    return df[query_vector_col_name].to_list()


exprs = ["", f"{pk_field} > 2000", f"{pk_field} > 10000", f"{pk_field} > 18000"]
groudtruth_dir = ""
groudtruth_col_name = "neighbors_id"
groudtruth_files = []


def get_groundtruth(expr: str) -> list[list[int]]:
    return np.random.randint(1, 100, size=(1000, 10))
    gt_file = groudtruth_files[exprs.index(expr)]
    df = pl.read_parquet(Path(groudtruth_dir, gt_file))
    return df[groudtruth_col_name].to_list()


results_file = "results.json"
