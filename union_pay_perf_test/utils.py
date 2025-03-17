import time
import traceback
import numpy as np
from pymilvus import (
    Collection,
    utility,
    connections,
    CollectionSchema,
    DataType,
    FieldSchema,
)
from config import (
    milvus_uri,
    collection_name,
    pk_field,
    vector_field,
    M,
    efConstruction,
    num_insert_batch,
    vector_index_name,
    dim,
)
import polars as pl
from tqdm import tqdm
from loguru import logger
import concurrent
import multiprocessing as mp


def connect():
    connections.connect(uri=milvus_uri, timeout=30)


def get_collection():
    return Collection(collection_name)


def drop_collection_if_existed():
    if utility.has_collection(collection_name):
        logger.info(f"drop_old collection: {collection_name}")
        utility.drop_collection(collection_name)


def create_collection():
    logger.info(f"create_collection - {collection_name}")
    fields = [
        FieldSchema(pk_field, DataType.INT64, is_primary=True),
        FieldSchema(
            vector_field,
            DataType.FLOAT_VECTOR,
            dim=dim,
        ),
    ]

    Collection(
        name=collection_name,
        schema=CollectionSchema(fields),
    )


def release_collection():
    logger.info("release collection")
    col = Collection(collection_name)
    col.release()


def drop_index():
    logger.info("drop index")
    col = Collection(collection_name)
    col.drop_index(index_name=vector_index_name)


def create_flat_index(metric_type: str = "COSINE"):
    logger.info("create FLAT index")
    col = Collection(collection_name)
    index_params = {"metric_type": metric_type, "index_type": "FLAT"}
    col.create_index(vector_field, index_params, index_name=vector_index_name)


def create_index(metric_type: str = "COSINE"):
    logger.info("create index")
    col = Collection(collection_name)
    index_params = {
        "metric_type": metric_type,
        "index_type": "HNSW",
        "params": {
            "M": M,
            "efConstruction": efConstruction,
        },
    }
    col.create_index(vector_field, index_params, index_name=vector_index_name)


def optimize():
    logger.info("optimizing. it may take some time, please wait ...")
    col = Collection(collection_name)
    utility.wait_for_index_building_complete(collection_name)

    def wait_index():
        while True:
            progress = utility.index_building_progress(collection_name)
            if progress.get("pending_index_rows", -1) == 0:
                break
            time.sleep(30)

    wait_index()
    col.compact()
    col.wait_for_compaction_completed()
    wait_index()


def load_index():
    logger.info("load index")
    col = Collection(collection_name)
    col.load()


def insert_data(embs: list[list[float]], cur_idx: int) -> int:
    col = Collection(collection_name)
    num_rows = len(embs)
    for i in tqdm(range(0, num_rows, num_insert_batch)):
        vector_data = embs[i : i + num_insert_batch]
        new_idx = cur_idx + len(vector_data)
        pk_data = list(range(cur_idx, new_idx))
        cur_idx = new_idx
        col.insert([pk_data, vector_data])
    return cur_idx


def compute_recall(ids: list[int], gt: list[int]):
    return sum([id in gt for id in ids]) / len(ids)


def search(
    col: Collection,
    query: pl.DataFrame,
    ef: int,
    k: int,
    expr: str = "",
) -> list[int]:
    res = col.search(
        data=[query],
        anns_field=vector_field,
        param=dict(params=dict(ef=ef)),
        limit=k,
        expr=expr,
    )
    return [r.id for r in res[0]]


def search_by_dur(
    queries: list[list[float]],
    duration: int,
    ef: int,
    k: int,
    expr: str,
    q: mp.Queue,
    cond: mp.Condition,  # type: ignore
) -> float:
    """return the number of finished search requests."""
    connect()
    col = get_collection()

    count = 0
    query_len = len(queries)
    idx = np.random.randint(query_len)

    # sync all process
    q.put(1)
    with cond:
        cond.wait()
    start_time = time.perf_counter()
    while time.perf_counter() < start_time + duration:
        search(col, queries[idx], ef=ef, k=k, expr=expr)
        count += 1
        if idx >= query_len - 1:
            idx = 0
        else:
            idx += 1
    return count


def conc_search(
    conc: int,
    queries: list[list[float]],
    conc_duration: int,
    ef: int,
    k: int,
    expr: str,
) -> float:
    """return qps"""
    logger.info(f"conc_test [start] - conc: {conc}")
    try:
        with mp.Manager() as m:
            q, cond = m.Queue(), m.Condition()
            with concurrent.futures.ProcessPoolExecutor(
                mp_context=mp.get_context("spawn"), max_workers=conc
            ) as executor:
                future_iter = [
                    executor.submit(
                        search_by_dur, queries, conc_duration, ef, k, expr, q, cond
                    )
                    for _ in range(conc)
                ]

                # sync all processes
                while q.qsize() < conc:
                    time.sleep(10)

                with cond:
                    cond.notify_all()
                    logger.info(
                        f"syncing all process and start concurrency search, concurrency={conc}"
                    )

                process_counts = [r.result() for r in future_iter]
                all_count = sum(process_counts)
                qps = round(all_count / conc_duration, 4)
                logger.info(
                    f"test done, conc: {conc}, all_count={all_count}, qps={qps}"
                )
                return qps
    except Exception as e:
        logger.warning(f"Fail to search all concurrencies: {conc}, reason={e}")
        traceback.print_exc()
