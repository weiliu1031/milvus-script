import json
from loguru import logger
import numpy as np
import polars as pl
from tqdm import tqdm
from config import (
    get_groundtruth,
    get_query_vectors,
    train_file_paths,
    train_vector_col_name,
    exprs,
    ef_list,
    k,
    conc_list,
    conc_duration,
    results_file,
)
from utils import (
    compute_recall,
    conc_search,
    connect,
    create_collection,
    create_index,
    drop_collection_if_existed,
    get_collection,
    insert_data,
    load_index,
    optimize,
    search,
)
import time


def insert_test() -> float:
    connect()
    drop_collection_if_existed()
    create_collection()
    create_index()
    load_index()

    start_time = time.perf_counter()
    cur_idx = 1
    for i, file in enumerate(train_file_paths):
        logger.info(
            f"[{i+1}/{len(train_file_paths)}] read and insert embeddings from {file}"
        )
        df = pl.read_parquet(file)
        embs: list[list[float]] = df[train_vector_col_name].to_list()
        cur_idx = insert_data(embs, cur_idx)
    cost = round(time.perf_counter() - start_time, 4)
    logger.info(f"insert finished. cost {cost}s")
    return cost


def optimize_test() -> float:
    start_time = time.perf_counter()
    optimize()
    cost = round(time.perf_counter() - start_time, 4)
    logger.info(f"optimized finished. cost {cost}s")
    return cost


def serial_search_test(
    queries: list[list[float]], gts: list[list[int]], expr: str, ef: int, k: int
) -> tuple[float, float, float]:
    """
    return: recall, latency_p99, latency_avg
    """
    connect()
    col = get_collection()

    logger.info("start serial search test")
    latencies = []
    recalls = []
    for i, q in tqdm(enumerate(queries), total=len(queries)):
        start_time = time.perf_counter()
        ids = search(col, q, ef, k, expr)
        latencies.append((time.perf_counter() - start_time) * 1000)
        recall = compute_recall(ids[:k], gts[i][:k])
        recalls.append(recall)

    recall = round(np.mean(recalls), 4)
    latency_p99 = round(np.percentile(latencies, 99), 4)
    latency_avg = round(np.mean(latencies), 4)

    logger.info(
        f"finish serial search test. recall={recall}, latency_p99={latency_p99}ms, latency_avg={latency_avg}ms"
    )
    return recall, latency_p99, latency_avg


def conc_search_test(queries: list[list[float]], expr: str, ef: int, k: int):
    max_conc_qps = 0
    for conc in conc_list:
        conc_qps = conc_search(
            conc=conc,
            queries=queries,
            conc_duration=conc_duration,
            ef=ef,
            k=k,
            expr=expr,
        )
        max_conc_qps = max(max_conc_qps, conc_qps)
    return max_conc_qps


def search_test():
    logger.info("read query vectors")
    queries = get_query_vectors()

    search_results = []
    for expr in exprs:
        logger.info(f"read groundtruth for expr: {expr}")
        gts = get_groundtruth(expr)

        for ef in ef_list:
            logger.info(f"search test with expr='{expr}', ef={ef}")
            max_conc_qps = conc_search_test(queries=queries, expr=expr, ef=ef, k=k)
            recall, latency_p99, latency_avg = serial_search_test(
                queries=queries, gts=gts, expr=expr, ef=ef, k=k
            )
            search_results.append(
                dict(
                    expr=expr,
                    ef=ef,
                    recall=recall,
                    latency_p99=latency_p99,
                    latency_avg=latency_avg,
                    qps=max_conc_qps,
                )
            )
    return search_results


def save_results(insert_time: float, optimize_time: float, search_results: list[dict]):
    logger.info("====> all test results:")
    logger.info(f"insert cost {insert_time}")
    logger.info(f"optimize cost {optimize_time}")
    for search_res in search_results:
        logger.info(search_res)
    with open(results_file, "w") as f:
        json.dump(
            dict(
                insert_time=insert_time,
                optimize_time=optimize_time,
                search_res=search_res,
            ),
            f,
        )


def main():
    # insert
    insert_time = insert_test()

    # compact
    optimize_time = optimize_test()

    # search (including filter)
    search_results = search_test()

    # output results
    save_results(insert_time, optimize_time, search_results)


if __name__ == "__main__":
    main()
