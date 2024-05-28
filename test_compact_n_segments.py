from pymilvus import connections, Collection
import os
from  pathlib import Path

# local
from load_data import prepare_collection
from generate_segment import generate_segments, SegmentDistribution, Unit


def generate_n_segments(name: str, n: int = 20):
    connections.connect()
    prepare_collection(name, 768, False)
    c = Collection(name)
    if not c.has_index():
        c.create_index("embeddings", {"index_type": "FLAT", "params": {"metric_type": "L2"}})

    ten_segs = [123 for i in range(n)]
    dist = SegmentDistribution(
        collection_name=name,
        size_dist=ten_segs,
        unit=Unit.MB,
    )
    return generate_segments(dist)


def delete_n_percent(name: str, all_pks: list[list] = None, n: int = 20):
    import numpy as np
    connections.connect()
    c = Collection(name)
    c.load()

    if not isinstance(all_pks, list):
        raise TypeError(f"pks should be a list, but got {type(all_pks)}")

    for i, pks in enumerate(all_pks):
        sample_pks = np.random.choice(pks, size=int(n*0.01*len(pks)), replace=False)
        expr = f"pk in {sample_pks.tolist()}"
        c.delete(expr)
        print(f"sampled pk counts: {len(sample_pks)} and delete done")
        c.flush()


def delete_n_percent_to_files(name: str, all_pks: list[list] = None, n: int = 20):
    import numpy as np
    connections.connect()
    c = Collection(name)
    c.load()

    if not isinstance(all_pks, list):
        raise TypeError(f"pks should be a list, but got {type(all_pks)}")

    for i, pks in enumerate(all_pks):
        sample_pks = np.random.choice(pks, size=int(n*0.01*len(pks)), replace=False)
        with Path(f"pks_{i}.txt").open("w") as f:
            for pk in sample_pks:
                f.write(f"{pk}\n")


def delete_all(name):
    connections.connect()
    c = Collection(name)
    c.load()
    ret = c.delete(f"pk > 1")
    c.flush()
    print(f"delete counts: {ret.delete_count}")


def delete_by_files(name: str):
    connections.connect()
    c = Collection(name)
    c.load()
    del_count = 0

    for i in range(20):
        with Path(f"pks_{i}.txt").open("r") as f:
            pks = [int(line.strip()) for line in f.readlines()]
        expr = f"pk in {pks}"
        ret = c.delete(expr)
        print(f"sampled pk counts: {len(pks)} and delete done")
        del_count += ret.delete_count
        c.flush()

    print(f"delete counts: {del_count}")


def test_case_generate_20_segments_del_20perc():
    name = "test_l0_compact_20_seg"
    pks = generate_n_segments(name, 20)
    print(f"generated 20 segments, total row counts: {len(pks)}")
    delete_n_percent(name, pks, 20)


def test_case_generate_20_segments_del_20perc_to_files():
    name = "test_l0_compact_20_seg"
    pks = generate_n_segments(name, 20)
    delete_n_percent_to_files(name, pks, 20)


def test_case_generate_20_segments_del_all():
    name = "test_l0_compact_20_seg_clean_all"
    generate_n_segments(name, 20)
    delete_all(name)

def test_case_generate_20_segments_no_del():
    name = "test_l0_compact_20_seg_clean_all"
    generate_n_segments(name, 20)
    delete_all(name)


if __name__ == "__main__":
    #  test_case_generate_20_segments_del_20perc()
    test_case_generate_20_segments_del_all()
