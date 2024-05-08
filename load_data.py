""" python load_data.py -c test1 """

import concurrent
import threading
import argparse
import time

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    connections,
    DataType,
    FieldSchema,
    utility,
)


def prepare_collection(name: str, dim: int, recreate_if_exist: bool=False):
    connections.connect()

    def create():
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="random", dtype=DataType.DOUBLE),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]

        schema = CollectionSchema(fields)
        Collection(name, schema)

    if not utility.has_collection(name):
        create()

    elif recreate_if_exist is True:
        utility.drop_collection(name)
        create()

class MilvusMultiThreadingInsert:
    def __init__(self, collection_name: str, total_count: int, num_per_batch: int, dim: int):

        batch_count = int(total_count / num_per_batch)

        self.thread_local = threading.local()
        self.collection_name = collection_name
        self.dim = dim
        self.total_count = total_count
        self.num_per_batch = num_per_batch
        self.batchs = [i for i in range(batch_count)]

    def connect(self, uri: str):
        connections.connect(uri=uri)

    def get_thread_local_collection(self):
        if not hasattr(self.thread_local, "collection"):
            self.thread_local.collection = Collection(self.collection_name)
        return self.thread_local.collection

    def insert_work(self, number: int):
        print(f"No.{number:2}: Start inserting entities")
        rng = np.random.default_rng(seed=number)
        entities = [
            [i for i in range(self.num_per_batch*number, self.num_per_batch*(number+1))],
            rng.random(self.num_per_batch).tolist(),
            rng.random((self.num_per_batch, self.dim)),
        ]

        insert_result = self.get_thread_local_collection().insert(entities)
        assert len(insert_result.primary_keys) == self.num_per_batch
        print(f"No.{number:2}: Finish inserting entities")

    def _insert_all_batches(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            executor.map(self.insert_work, self.batchs)

    def run(self):
        start_time = time.time()
        self._insert_all_batches()
        duration = time.time() - start_time
        print(f'Inserted {len(self.batchs)} batches of entities in {duration} seconds')
        print(f"Expected num_entities: {self.total_count}. \
                Acutal num_entites: {self.get_thread_local_collection().num_entities}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collection", type=str, required=True, help="collection name")
    parser.add_argument("-d", "--dim", type=int, default=128, help="dimension of the vectors")
    parser.add_argument("-n", "--new", action="store_true", help="Whether to create a new collection or use the existing one")

    flags = parser.parse_args()
    uri = "http://localhost:19530"

    prepare_collection(flags.collection, flags.dim, recreate_if_exist=flags.new)

    mp_insert = MilvusMultiThreadingInsert(flags.collection, 100_000, 5000, flags.dim)
    mp_insert.connect(uri)
    mp_insert.run()
