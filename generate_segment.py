"""Give SegmentDistribution(segment size, collection, and segments count), generate segments.

Notes:
    Before running the scripts, The server MUST have:
        1. started
        2. created collection with zero num_rows

    If the segment size is larger than 100MB, its recommended to:
        1. set the server config: dataCoord.segment.sealProportion = 1
"""

from pydantic import BaseModel
from pymilvus import Collection, connections, utility, DataType
import pymilvus
import numpy as np
import uuid

class SegmentDistribution(BaseModel):
    collection_name: str
    size_dist: tuple # in bytes (16*1024*1024, 32 * 1024 * 1024)


def generate_segments(dist: SegmentDistribution):
    connections.connect()

    if not utility.has_collection(dist.collection_name):
        msg = f"Collection {dist.collection_name} does not exist"
        raise ValueError(msg)

    c = Collection(dist.collection_name)
    #  if c.num_entities != 0:
    #      msg = f"Collection {dist.collection_name} has {c.num_entities} entities, please provide an empty collection"
    #      raise ValueError(msg)

    pks = []
    for size in dist.size_dist:
        pks.append(generate_one_segment(c, size))

    return pks


def generate_one_segment(c: Collection, size: int) -> list:
    count = estimate_count_by_size(size, c.schema)
    print(f"generate {count} entities for size: {size}")
    data = gen_data_by_schema(c.schema, count)

    ret = c.insert(data)
    c.flush()

    return ret.primary_keys


def estimate_count_by_size(size: int, schema: pymilvus.CollectionSchema) -> int:
    size_per_row = 0
    for fs in schema.fields:
        if fs.dtype == DataType.INT64:
            size_per_row += 8
        elif fs.dtype == DataType.VARCHAR:
            size_per_row += fs.max_length
        elif fs.dtype == DataType.FLOAT_VECTOR:
            size_per_row += fs.dim * 4
        elif fs.dtype == DataType.DOUBLE:
            size_per_row += 8
        else:
            msg = f"Unsupported data type: {fs.dtype.name}, please impl in generate_segment.py yourself"
            raise ValueError(msg)

    return int(size / size_per_row)



def gen_data_by_schema(schema: pymilvus.CollectionSchema, count: int) -> list:
    rng = np.random.default_rng()
    data = []
    for fs in schema.fields:
        if fs.dtype == DataType.INT64:
            if fs.is_primary and not fs.auto_id:
                data.append([uuid.uuid1().int >> 65  for _ in range(count)])
            else:
                data.append(list(range(count)))

        elif fs.dtype == DataType.VARCHAR:
            if fs.is_primary and not fs.auto_id:
                data.append([str(uuid.uuid4()) for _ in range(count)])
            else:
                data.append([str(i) for i in range(count)])

        elif fs.dtype == DataType.FLOAT_VECTOR:
            data.append(rng.random((count, fs.dim)))

        elif fs.dtype == DataType.DOUBLE:
            data.append(rng.random(count))

        else:
            msg = f"Unsupported data type: {fs.dtype.name}, please impl in generate_segment.py yourself"
            raise ValueError(msg)
    return data


if __name__ == "__main__":
    dist = [32*1024*1024, 32*1024*1024]
    pks = generate_segments(SegmentDistribution(collection_name="test1", size_dist=(16*1024*1024, 32 * 1024 * 1024)))
