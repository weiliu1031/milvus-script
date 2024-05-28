"""Give SegmentDistribution(segment size, collection, and segments count), generate segments.

Notes:
    Before running the scripts, The server MUST have:
        1. started
        2. created collection with zero num_rows

    If the segment size is larger than 100MB, its recommended to:
        1. set the server config: dataCoord.segment.sealProportion = 1
"""

from typing import Union
from pydantic import BaseModel
from pymilvus import Collection, connections, utility, DataType, Partition
import pymilvus
import numpy as np
import uuid
from enum import Enum


class Unit(str, Enum):
    B = "Bytes"
    KB = "Kilobytes"
    MB = "Megabytes"
    GB = "Gigabytes"


class SegmentDistribution(BaseModel):
    collection_name: str
    partition_name: str = "_default"
    size_dist: tuple # in bytes (16*1024*1024, 32 * 1024 * 1024)
    unit: Unit = Unit.B

    def as_bytes(self, size: int) -> int:
        factor = 1
        if self.unit == Unit.B:
            factor = 1
        elif self.unit == Unit.KB:
            factor = 1024
        elif self.unit == Unit.MB:
            factor = 1024 * 1024
        elif self.unit == Unit.GB:
            factor = 1024 * 1024 * 1024
        return size * factor


def generate_segments(dist: SegmentDistribution):

    if not utility.has_collection(dist.collection_name):
        msg = f"Collection {dist.collection_name} does not exist"
        raise ValueError(msg)

    c = Collection(dist.collection_name)
    p = c.partition(dist.partition_name)

    pks = []
    for size in dist.size_dist:
        pks.append(generate_one_segment(p, c.schema, dist.as_bytes(size)))

    return pks


def generate_one_segment(c: Union[Collection, Partition], schema: pymilvus.CollectionSchema, size: int) -> list:
    max_size = 5 * 1024 * 1024
    total_count = 0
    pks = []

    if size > max_size:
        batch =  size // (5 * 1024 * 1024)
        tail = size - batch * max_size

        for i in range(batch):
            count = estimate_count_by_size(max_size, schema)
            data = gen_data_by_schema(schema, count)
            rt = c.insert(data)
            print(f"inserted {max_size * (i+1)}/{size}Bytes entities in batch 5MB, nun rows: {count}")
            pks.extend(rt.primary_keys)
            total_count += count
    else:
        tail = size

    if tail > 0:
        count = estimate_count_by_size(tail, schema)
        data = gen_data_by_schema(schema, count)
        c.insert(data)
        print(f"inserted entities size: {tail}Bytes, {tail/1024/1024}MB, nun rows: {count}")
        pks.extend(rt.primary_keys)
        total_count += count

    c.flush()
    print(f"One segment num rows: {c.num_entities}, size: {size}Bytes, {size/1024/1024}MB")
    return pks


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
    connections.connect()
    pks = generate_segments(SegmentDistribution(collection_name="test1", size_dist=(16*1024*1024, 32 * 1024 * 1024)))
