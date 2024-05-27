from pymilvus import connections, Collection

# local
from load_data import prepare_collection
from generate_segment import generate_segments, SegmentDistribution


if __name__ == "__main__":

    connections.connect()
    name = "test_drop_partition"
    prepare_collection(name, 8, False)

    test_partition_name = "test_ppp"
    c = Collection(name)
    if not c.has_partition(test_partition_name):
        test_p = c.create_partition(test_partition_name)
    else:
        test_p = c.partition(test_partition_name)

    dist = [512, 512, 512, 512]
    pks = generate_segments(SegmentDistribution(
        collection_name=name,
        partition_name=test_partition_name,
        size_dist=dist))

    test_p.drop()
