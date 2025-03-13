import logging
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
try:
    # 1. Connect to Milvus
    logging.info("Connecting to Milvus server...")
    connections.connect(host='127.0.0.1', port='19530')
    logging.info("Successfully connected to Milvus")

    utility.drop_collection("json_test")
    # 2. Create dense vector collection
    logging.info("Creating collection...")
    dim = 768
    metric_type = "L2"  # or "IP"
    collection = Collection(
        "json_test",
        CollectionSchema([
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema("metadata", DataType.JSON)
        ]),
        consistency_level="Strong"
    )
    logging.info(f"Collection created: {collection.name}")

    # 3. Create index and load
    logging.info("Creating index...")
    collection.create_index(
        "vector",
        {"index_type": "IVF_FLAT", "metric_type": metric_type, "params": {"nlist": 16}}
    )
    collection.load()
    logging.info("Index created and collection loaded")

    # 4. Insert dense vectors
    logging.info("Generating and inserting vectors...")
    vectors = np.random.randn(100, dim).astype(np.float32)
    data = [
        list(range(100)),
        vectors,
        [{
            "title": f"doc_{i}",
            "tags": ["tag1", "tag2"] if i%2 else ["tag3"],
            "stats": {"views": i*10, "rating": round(np.random.uniform(1, 5), 1)}
        } for i in range(100)]
    ]
    collection.insert(data)

    logging.info(f"Inserted {len(vectors)} vectors")

    # 5. Verify JSON query
    logging.info("Querying JSON data...")
    res = collection.query(
        expr="metadata['title'] == 'doc_0' and metadata['stats']['rating'] > 0",
        output_fields=["metadata", "id"]
    )
    print("\nJSON query results:")
    for r in res:
        print(f"ID:{r['id']} | Metadata:{r['metadata']}")

    # 6. Verify JSON in search results
    search_result = collection.search(
        data[1][:1],  # Use first vector
        "vector",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=3,
        output_fields=["metadata"]
    )
    print("\nJSON in search results:")
    for hit in search_result[0]:
        print(f"ID:{hit.id} | Metadata:{hit.entity.fields['metadata']}")

    # Cleanup
    logging.info("Cleaning up...")
    utility.drop_collection("float_vector_test")
    logging.info("Collection dropped successfully")

except Exception as e:
    logging.error(f"Error occurred: {str(e)}", exc_info=True)
    raise