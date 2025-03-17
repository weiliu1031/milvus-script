import logging
from pymilvus import MilvusClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

try:
    # 1. Initialize Milvus Client
    logging.info("Initializing Milvus client...")
    client = MilvusClient(
        uri="http://127.0.0.1:19530",
        # token='username:password'  # Add if authentication enabled
    )
    logging.info("Successfully connected to Milvus")

    # 2. Cleanup existing collection
    if client.has_collection("sparse_test"):
        client.drop_collection("sparse_test")
    
    # 3. Create collection with sparse vector support
    logging.info("Creating collection...")
    client.create_collection(
        collection_name="sparse_test",
        dimension=1,  # Dummy dimension for sparse vectors
        primary_field_name="id",
        vector_field_name="vector",
        id_type="int64",
        metric_type="IP",
        vector_type="SPARSE_FLOAT_VECTOR",
        auto_id=False
    )
    logging.info("Collection created: sparse_test")

    # 4. Create index
    logging.info("Creating sparse index...")
    client.create_index(
        collection_name="sparse_test",
        field_name="vector",
        index_params={
            "index_type": "SPARSE_INVERTED_INDEX",
            "params": {"nlist": 16}
        }
    )
    logging.info("Sparse index created")

    # 5. Insert vectors
    logging.info("Generating and inserting vectors...")
    sparse_vectors = [
        {"id": 0, "vector": {1: 0.5, 100: 0.3, 500: 0.8}},
        {"id": 1, "vector": {10: 0.1, 200: 0.7, 1000: 0.9}},
        {"id": 2, "vector": {20: 0.2, 300: 0.6, 1500: 0.7}},
        {"id": 3, "vector": {30: 0.3, 400: 0.5, 2000: 0.8}},
    ]
    
    client.insert("sparse_test", sparse_vectors)
    logging.info(f"Inserted {len(sparse_vectors)} vectors")

    # 6. Query validation
    logging.info("Running query...")
    query_result = client.query(
        collection_name="sparse_test",
        filter="id == 0",
        output_fields=["vector"]
    )
    
    logging.info("Query results:")
    for idx, result in enumerate(query_result):
        logging.info(f"Result {idx}: ID={result['id']}, Vector={result['vector']}")

    # 7. Search validation
    logging.info("Running search...")
    search_result = client.search(
        collection_name="sparse_test",
        data=[sparse_vectors[0]["vector"]],
        anns_field="vector",
        search_params={"params": {"drop_ratio_search": 0.2}},
        limit=3,
        output_fields=["vector"]
    )
    
    logging.info("Search results:")
    for idx, hit in enumerate(search_result[0]):
        logging.info(f"Hit {idx}: ID={hit['id']}, Distance={hit['distance']:.4f}")

    # 8. Cleanup
    logging.info("Cleaning up...")
    client.drop_collection("sparse_test")
    logging.info("Collection dropped successfully")

except Exception as e:
    logging.error(f"Error occurred: {str(e)}", exc_info=True)
    raise