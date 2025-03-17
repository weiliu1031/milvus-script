import logging
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np
from pymilvus.bulk_writer import bulk_import, list_import_jobs, RemoteBulkWriter, BulkFileType
import json, time

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
    
    collection_name = "bulk_import_test"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    # 2. Create dense vector collection
    logging.info("Creating collection...")
    metric_type = "L2"  # or "IP"
    
    
    schema = CollectionSchema([
        FieldSchema("id", DataType.INT64, is_primary=True),
        FieldSchema("vector", DataType.FLOAT_VECTOR, dim=256),
    ])
    collection = Collection(
        collection_name,
        schema,
        consistency_level="Strong"
    )
    logging.info(f"Collection created: {collection.name}")
    
    # Third-party constants
    ACCESS_KEY="minioadmin"
    SECRET_KEY="minioadmin"
    BUCKET_NAME="a-bucket"
    
    minio_endpoint = "localhost:9000"  # the default MinIO service started along with Milvus
    remote_path = "/bulk_data/" + time.strftime("%Y-%m-%d-%H-%M-%S")
    url = f"http://127.0.0.1:19530"

    # Connections parameters to access the remote bucket
    conn = RemoteBulkWriter.S3ConnectParam(
        endpoint=minio_endpoint,
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        bucket_name=BUCKET_NAME,
        secure=False
    )

    writer = RemoteBulkWriter(
        schema=schema,
        remote_path=remote_path,
        connect_param=conn,
        file_type=BulkFileType.PARQUET
    )
    print('bulk writer created.')
    
    for i in range(10000):
        writer.append_row({
            "id": i,
            "vector": np.random.randn(256).astype(np.float32).tolist(),
        })
        
        if i+1 % 1000 == 0:
            writer.commit()
            print(f'bulk writer flushed {i} rows.')
            
    writer.commit()
    print('bulk writer flushed all rows.')
    print(writer.batch_files)
    
    resp = bulk_import(
        url,
        collection_name,
        files=writer.batch_files,
    )

    job_id = resp.json()['data']['jobId']
    print(f'bulk import job id: {job_id}')
    
    progress = 0
    while True:
        resp = list_import_jobs(
            url=url,
            collection_name=collection_name,
        )
        new_progress = resp.json()['data']['records'][0]['progress']
        if new_progress > progress:
            progress = new_progress
            print(json.dumps(resp.json(), indent=4))
        
        if (resp.json()['data']['records'][0]['jobId'] == job_id) and (new_progress== 100):
            break

    # 3. Create index and load
    logging.info("Creating index...")
    index_params = [
        ("vector", {"index_type": "HNSW", "metric_type": "L2", "params": {"M": 64, "efConstruction": 128}})
    ]
    
    for field, params in index_params:
        collection.create_index(field, params)
    collection.load()
    logging.info("Index created and collection loaded")
    
    # verify count(*) result
    res = collection.query(
        expr="id >= 0",  # Match all records
        output_fields=["count(*)"],
        count=True
    )
    actual_count = res[0]["count(*)"]
    logging.info(f"Count(*) result: {actual_count} (Expected: 10000)") 

    # Cleanup
    logging.info("Cleaning up...")
    utility.drop_collection(collection_name)
    logging.info("Collection dropped successfully")

except Exception as e:
    logging.error(f"Error occurred: {str(e)}", exc_info=True)
    raise