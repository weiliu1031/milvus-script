import logging
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np
from pymilvus.bulk_writer import bulk_import, list_import_jobs, RemoteBulkWriter, BulkFileType
import json, time
import os
import pandas as pd
import pyarrow.parquet as pq
from minio import Minio 
from minio.error import S3Error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def bulk_insert_from_parquet(
    parquet_directory: str,
    rewrite: bool = True,
    collection_name: str = "bulk_import_test",
    vector_dim: int = 1024,
    host: str = os.environ.get('MILVUS_HOST', '127.0.0.1')
):
    try:
        host = os.environ.get('MILVUS_HOST', '127.0.0.1')
        collection_name = "bulk_import_test"
        
        # 1. Connect to Milvus
        logging.info("Connecting to Milvus server...")
        connections.connect(host=host, port='19530')
        
        # Collection management
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)

        # 2. Create collection schema
        schema = CollectionSchema([
            FieldSchema("_id", DataType.VARCHAR, is_primary=True, max_length=65535),
            FieldSchema("url", DataType.VARCHAR, max_length=65535),
            FieldSchema("title", DataType.VARCHAR, max_length=65535),
            FieldSchema("text", DataType.VARCHAR, max_length=65535),
            FieldSchema("emb", DataType.FLOAT_VECTOR, dim=1024),
        ])
        
        collection = Collection(collection_name, schema, consistency_level="Strong")
        logging.info(f"Collection created: {collection.name}")

        remote_path = "/bulk_data/" + time.strftime("%Y-%m-%d-%H-%M-%S")
        file_list = []
        
        # Third-party constants
        minio_endpoint = os.environ.get('MINIO_ENDPOINT', 'localhost:9000')
        ACCESS_KEY="minioadmin"
        SECRET_KEY="minioadmin"
        BUCKET_NAME="a-bucket"
        
        if rewrite:
            # 3. Setup MinIO connection
            conn = RemoteBulkWriter.S3ConnectParam(
                endpoint=minio_endpoint,
                access_key=ACCESS_KEY,
                secret_key=SECRET_KEY,
                bucket_name=BUCKET_NAME,
                secure=False
            )

            # 4. Process Parquet files
            writer = RemoteBulkWriter(
                schema=schema,
                remote_path="/bulk_data/" + time.strftime("%Y-%m-%d-%H-%M-%S"),
                connect_param=conn,
                file_type=BulkFileType.PARQUET
            )
            
            for filename in os.listdir(parquet_directory):
                if not filename.endswith('.parquet'):
                    continue
                    
                file_path = os.path.join(parquet_directory, filename)
                df = pq.ParquetDataset(file_path).read().to_pandas()
                
                for _, row in df.iterrows():
                    writer.append_row({
                        "_id": str(row['_id']),
                        "url": str(row['url']),
                        "title": str(row['title']),
                        "text": str(row['text']),
                        "emb": row['emb'].astype(np.float32).tolist()
                    })
                    
                    if (_ + 1) % 10000 == 0:
                        writer.commit()
                        logging.info(f"Processed {_ + 1} rows from {filename}")
            
            writer.commit()
            logging.info(f"Total files generated: {writer.batch_files}")
            file_list.append(writer.batch_files)
        else:
             # Direct upload to MinIO
            minio_client = Minio(
                endpoint=minio_endpoint,
                access_key=ACCESS_KEY,
                secret_key=SECRET_KEY,
                secure=False
            )
            
            # Ensure bucket exists
            if not minio_client.bucket_exists("bulk-insert-bucket"):
                minio_client.make_bucket("bulk-insert-bucket")

            # Upload all parquet files
            for filename in os.listdir(parquet_directory):
                if not filename.endswith('.parquet'):
                    continue
                
                object_name = f"{remote_path}/{filename}"
                file_path = os.path.join(parquet_directory, filename)
                
                minio_client.fput_object(
                    BUCKET_NAME,
                    object_name,
                    file_path
                )
                logging.info(f"Uploaded {filename} to MinIO")
                file_list.append([f"{object_name}"])
            logging.info(f"Total files uploaded: {file_list}")

        # 5. Execute bulk import
        resp = bulk_import(
            f"http://{host}:19530",
            collection_name,
            files=file_list,
        )
        job_id = resp.json()['data']['jobId']
        logging.info(f"Bulk import job started: {job_id}")

        # 6. Monitor import progress
        progress = 0
        while True:
            resp = list_import_jobs(f"http://{host}:19530", collection_name)
            job_data = resp.json()['data']['records'][0]
            
            if job_data['state'] == 'Failed':
                raise Exception(f"Bulk import job failed: {job_data['reason']}")
            
            if job_data['progress'] > progress:
                progress = job_data['progress']
                logging.info(f"Import progress: {progress}%")
                
            if job_data['jobId'] == job_id and progress == 100:
                break
            time.sleep(5)

        # 7. Final verification
        logging.info("Creating index...")
        index_params = [
            ("emb", {"index_type": "HNSW", "metric_type": "L2", "params": {"M": 64, "efConstruction": 128}})
        ]
        for field, params in index_params:
            collection.create_index(field, params)
        collection.load()
        logging.info("Index created and collection loaded")
        collection.load()
        res = collection.query(
            expr='_id != ""', 
            output_fields=["count(*)"], count=True)
        logging.info(f"Final record count: {res[0]['count(*)']}")

        if rewrite:
            utility.drop_collection(collection_name)
            logging.info("Temporary collection cleaned up")

    except Exception as e:
        logging.error(f"Bulk load failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    bulk_insert_from_parquet(
        parquet_directory="/home/zilliz/data",
        rewrite=False
    )