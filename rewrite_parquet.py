import logging
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np
from pymilvus.bulk_writer import bulk_import, list_import_jobs, RemoteBulkWriter, BulkFileType, LocalBulkWriter
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

def process_parquet_files(parquet_dir: str = "/home/zilliz/data", output_dir: str = "/home/zilliz/rewrite_data"):
    """Process all parquet files in directory and generate bulk load files"""
    try:
        # Setup writer with collection schema
        schema = CollectionSchema([
            FieldSchema("_id", DataType.VARCHAR, is_primary=True, max_length=65535),
            FieldSchema("url", DataType.VARCHAR, max_length=65535),
            FieldSchema("title", DataType.VARCHAR, max_length=65535),
            FieldSchema("text", DataType.VARCHAR, max_length=65535),
            FieldSchema("emb", DataType.FLOAT_VECTOR, dim=1024),
        ])
        
        with LocalBulkWriter(
            schema=schema,
            local_path=output_dir,
            segment_size=512*1024*1024,
            file_type=BulkFileType.PARQUET
        ) as writer:
            
            # Process files
            for filename in os.listdir(parquet_dir):
                if not filename.endswith('.parquet'):
                    continue
                
                file_path = os.path.join(parquet_dir, filename)
                df = pq.ParquetDataset(file_path).read().to_pandas()
                
                # Batch processing
                for idx, row in df.iterrows():
                    writer.append_row({
                        "_id": str(row['_id']),
                        "url": str(row['url']),
                        "title": str(row['title']),
                        "text": str(row['text']),
                        "emb": row['emb'].astype(np.float32).tolist()
                    })
                    
                    if (idx + 1) % 10000 == 0:
                        writer.commit()
                        logging.info(f"Processed {idx + 1} rows from {filename}")
                
                writer.commit()
                logging.info(f"Total files generated: {writer.batch_files}")

    except Exception as e:
        logging.error(f"Processing failed: {str(e)}", exc_info=True)
        raise

def main():
    logging.info("Starting parquet processing...")
    process_parquet_files()
    logging.info("Processing completed successfully")

if __name__ == "__main__":
    main()