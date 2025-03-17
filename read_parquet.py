import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc  # Add this import
import pyarrow as pa

def read_parquet(file_path):
    parquet_file = pq.ParquetFile(file_path)
    table = parquet_file.read()
    
    print("=============================================================")
    print(table)
    print("=============================================================")
    print(parquet_file.schema)
    print(parquet_file.num_row_groups)
    print("=============================================================")
    # New analysis for list columns
    for i, column in enumerate(table):
        if pa.types.is_list(column.type):
            list_lengths = pc.list_value_length(column)
            print(f"\nList column found: {column._name}")
            print(f"Average length: {pc.mean(list_lengths).as_py():.2f}")
            print(f"Max length: {pc.max(list_lengths).as_py()}")
            print(f"Min length: {pc.min(list_lengths).as_py()}")
            print(f"Standard deviation: {pc.stddev(list_lengths).as_py():.2f}")
    print("=============================================================")

# Example usage
if __name__ == "__main__":
    parquet_path = "~/0000.parquet"  # Replace with your file path
    read_parquet(parquet_path)
