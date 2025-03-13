from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor

def connect_to_milvus():
    """连接到Milvus服务器"""
    connections.connect(
        alias="default",
        host="localhost",
        port="19530"
    )
    print("Successfully connected to Milvus")

def create_collection():
    """创建集合和字段"""
    collection_name = "test_collection"
    dim = 128  # 向量维度

    # 检查集合是否存在
    if utility.has_collection(collection_name):
        print(f"Collection {collection_name} already exists, dropping it...")
        utility.drop_collection(collection_name)
        print(f"Successfully dropped collection: {collection_name}")

    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    
    # 创建集合模式
    schema = CollectionSchema(fields=fields, description="测试集合")
    
    # 创建集合
    collection = Collection(name=collection_name, schema=schema)
    print(f"Successfully created collection: {collection_name}")
    return collection

def insert_data(collection, num_entities=100000):
    """插入数据"""
    # 生成随机数据
    ids = np.arange(num_entities)
    vectors = np.random.random((num_entities, 128))  # 128维向量
    
    # 准备插入的数据
    entities = [
        ids.tolist(),
        vectors.tolist()
    ]
    
    # 插入数据
    insert_result = collection.insert(entities)
    print(f"Successfully inserted {num_entities} entities")
    return ids.tolist()

def create_index(collection):
    """创建索引"""
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name="vector", index_params=index_params)
    print("Successfully created index")

def search_data(collection):
    """执行查询操作"""
    search_count = 0
    while True:
        try:
            # 随机生成一个查询向量
            search_vectors = np.random.random((1, 128))
            search_param = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }
            results = collection.search(
                data=[search_vectors[0].tolist()],
                anns_field="vector",
                param=search_param,
                limit=10
            )
            
            # 验证查询结果
            hits = results[0]  # 获取第一个查询向量的结果
            num_hits = len(hits)
            search_count += 1
            
            print(f"Search #{search_count}: Found {num_hits} results, distances: "
                  f"[{hits[0].distance:.4f} - {hits[-1].distance:.4f}]")
            
            time.sleep(1)  # 避免查询太频繁
        except Exception as e:
            print(f"Search error: {e}")
            break

def delete_entities(collection, ids):
    """分批删除实体并定期执行flush"""
    batch_size = 2000
    total_batches = len(ids) // batch_size + (1 if len(ids) % batch_size != 0 else 0)
    
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        expr = f"id in {batch_ids}"
        try:
            collection.delete(expr)
            collection.flush()  # 每删除一批数据后执行flush
            print(f"Successfully deleted and flushed batch {i//batch_size + 1}/{total_batches} ({len(batch_ids)} entities)")
            time.sleep(2)  # 每批删除后暂停2秒
        except Exception as e:
            print(f"Delete error in batch {i//batch_size + 1}: {e}")

def main():
    # 1. 连接到Milvus
    connect_to_milvus()
    
    # 2. 创建集合
    collection = create_collection()
    
    # 3. 插入数据
    start_time = time.time()
    ids = insert_data(collection)
    print(f"Insertion time: {time.time() - start_time:.2f} seconds")
    
    # 4. 创建索引
    start_time = time.time()
    create_index(collection)
    print(f"Index creation time: {time.time() - start_time:.2f} seconds")
    
    # 5. 加载集合
    collection.load()
    print("Successfully loaded collection")
    
    # 6. 并发执行删除和查询
    start_time = time.time()
    
    # 创建一个线程来执行查询
    search_thread = threading.Thread(target=search_data, args=(collection,))
    search_thread.daemon = True  # 设置为守护线程，这样主程序结束时会自动终止
    search_thread.start()
    
    # 执行删除操作
    delete_entities(collection, ids)
    
    print(f"Deletion time: {time.time() - start_time:.2f} seconds")
    
    # 释放集合
    collection.release()
    
    # 断开连接
    connections.disconnect("default")

if __name__ == "__main__":
    main()
