from pymilvus import MilvusClient # 连接服务
client = MilvusClient(uri="http://localhost:19530")

# 创建数据库
db_name = "demo_db"
if db_name not in client.list_databases():
    client.create_database(db_name)
    print(f"数据库 {db_name} 创建成功")
else:
    print(f"数据库 {db_name} 已存在")

# 切换至目标数据库
# client.using_database(db_name)