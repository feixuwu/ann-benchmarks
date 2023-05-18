import clickhouse_connect
import time

from .base import BaseANN


class MyScale(BaseANN):
    def __init__(self, metric, M):
        self.metric = metric
        self.ef_construction = 500
        self.M = M
        self.index_name = "categorical_vector_idx"
        self.table_name = "default.myscale_test"
        
        self.client = clickhouse_connect.get_client(
            host='',
            port=8443,
            username='',
            password=''
        )

    
    

    def fit(self, X):
        
        print("create table------")
        exists_query = f"EXISTS TABLE {self.table_name}"
        table_exists = self.client.command(exists_query)

        if table_exists:
            # 构建删除表的SQL语句
            drop_query = f"DROP TABLE {self.table_name}"

            # 执行删除表操作
            self.client.command(drop_query)
            print(f"表 {self.table_name} 删除成功")
        else:
            print(f"表 {self.table_name} 不存在")
            
        sql=f"""CREATE TABLE {self.table_name}
            (
                id    UInt32,
                data  Array(Float32),
                CONSTRAINT check_length CHECK length(data) = {X.shape[1]}
            )
            ENGINE = MergeTree ORDER BY id"""
        
        print("---try create table:", sql)
        self.client.command(sql)
        print("---create table success")

        print("-----try insert data")
        data_list=[]
        sql = f"INSERT INTO {self.table_name} (id, data) VALUES"
        #values = ','.join(client.escape((name, age)) for name, age in data)

        for i, v in enumerate(X):
            data_list.append([i, v.tolist()])
            if len(data_list) == 1:
                self.client.insert(self.table_name, data_list, column_names=['id', 'data'])
                data_list = []
            
        if len(data_list) > 0:
                self.client.insert(self.table_name, data_list, column_names=['id', 'data'])
                data_list = []
        
        print("-----build index")
        self.client.command(f"""
            ALTER TABLE {self.table_name}
                ADD VECTOR INDEX {self.index_name} data
                TYPE MSTG
            """)
        
        print("-----build success")
        
        get_index_status=f"SELECT status FROM system.vector_indices WHERE table='{self.table_name}'"
        while self.client.command(get_index_status) != 'Built':
            time.sleep(3)
            

    def set_query_arguments(self, ef):
        self.ef = ef

    def query(self, v, n):
        sql=f"""
            SELECT id, date, 
                distance(data, {v.tolist()}) as dist FROM default.myscale_categorical_search ORDER BY dist LIMIT {n}
            """
        res  = self.client.command(sql)
        res_list = []
        for res_id_dis in res.result_rows:
            res_list.append((res_id_dis[0], res_id_dis[1]))

        return res_list

    def __str__(self):
        return f"MyScale(M={self.M}, ef={self.ef})"
    
        
        
