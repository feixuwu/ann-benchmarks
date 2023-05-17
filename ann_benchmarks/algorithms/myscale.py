import clickhouse_connect
import time

from .base import BaseANN


class PineCone(BaseANN):
    def __init__(self, metric, M):
        self.metric = metric
        self.ef_construction = 500
        self.M = M
        self.index_name = "ann"
        self.client = clickhouse_connect.get_client(
            host='',
            port=8443,
            username='',
            password=''
        )

    def fit(self, X):
        
        print("create table------")
        sql="""CREATE TABLE default.myscale_categorical_search
            (
                id    UInt32,
                data  Array(Float32),
                CONSTRAINT check_length CHECK length(data) = self.M
            )
            ENGINE = MergeTree ORDER BY id"""
        

        self.client.command(sql)

        
        print("---insert vector")
        data_list=[]
        for i, v in enumerate(X):
            sql = """INSERT INTO default.myscale_categorical_search
                values(i, v.tolist() ) """
            self.client.command(sql)

        
        print("-----build index")
        self.client.command("""
            ALTER TABLE default.myscale_categorical_search
                ADD VECTOR INDEX categorical_vector_idx data
                TYPE MSTG
            """)
        
        get_index_status="SELECT status FROM system.vector_indices WHERE table='myscale_categorical_search'"
        while self.client.command(get_index_status) != 'Built':
            time.sleep(3)
            

    def set_query_arguments(self, ef):
        self.ef = ef

    def query(self, v, n):
        sql="""
            SELECT id, date, 
                distance(data, {target_row_data}) as dist FROM default.myscale_categorical_search ORDER BY dist LIMIT 10
            """
        res  = self.client.command(sql)
        res_list = []
        
        for res_id_dis in res.result_rows:
            res_list.append((res_id_dis[0], res_id_dis[1]))

        return res_list

    def __str__(self):
        return f"MyScale(M={self.M}, ef={self.ef})"
