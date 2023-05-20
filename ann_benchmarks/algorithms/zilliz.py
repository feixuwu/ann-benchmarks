from pymilvus import connections
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
from pymilvus import utility

from .base import BaseANN


class Zilliz(BaseANN):
    def __init__(self, metric, nouse):
        self.metric = metric
        self.field_name = "vector"
        self.collection_name = "collection"

        connections.connect(
            alias="default", 
            uri='https://in01-626c35782a70b5e.gcp-us-west1.vectordb.zillizcloud.com:443', # Endpoint URI obtained from Zilliz Cloud
            secure=True,
            user='db_admin', # Username specified when you created this database
            password=''
        )

    def fit(self, X):
        
        try:
            self.collection = Collection(self.collection_name)    
            self.collection.release() 
            self.collection.drop_index()
            utility.drop_collection(self.collection_name)
        except:
            pass

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name=self.field_name, dtype=DataType.FLOAT_VECTOR, dim=X.shape[1])
        ]

        schema = CollectionSchema(
            fields,
            description="Schema of Medium articles"
        )

        self.collection = Collection(
            name=self.collection_name, 
            description="Medium articles published between Jan 2020 to August 2020 in prominent publications",
            schema=schema
        )

        ids = []
        vec = []

        for i, v in enumerate(X):
            ids.append(i)
            vec.append(v.tolist())

            if len(ids) > 10000:
                self.collection.insert([ids, vec])
                print(f"-------insert{i}")
                ids = []
                vec = []
        
        if len(ids) > 0:
            self.collection.insert([ids, vec])

        index_params = {
            'index_type': 'AUTOINDEX',
            'metric_type': {"angular": "IP", "euclidean": "L2"}[self.metric],
            'params': {}
        }

        self.collection.release()
        self.collection.create_index(self.field_name, index_params)
        self.collection.load()
        

    def set_query_arguments(self, level):
        self.level = level

    def query(self, v, n):
        search_params = {
            "metric_type": {"angular": "IP", "euclidean": "L2"}[self.metric],
            "params":{"level":self.level}
        }
        res = self.collection.search(
            data = [v.tolist()],
            anns_field=self.field_name,
            param=search_params,
            limit = n
        )
        
        return res[0].ids
    
    def __str__(self):
        return f"Zilliz(level={self.level})"
