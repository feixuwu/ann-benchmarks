import pinecone

from .base import BaseANN


class PineCone(BaseANN):
    def __init__(self, metric, M):
        self.metric = metric
        self.ef_construction = 500
        self.M = M
        self.index_name = "ann"
        pinecone.init(api_key="45deb16a-4b38-40ef-92d7-ec7c05e92285", environment="us-central1-gcp")

    def fit(self, X):
        print("create index")
        pinecone.delete_index(self.index_name)
        self.index = pinecone.create_index(self.index_name, dimension=X.shape[1], metric=self.metric)
        
        print("insert vector")
        for i, v in enumerate(X):
            self.index.upsert([
                (i, v)
            ])

    def set_query_arguments(self, ef):
        self.ef = ef

    def query(self, v, n):
        return self.index.query(
        vector=v,
        top_k=n,
        include_values=True
        )

    def __str__(self):
        return f"PineCone(M={self.M}, ef={self.ef})"
