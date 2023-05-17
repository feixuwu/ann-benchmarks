import pinecone

from .base import BaseANN


class PineCone(BaseANN):
    def __init__(self, metric, M):
        self.metric = metric
        self.ef_construction = 500
        self.M = M
        self.index_name = "ann"
        pinecone.init(api_key="", environment="")

    def fit(self, X):
        
        try:
            pinecone.delete_index(self.index_name)
        except:
            print("----no index exist")
        finally:
            print("----delete index finish")

        pinecone.create_index(self.index_name, dimension=X.shape[1], metric={"angular": "cosine", "euclidean": "euclidean"}[self.metric])
        print("---create index success")
        self.index = pinecone.Index(self.index_name)

        print("---insert vector, todo:batch")
        for i, v in enumerate(X):
            self.index.upsert([
                (str(i), v.tolist())
            ])

    def set_query_arguments(self, ef):
        self.ef = ef

    def query(self, v, n):
        return self.index.query(
        vector=v.tolist(),
        top_k=n,
        include_values=True
        )

    def __str__(self):
        return f"PineCone(M={self.M}, ef={self.ef})"
