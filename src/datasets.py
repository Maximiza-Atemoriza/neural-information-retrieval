from .utils import remove_stopwords
import dill
from collections import namedtuple

Query = namedtuple("Query", "query_id text")
Document = namedtuple("Document", "doc_id text")
Qrel = namedtuple("Qrel", "query_id doc_id relevance")


class IRDataset:
    def __init__(self) -> None:
        self.queries: dict[int, str] | None = None
        self.docs: dict[int, str] | None = None
        self.qrels: list | None = None

    def save(self, path):
        with open(path, "wb") as f:
            dill.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            model = dill.load(f)
            return model

    def process_dataset(self, dataset, verbose=False, lemma=True):
        self.docs = {}

        pipeline = lambda x: remove_stopwords(x, lemma)

        for doc in dataset.docs_iter():
            if verbose:
                print(f"Processing document {doc.doc_id}...")
            if doc.text == "":
                continue
            self.docs[int(doc.doc_id)] = pipeline(doc.text)

        self.queries = {}
        for query in dataset.queries_iter():
            if verbose:
                print(f"Processing query {query.query_id}...")
            self.queries[int(query.query_id)] = pipeline(query.text)

        self.qrels = []
        for index, qrel in enumerate(dataset.qrels_iter()):
            if verbose:
                print(f"Processing document-query relation {index}...")
            self.qrels.append(
                (Qrel(int(qrel.query_id), int(qrel.doc_id), int(qrel.relevance)))
            )
        if verbose:
            print("Processing done!")

    def queries_iter(self):
        if self.queries is None:
            raise Exception("No dataset found!")
        return [Query(*item) for item in self.queries.items()]

    def docs_iter(self):
        if self.docs is None:
            raise Exception("No dataset found!")
        return [Document(*item) for item in self.docs.items()]

    def qrels_iter(self):
        if self.qrels is None:
            raise Exception("No dataset found!")
        return [Qrel(*item) for item in self.qrels]
