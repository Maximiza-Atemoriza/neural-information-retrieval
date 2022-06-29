from .utils import remove_stopwords
import dill


class IRDataset:
    def __init__(self) -> None:
        self.queries: dict[int, str] | None = None
        self.docs: dict[int, str] | None = None
        self.qrels: list | None = None

    def save(self, path):
        file = open(path, "wb")
        dill.dump(self, file)
        file.close()

    @staticmethod
    def load(path):
        file = open(path, "rb")
        model = dill.load(file)
        file.close()
        return model

    def process_dataset(self, dataset, verbose=False):
        self.docs = {}
        for doc in dataset.docs_iter():
            if verbose:
                print(f"Processing document {doc.doc_id}...")
            self.docs[int(doc.doc_id)] = remove_stopwords(doc.text)

        self.queries = {}
        for query in dataset.queries_iter():
            if verbose:
                print(f"Processing query {query.query_id}...")
            self.queries[int(query.query_id)] = remove_stopwords(query.text)

        self.qrels = []
        for index, qrel in enumerate(dataset.qrels_iter()):
            if verbose:
                print(f"Processing document-query relation {index}...")
            self.qrels.append(
                (
                    int(qrel.doc_id),
                    int(qrel.query_id),
                    int(qrel.relevance),
                )
            )
        if verbose:
            print("Processing done!")

    def queries_iter(self):
        if self.queries is None:
            raise Exception("No dataset found!")
        return self.queries.items()

    def docs_iter(self):
        if self.docs is None:
            raise Exception("No dataset found!")
        return self.docs.items()

    def qrels_iter(self):
        if self.qrels is None:
            raise Exception("No dataset found!")
        return self.qrels
