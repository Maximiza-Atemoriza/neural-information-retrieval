from typing import Any, List, Tuple
import streamlit as st
import ir_datasets

from src.ir_models.neural_network import NetRank
from src.ir_models.neural_network_regression import NetRankRegression
from src.ir_models.rnn import RNN as Rnn
from src.ir_models.rnn_regression import RNNRegression as RnnRegression
from src.ir_models.vector import VectorModel
from src.datasets import IRDataset


RankedDocs = List[Tuple[Any, int]]

LR_PREFIX = "Learning to Rank"
LR_RNN_PREFIX = f"{LR_PREFIX} using RNN"

model_possible_sections = [LR, LR_REGRESSION, RNN, RNN_REGRESSION, VECT] = [
    f"{LR_PREFIX} (Classifier)",
    f"{LR_PREFIX} (Regression)",
    f"{LR_RNN_PREFIX} (Classifier)",
    f"{LR_RNN_PREFIX} (Regression)",
    "Vectorial",
]
dataset_possible_sections = [CRAN, VASWANI, ANTIQUE] = [
    "Cranfield (Processed)",
    "Vaswani (Processed)",
    "Antique (Processed)",
]

name_to_model = {
    LR: NetRank,
    LR_REGRESSION: NetRankRegression,
    RNN: Rnn,
    RNN_REGRESSION: RnnRegression,
}

# --------------------------------- Util Functions ---------------------------------
def preprocess_datatset(dataset_name: str, verbose: bool, lemma: bool):
    dataset = ir_datasets.load(dataset_name)
    d = IRDataset()
    d.process_dataset(dataset, verbose, lemma)
    d.save(f"processed_{dataset_name}")
    return d


def prepare_dataset(dataset: str):
    dataset_name: str
    cat: int
    if dataset == CRAN:
        dataset_name = "cranfield"
        cat = 5
    elif dataset == VASWANI:
        dataset_name = "vaswani"
        cat = 1
    elif dataset == ANTIQUE:
        dataset_name = "antique"
        cat = 2
    else:
        raise Exception(f"Dataset option {dataset} not yet supported")

    try:
        return IRDataset.load(f"processed_{dataset_name}"), cat
    except FileNotFoundError:
        st.write(
            f"Cache miss! {dataset} does not exists, processing. This may take a while."
        )
        d = preprocess_datatset(dataset_name, True, True)
        return d, cat


# Cache accordingly to input parameters
@st.experimental_singleton(suppress_st_warning=True)  # pyright: ignore
def prepare_model(
    model_select: str, dataset_select: str
) -> Tuple[VectorModel, NetRank | None, Any]:
    dataset, cat = prepare_dataset(dataset_select)

    if model_select == VECT or model_select.startswith(LR_PREFIX):
        st.write(f"Cache Miss! Training Vector model with {dataset_select}")
        v = VectorModel()
        v.index(dataset)
        st.success(f"Vector model trained succesfully with {dataset_select}")
        if model_select.startswith(LR_PREFIX):
            st.write(f"Cache Miss! Training {model_select} model with {dataset_select}")
            lr: NetRank = name_to_model[model_select]()
            lr.train(dataset, cat)
            st.success(
                f"{model_select} model trained succesfully with {dataset_select}"
            )
            return (v, lr, dataset)
        else:
            return (v, None, dataset)

    raise Exception(f"Model {model_select} not yet supported")


def predict(
    vector_model: VectorModel,
    netrank_model: NetRank | None,
    dataset: Any,
    query: str,
) -> RankedDocs:
    vector_ranked: RankedDocs = vector_model.get_ranked_docs(query, dataset)
    vector_ranked.sort(key=lambda x: -x[1])
    if netrank_model is None:
        return vector_ranked

    net_ranked: RankedDocs = []
    for (doc, _) in vector_ranked[0 : min(max_vect_use, len(vector_ranked))]:
        doc_text: str = doc.text
        doc_score: int = netrank_model.predict_score(doc_text, query)
        net_ranked.append((doc, doc_score))

    net_ranked.sort(key=lambda x: -x[1])
    return net_ranked


def printItmes(ranked: RankedDocs, amount: int):
    amount = min(len(ranked), amount)
    if amount == 0:
        st.warning("No items to display")
    else:
        st.text(f"Showing {amount} items")
    markdow_table = "Id | Snippet | Score |\n" "| ---- | ---- | ---- |\n"
    markdow_table += "\n".join(
        f"{ranked[i][0].doc_id} |  {' '.join(ranked[i][0].text.split()[0:5])} | {ranked[i][1]}|"
        for i in range(amount)
    )
    st.write(markdow_table)


# ------------------------------ Main Visual Stuff ------------------------------
st.title("Learning to Rank")

model_select: str = st.selectbox(
    "Select Model", options=model_possible_sections, index=1
)  # pyright: ignore

dataset_select: str = st.selectbox(
    "Select dataset", options=dataset_possible_sections, index=0
)  # pyright: ignore

max_results = st.sidebar.select_slider("Results Amount", [10 + i for i in range(91)])

max_vect_use = st.sidebar.select_slider(
    "Docs extracted from Vectorial model",
    [100 + 50 * i for i in range(9)],
    value=100,
    disabled=not model_select.startswith(LR_PREFIX),
)

max_rel_test = st.sidebar.select_slider(
    "Max relevance test size",
    [100 + 50 * i for i in range(9)],
    value=100,
    disabled=not model_select.startswith(LR_PREFIX),
)

get_relevance = False
if model_select.startswith(LR_PREFIX):
    get_relevance = st.checkbox(
        "Get model relevance metrics using predefined queries", value=True
    )

if get_relevance:
    query = ""
else:
    query = st.text_input("Query:", placeholder="I dare you to query me!")

search = st.button(
    "Get Relevance" if get_relevance else "Go!",
    disabled=not get_relevance and len(query) == 0,
)

if search and not get_relevance:
    vector_model, netrank_model, dataset = prepare_model(model_select, dataset_select)

    st.write(f"Querying _{query}_ ...")
    ranked = predict(vector_model, netrank_model, dataset, query)
    printItmes(ranked, max_results)

elif search and get_relevance:
    vector_model, netrank_model, dataset = prepare_model(model_select, dataset_select)
    assert netrank_model is not None

    # relevant = [relevant for relevant in dataset.qrels_iter() if relevant.relevance > 0]
    docs = {doc.doc_id: doc.text for doc in dataset.docs_iter()}
    queries = {query.query_id: query.text for query in dataset.queries_iter()}
    qreldocs = dict()
    for qrel in dataset.qrels_iter():
        if qrel.relevance < 3:
            continue
        if qrel.query_id in qreldocs:
            qreldocs[qrel.query_id].append(int(qrel.doc_id))
        else:
            qreldocs[qrel.query_id] = [int(qrel.doc_id)]

    precission_total = []
    recall_total = []
    querysings = set()
    for i, qrel in enumerate(dataset.qrels_iter()):
        if i > 0 and i % 25:
            st.write("Procesando", "==" * (i % 25))

        if qrel.query_id in querysings:
            continue
        querysings.add(qrel.query_id)
        try:
            query = queries[qrel.query_id]
            dummy = qreldocs[qrel.query_id]
        except KeyError:
            continue

        vector_ranked_docs = vector_model.get_ranked_docs(query, dataset)
        vector_ranked_docs.sort(key=lambda x: -x[1])
        vector_ranked_docs = vector_ranked_docs[0:max_vect_use]

        net_ranked_docs = []
        for (doc, _) in vector_ranked_docs[
            0 : min(max_vect_use, len(vector_ranked_docs))
        ]:
            doc_text: str = doc.text
            doc_score: int = netrank_model.predict_score(doc_text, query)
            net_ranked_docs.append((doc, doc_score))

        net_ranked_docs.sort(key=lambda x: -x[1])
        expected_ranked_docs = qreldocs[qrel.query_id]

        intersect = set([int(d[0].doc_id) for d in net_ranked_docs]).intersection(
            set(expected_ranked_docs)
        )
        precission_total.append(len(intersect) / max_vect_use)
        recall_total.append(len(intersect) / len(expected_ranked_docs))

    recall = sum(recall_total) / len(recall_total)
    precission = sum(precission_total) / len(precission_total)
    fscore = 2 * (precission * recall) / (precission + recall)

    st.write(
        " | Relevance Metric | Result |\n",
        " | ---- | ---- |\n",
        f"| Precission | {precission}\n",
        f"| Recall | {recall}\n",
        f"| F1 | {fscore}\n",
    )
