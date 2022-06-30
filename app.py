from typing import Any, List, Tuple
import streamlit as st
import ir_datasets

from src.ir_models.neural_network import NetRank
from src.ir_models.neural_network_regression import NetRankRegression
from src.ir_models.vector import VectorModel
from src.datasets import IRDataset


RankedDocs = List[Tuple[Any, int]]

LR_PREFIX = "Learning to Rank"

model_possible_sections = [LR, LR_REGRESSION, VECT] = [
    f"{LR_PREFIX} (Regression)",
    f"{LR_PREFIX} (Classifier)",
    "Vectorial",
]
dataset_possible_sections = [CRAN, VASWANI, CRANMOD] = [
    "Cranfield",
    "Vaswani",
    "Processed Cranfield",
]

# --------------------------------- Util Functions ---------------------------------
def preprocess_datatset(dataset_name: str, verbose: bool, lemma: bool):
    dataset = ir_datasets.load(dataset_name)
    d = IRDataset()
    d.process_dataset(dataset, verbose, lemma)
    d.save(f"processed_{dataset_name}")


def prepare_dataset(dataset: str):
    if dataset == CRAN:
        return ir_datasets.load("cranfield"), 5
    if dataset == CRANMOD:
        try:
            return IRDataset.load("processed_cranfield"), 5
        except FileNotFoundError:
            st.write(
                f"Cache miss! {dataset} does not exists, processing. This may take a while."
            )
            preprocess_datatset("cranfield", True, True)
            return IRDataset.load("preprocess_datatset"), 5
    if dataset == VASWANI:
        return ir_datasets.load("vaswani"), 2
    raise Exception(f"Dataset option {dataset} not yet supported")


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
            lr = NetRankRegression() if model_select == LR else NetRank()
            lr.train(dataset, cat)
            st.success(
                f"{model_select} model trained succesfully with {dataset_select}"
            )
            return (v, lr, dataset)
        else:
            return (v, None, dataset)

    raise Exception(f"Model option {model_select} not yet supported")


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
st.title("Information Retrieval Final Project")

model_select: str = st.selectbox(
    "Select Model", options=model_possible_sections
)  # pyright: ignore

dataset_select: str = st.selectbox(
    "Select dataset", options=dataset_possible_sections, index=2
)  # pyright: ignore

max_results = st.sidebar.select_slider("Results Amount", [10 + i for i in range(91)])

max_vect_use = st.sidebar.select_slider(
    "Docs extracted from Vectorial model",
    [100 + 50 * i for i in range(9)],
    value=100,
    disabled=model_select != LR,
)


query = st.text_input("Query:", placeholder="I dare you to query me!")

search = st.button(
    "Empty Query" if (len(query) == 0) else "Go!", disabled=len(query) == 0
)

if search:
    vector_model, netrank_model, dataset = prepare_model(model_select, dataset_select)

    st.write(f"Querying _{query}_ ...")
    ranked = predict(vector_model, netrank_model, dataset, query)

    printItmes(ranked, max_results)
