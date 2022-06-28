from typing import Any, List, Tuple
import streamlit as st

from src.ir_models.neural_network import NetRank
from src.ir_models.vector import Vectorial

import ir_datasets

RankedDocs = List[Tuple[str, int]]

# --------------------------------- Util Functions ---------------------------------
def prepare_dataset(dataset: str):
    if dataset == CRAN:
        return ir_datasets.load("cranfield")
    if dataset == VASWANI:
        return ir_datasets.load("vaswani")
    raise Exception(f"Dataset option {dataset} not yet supported")


# Cache accordingly to input parameters
@st.experimental_singleton(suppress_st_warning=True)  # pyright: ignore
def prepare_model(
    model_select: str, dataset_select: str
) -> Tuple[NetRank | Vectorial, Any]:
    dataset = prepare_dataset(dataset_select)
    if model_select == LR:
        st.write(f"Cache Miss! Training LR model with {dataset_select}")
        lr = NetRank()
        lr.train(dataset)
        st.success(f"LR model trained succesfully with {dataset_select}")
        return (lr, dataset)
    if model_select == VECT:
        pass
    raise Exception(f"Model option {model} not yet supported")


def predict(model: NetRank | Vectorial, dataset: Any, query: str) -> RankedDocs:
    assert isinstance(model, NetRank)
    ranked: RankedDocs = []
    for doc in dataset.docs_iter():
        doc_text: str = doc.text
        doc_score: int = model.predict_score(doc_text, query)
        ranked.append((doc_text, doc_score))

    ranked.sort(key=lambda x: x[1])
    return ranked


# ---------------------------------- Visual Stuff ----------------------------------
st.title("Information Retrieval Final Project")

model_possible_sections = [LR, VECT] = ["Learning to Rank (LR)", "Vectorial"]
model_select: str = st.selectbox(
    "Select Model", options=model_possible_sections
)  # pyright: ignore

dataset_possible_sections = [CRAN, VASWANI] = ["Cranfield", "Vaswani"]
dataset_select: str = st.selectbox(
    "Select dataset", options=dataset_possible_sections
)  # pyright: ignore

query = st.text_input("Query:", placeholder="I dare you to query me!")

search = st.button("Go!", disabled=len(query) == 0)

if search:
    # Notify about query and query parameters
    model, dataset = prepare_model(model_select, dataset_select)

    st.write(f"Querying _{query}_ ...")
    ranked = predict(model, dataset, query)

    for doc_text, doc_score in ranked:
        st.write(doc_text, " ", doc_score)
