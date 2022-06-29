from typing import Any, List, Tuple
import streamlit as st

from src.ir_models.neural_network import NetRank
from src.ir_models.vector import Vectorial

import ir_datasets

RankedDocs = List[Tuple[str, int]]

# --------------------------------- Util Functions ---------------------------------
def prepare_dataset(dataset: str):
    if dataset == CRAN:
        return ir_datasets.load("cranfield"), 5
    if dataset == VASWANI:
        return ir_datasets.load("vaswani"), 2
    raise Exception(f"Dataset option {dataset} not yet supported")


# Cache accordingly to input parameters
@st.experimental_singleton(suppress_st_warning=True)  # pyright: ignore
def prepare_model(
    model_select: str, dataset_select: str
) -> Tuple[NetRank | Vectorial, Any]:
    dataset, cat = prepare_dataset(dataset_select)
    if model_select == LR:
        st.write(f"Cache Miss! Training LR model with {dataset_select}")
        lr = NetRank()
        lr.train(dataset, cat)
        st.success(f"LR model trained succesfully with {dataset_select}")
        return (lr, dataset)
    if model_select == VECT:
        pass
    raise Exception(f"Model option {model} not yet supported")


def predict(model: NetRank | Vectorial, dataset: Any, query: str) -> RankedDocs:
    assert isinstance(model, NetRank), print(model)
    ranked: RankedDocs = []
    for doc in dataset.docs_iter():
        # Patch for cranfield error
        # if int(doc.doc_id) == 470:
        # break
        doc_text: str = doc.text[:10]
        doc_score: int = int(model.predict_class(doc_text, query))
        ranked.append((doc_text, doc_score))

    ranked.sort(key=lambda x: x[1])
    return ranked


def printItmes(ranked: RankedDocs, amount: int):
    markdow_table = "| document name | score |\n" "|    ----       |  ---- |\n"
    markdow_table += "\n".join(
        f"{ranked[i][0]} | {ranked[i][1]}|" for i in range(amount)
    )
    st.write(markdow_table)


# ------------------------------ Side Visual Stuff ------------------------------
max_results = st.sidebar.select_slider("Results Amount", [10 + i for i in range(41)])

# ------------------------------ Main Visual Stuff ------------------------------
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

search = st.button(
    "Empty Query" if (len(query) == 0) else "Go!", disabled=len(query) == 0
)

if search:
    # Notify about query and query parameters
    model, dataset = prepare_model(model_select, dataset_select)

    st.write(f"Querying _{query}_ ...")
    ranked = predict(model, dataset, query)

    printItmes(ranked, 10)
