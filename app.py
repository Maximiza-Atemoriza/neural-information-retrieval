import streamlit as st

from src.ir_models.neural_network import NetRank
from src.ir_models.vector import Vectorial


def prepare_dataset(dataset: str) -> None:
    pass


def prepare_model(model: str, dataset: None) -> NetRank | Vectorial | None:
    pass


st.title("Learning to Rank")

model_possible_sections = [LR, VECT] = ["Learning to Rank (LR)", "Vectorial"]
model_select: str = st.selectbox(
    "Select Model", options=model_possible_sections
)  # pyright: ignore

dataset_possible_sections = [CRAN, WATSANI] = ["CRAN", "WATSANI"]
dataset_select: str = st.selectbox(
    "Select dataset", options=dataset_possible_sections
)  # pyright: ignore

query = st.text_input("Query:", placeholder="I dare you to query me!")

search = st.button("Go!", disabled=len(query) == 0)

if search:
    # Notify about query and query parameters
    st.write(
        f"Querying _{query}_ with {model_select} model in {dataset_select} dataset"
    )

    dataset = prepare_dataset(dataset_select)
    st.success("Dataset loaded")

    model = prepare_model(model_select, dataset)
    st.success("Model loaded")
