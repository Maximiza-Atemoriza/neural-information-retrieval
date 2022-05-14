from parsers.cran_parser import parse_cran
from information_retrieval import vector_model
import typer


def main(dataset: str, ranking: int):
    if "cran" == dataset.lower():
        corpus = parse_cran()
        documents = [doc[".W"] for doc in corpus]
    else:
        typer.echo(typer.style("Unknown dataset :(", fg=typer.colors.RED, bold=True))
        exit(0)

    typer.secho("Indexing documents...", fg=typer.colors.BLUE)
    vectorizer, idf, tfidf = vector_model.documents_tfidf(documents)
    typer.secho("Done!:)", fg=typer.colors.BLUE)

    while True:
        typer.secho("Enter query:", fg=typer.colors.BRIGHT_WHITE, bold=True)
        query = input()
        query = query.replace("\n", " ")
        typer.echo(typer.style("Document ID:", fg=typer.colors.GREEN, bold=True))
        query_vector = vector_model.query_tfidf(query, vectorizer, idf)
        for doc_id, score in vector_model.relevant_documents(query_vector, tfidf)[
            :ranking
        ]:
            print(doc_id)


if __name__ == "__main__":
    typer.run(main)
