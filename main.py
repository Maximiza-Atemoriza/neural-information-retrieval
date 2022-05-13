from parsers.cran_parser import parse_cran
from information_retrieval import vector_model

corpus = parse_cran()
documents = [doc[".W"] for doc in corpus]

documents = [ 'leon leon leon','leon leon leon zorro ', 'leon zorro nutria ', 'nutria' ]
vectorizer, idf , tfidf = vector_model.documents_tfidf(documents) 
query = 'what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft'
query = 'nutria leon leon casa'
query_vector = vector_model.query_tfidf(query,vectorizer, idf)

print(vector_model.relevant_documents(query_vector,tfidf)[:10])






