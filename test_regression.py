from src.ir_models.neural_network_regression import NetRankRegression
import ir_datasets

a = NetRankRegression()
dataset = ir_datasets.load("cranfield")

doc_id = 1
query_id = 29

# Get doc
d = None
for doc in dataset.docs_iter():
    d = doc
    if int(d.doc_id) == doc_id:
        break

# Get query
q = None
for query in dataset.queries_iter():
    q = query
    if int(q.query_id) == query_id:
        break

q = "experimental investigation of the aerodynamics of a wing in a slipstream"
print("***********************DOCUMENT************************")
print(d)
print("*******************************************************")
print("***********************QUERY************************")
print(q)
print("*******************************************************")
a.train(dataset)
predicted_score = a.predict_score(d.text, q)
print(f"Predicted score: {predicted_score}")
