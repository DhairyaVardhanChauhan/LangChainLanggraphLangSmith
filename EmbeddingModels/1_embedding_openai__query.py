from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from scipy.stats import cosine
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about rohit'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = (cosine_similarity([query_embedding], doc_embeddings))[0]

lis = list(enumerate(scores))
sorted_scores = sorted(lis, key=lambda x: x[1], reverse=True)
print(documents[sorted_scores[0][0]])

