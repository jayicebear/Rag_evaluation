# poetry run python -m chopperCommander.test.mongoVector_search_test

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)

def get_embedding(texts, precision="float32"):
    """문장 리스트를 벡터화"""
    return model.encode(texts, convert_to_numpy=True).astype(precision).tolist()

# client = MongoClient("mongodb://zium:zium1207%21%21@192.168.0.43:27017/admin?directConnection=true")
client = MongoClient("mongodb://zium:zium1207!!@192.168.0.43:27017/?authSource=admin")

# db = client["test"]
# collection = db["test"]

db = client["rag_test"]
collection = db["chunks_for_test"]
#ids = ["doc_documentIdtest_chunk_1", "doc_documentIdtest_chunk_2"] # Function to get the results of a vector search query


def get_query_results(query):
   query_embedding = get_embedding(query)
   pipeline = [
      {
            "$vectorSearch": {
               "index": "vector_search",
               "queryVector": query_embedding,
               "path": "chunk_embedding",
               "exact": True,
               "limit": 5
            }
      }, {
            "$project": {
               "_id": 0,
               # "summary": 1,
               # "listing_url": 1,
               "score": {
                  "$meta": "vectorSearchScore"
               }
            }
      }
   ]
   results = collection.aggregate(pipeline)
   array_of_results = []
   for doc in results:
      array_of_results.append(doc)
   return array_of_results

if __name__ == "__main__":
    import pprint
    pprint.pprint(get_query_results("beach house"))