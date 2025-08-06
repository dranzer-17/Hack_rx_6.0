import os
import weaviate
from langchain_weaviate import WeaviateVectorStore  as Weaviate
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings= HuggingFaceEmbeddings(model= 'BAAI/bge-large-en-v1.5')

# Load credentials
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

def get_vector_store():
    
    client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,                                    # Replace with your Weaviate Cloud URL
    auth_credentials=weaviate_api_key             # Replace with your Weaviate Cloud key
    )

    # ✅ Initialize LangChain Weaviate vector store
    vector_store = Weaviate(
        client=client,
        index_name="InsurancePolicy",
        text_key="text",
        embedding = embeddings
    )
    
    return vector_store,client