import os
import weaviate
from langchain_weaviate import WeaviateVectorStore
from dotenv import load_dotenv

# Import our new client class
from .embedding_client import HuggingFaceInferenceAPIEmbeddings

load_dotenv()

# Load credentials from .env file
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
# Get the new Hugging Face token
hf_api_token = os.getenv("HUGGINGFACE_API_TOKEN")

def get_vector_store():
    if not hf_api_token:
        raise ValueError("HUGGINGFACE_API_TOKEN is not set in the environment variables.")

    # Create an instance of our new embedding client
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_token=hf_api_token)

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key)
    )

    vector_store = WeaviateVectorStore(
        client=client,
        index_name="InsurancePolicy",
        text_key="text",
        embedding=embeddings  # Use the new embedding client instance
    )
    
    return vector_store, client