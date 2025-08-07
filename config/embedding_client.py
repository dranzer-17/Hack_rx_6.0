import httpx
from langchain.embeddings.base import Embeddings
from typing import List
import os

# The official URL for the BGE-Small model on the Hugging Face Inference API
MODEL_API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5"

class HuggingFaceInferenceAPIEmbeddings(Embeddings):
    """
    A robust client for the Hugging Face Inference API.
    - Sends data in batches for efficiency.
    - Adds the BGE prefix for queries.
    - Uses a long timeout to handle server cold starts.
    """
    def __init__(self, api_token: str, batch_size: int = 32):
        self.api_token = api_token
        self.batch_size = batch_size
        self.headers = {"Authorization": f"Bearer {api_token}"}
        # Use a long timeout to handle model cold starts
        self.timeout = httpx.Timeout(300.0) # 5 minutes

    def _embed_batch(self, batch: List[str], is_query: bool) -> List[List[float]]:
        """Sends a single batch of texts to the HF API."""
        texts_to_send = batch
        # For BGE models, we add the prefix to queries before sending.
        if is_query:
            texts_to_send = ["Represent this sentence for searching relevant passages: " + text for text in batch]

        try:
            # The HF API expects a specific JSON format
            response = httpx.post(
                MODEL_API_URL,
                headers=self.headers,
                json={"inputs": texts_to_send, "options": {"wait_for_model": True}},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"Hugging Face Inference API call failed with status {e.response.status_code}: {e.response.text}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds documents by breaking them into batches for efficiency."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            print(f"Embedding batch {i//self.batch_size + 1} of {len(texts)//self.batch_size + 1}...")
            all_embeddings.extend(self._embed_batch(batch, is_query=False))
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query."""
        embeddings = self._embed_batch([text], is_query=True)
        return embeddings[0]