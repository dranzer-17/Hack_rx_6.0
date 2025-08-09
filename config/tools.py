from .vector_store import get_vector_store
from typing import List

def reteriever_tool(query:str ):
    """
    Reterieves relevant documents from the vector store based on a query.
    """
    vector_store, _ = get_vector_store()

    # This retriever searches the ENTIRE database.
    reteriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 50, 'fetch_k': 500}
    )
    
    related_docs = reteriever.invoke(query)
    
    # --- THIS IS THE FIX ---
    # Convert the list of complex Document objects into a simple list of strings.
    return related_docs