from config.vector_store import get_vector_store
vector_store,_ = get_vector_store()

def reteriever_tool(query: str):
    reteriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 50, 'fetch_k': 500}
    )
    
    related_docs = reteriever.invoke(query)
    
    return related_docs