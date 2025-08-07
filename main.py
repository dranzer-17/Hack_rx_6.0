import asyncio
import os
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, Field
from typing import List

# Import our custom modules from their correct locations
from config.document_processor import process_document_from_url
from config.vector_store import get_vector_store
from teams.Round_Robin_Team import get_team
from config.prompt import qa_validator_system_message
from config.tools import reteriever_tool # The simple, unfiltered tool

# --- Pydantic Models for the API Request and Response ---
class HackRxRequest(BaseModel):
    """
    Defines the structure for the incoming POST request.
    """
    documents: HttpUrl = Field(..., description="URL to the policy document to be processed.")
    questions: List[str] = Field(..., description="A list of questions to ask about the document.")

class HackRxResponse(BaseModel):
    """
    Defines the structure for the JSON response.
    """
    answers: List[str] = Field(..., description="A list of answers corresponding to the questions asked.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="HackRx Insurance Policy Q&A",
    description="API to answer questions about an insurance policy document."
)

@app.on_event("startup")
def startup_event():
    """Initializes the vector store and client when the app starts."""
    global vector_store, client
    vector_store, client = get_vector_store()

@app.on_event("shutdown")
def shutdown_event():
    """Closes the client connection when the app shuts down."""
    client.close()

@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_hackrx_flow(request: HackRxRequest):
    """
    This endpoint ingests a document and runs the agentic workflow for each question.
    WARNING: This version adds all documents to the same shared database.
    """
    print(f"Processing document from URL: {request.documents}")

    # 1. Process document from the URL
    try:
        chunks = await process_document_from_url(str(request.documents))
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract any content from the document.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

    # 2. Ingest document chunks into Weaviate
    # All chunks from all requests are added to the same database.
    vector_store.add_documents(chunks)
    print(f"Ingested {len(chunks)} chunks into the shared database.")

    # 3. Loop through questions and run the agent team
    final_answers = []
    
    for question in request.questions:
        print(f"--- Running team for question: '{question}' ---")
        try:
            # This creates the standard team, which will use the simple, unfiltered tool.
            # The tool will search across ALL documents ever ingested.
            team_1 = get_team() 

            chat_result = await team_1.run(task=question)

            if chat_result and chat_result.chat_history:
                # The final message should be the answer from the Validator agent.
                answer = chat_result.chat_history[-1]['content'].replace("STOP", "").strip()
                final_answers.append(answer)
            else:
                final_answers.append("The agent team failed to produce a valid answer.")

        except Exception as e:
            print(f"An error occurred while running the agent team for a question: {e}")
            final_answers.append("An error occurred while processing this question.")
            
    return HackRxResponse(answers=final_answers)