from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import json
import re

# Your custom application imports
from config.vector_store import get_vector_store
from teams.Round_Robin_Team import get_team

# --- Initialization of your application components ---
# This part remains the same. It sets up your Weaviate client and agent team.
vector_store, client = get_vector_store()
team_1 = get_team()


# --- Pydantic Models for Request and Response ---
# These models define the expected structure for the API call, matching the problem description.
class HackRxInput(BaseModel):
    documents: str
    questions: List[str]

class HackRxOutput(BaseModel):
    answers: List[str]


# --- FastAPI Application Instance ---
app = FastAPI()


# --- API Endpoint Definition ---
@app.post("/hackrx/run", response_model=HackRxOutput)
async def run_hackrx_task(payload: HackRxInput):
    """
    Accepts a list of questions, runs the agent team for each one,
    extracts the answer from the agent's final JSON output,
    and returns a single JSON response with all answers.
    """
    all_answers = []

    try:
        # Loop over each question provided in the request payload
        for question in payload.questions:
            # Run the agent team. This is a synchronous call that waits for the
            # entire multi-agent conversation to complete for the current question.
            final_state = team_1.run(task=question)
            
            last_message_content = None
            # The agent's final response is in the chat history. We search backwards
            # to find the last message from the 'ValidatorAgent'.
            if final_state and final_state.chat_history:
                for msg in reversed(final_state.chat_history):
                    if msg.get("name") == "ValidatorAgent" and "justification" in msg.get("content", ""):
                        last_message_content = msg.get("content")
                        break

            if last_message_content:
                try:
                    # Agents often wrap JSON in text or markdown. A regular expression
                    # is the most reliable way to extract the JSON block.
                    json_match = re.search(r'\{.*\}', last_message_content, re.DOTALL)
                    
                    if json_match:
                        json_string = json_match.group(0)
                        agent_output = json.loads(json_string)
                        # Your ValidatorAgent is prompted to put the answer in the
                        # 'justification' field. We extract it from the parsed JSON.
                        answer = agent_output.get("justification", "No justification found.")
                        all_answers.append(answer)
                    else:
                        all_answers.append("Could not extract valid JSON from the agent's response.")

                except json.JSONDecodeError:
                    # This handles cases where the extracted string is not valid JSON.
                    all_answers.append("Failed to decode the JSON from the agent's response.")
            else:
                # This is a fallback if the ValidatorAgent failed to produce a message.
                all_answers.append("The agent team did not provide a final structured answer for this question.")

            # IMPORTANT: Reset the team's state to ensure the context of the previous
            # question does not leak into the next one.
            team_1.reset()

        # After the loop, return the final Pydantic model, which FastAPI will serialize into JSON.
        return HackRxOutput(answers=all_answers)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # It is good practice to close long-lived connections.
        client.close()
        print("Task finished. Client connection has been closed.")