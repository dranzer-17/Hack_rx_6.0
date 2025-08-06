from config.Constanst import MODELS
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from typing import List
load_dotenv()

class Response_format(BaseModel):
    decision: str = Field(description = 'Accepted or Rejected')
    amount: int = Field(description = "Amount to be claimed")
    justification: str = Field(description = "The justification regarding the desicion")
    clauses_referenced: List = Field(description = "The clauses related to the acceptance or denial of the claim")

def get_model_client():
    """
    This function returns the model client for the agents and teams to Run.
    Note:
    ```python
    model_client,model_client_1,model_client_2,model_client_3 = get_model_client
    ```
    
    here the model_client_1 will be used for the validator
    """
    model_client = OpenAIChatCompletionClient(
    model = 'gemini-2.5-flash',
    api_key = os.getenv("GEMINI_API_KEY_1")
    )

    model_client_1 = OpenAIChatCompletionClient(
        model = 'gemini-2.5-flash',
        api_key = os.getenv("GEMINI_API_KEY"),
        response_format = Response_format
    )

    model_client_2 = OpenAIChatCompletionClient(
        model = 'gemini-2.5-flash',
        api_key = os.getenv("GEMINI_API_KEY_2")
    )

    open_router_api_key = os.getenv('OPENROUTER_API')

    model_client_3 =  OpenAIChatCompletionClient(
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-r1:free",
        api_key = open_router_api_key,
        model_info={
            "family":'deepseek',
            "vision" :True,
            "function_calling":True,
            "json_output": False
        }
    )
    
    return model_client,model_client_1,model_client_2,model_client_3