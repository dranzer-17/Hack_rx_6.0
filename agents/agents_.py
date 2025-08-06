from autogen_agentchat.agents import AssistantAgent
from config.prompt import query_enhancer_system_message,Retrieval_system_message,Validator_system_message
from config.tools import reteriever_tool

def get_query_enhancer_agent(model_client):
    query_enhancer_agent = AssistantAgent(
    name = 'QueryEnhancerAgent',
    model_client = model_client,
    description = "This agent is useful for enhancing the query",
    system_message = query_enhancer_system_message 
    )
    return query_enhancer_agent



def get_reterival_agent(model_client):
    reterival_agent = AssistantAgent(
        name = 'ReterivalAgent',
        model_client = model_client,
        description = "This agent is used to reterieve relevant document",
        system_message = Retrieval_system_message,
        tools = [reteriever_tool]
    )
    return reterival_agent

def get_validator_agent(model_client):

    Validator_agent = AssistantAgent(
        name = 'ValidatorAgent',
        model_client = model_client,
        description = "This agent is used for validating wether the query is related to the reterieved docs or not.",
        system_message = Validator_system_message
    )
    
    return Validator_agent