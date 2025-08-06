from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from config.Constanst import STOP_WORD
from models.model_client import get_model_client
from agents.agents_ import get_query_enhancer_agent,get_reterival_agent,get_validator_agent
model_client,model_client_1,model_client_2,model_client_3 = get_model_client()










def get_team():

    query_enhancer_agent = get_query_enhancer_agent(model_client = model_client)
    reterival_agent = get_reterival_agent(model_client=model_client_2)
    Validator_agent = get_validator_agent(model_client=model_client_1)
    
    termination_condition = TextMentionTermination(STOP_WORD)
    team_1 = RoundRobinGroupChat(name="RAGAgentAndValidatorAgent",
                            description = "This agent is used to enhance the user query than reterieve documents and also to validate the documents",
                            participants = [query_enhancer_agent,reterival_agent,Validator_agent],
                            termination_condition = termination_condition,
                            max_turns= 6
                            )
    
    return team_1