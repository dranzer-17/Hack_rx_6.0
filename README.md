-----

# 🤖 ClaimAGI: The Autonomous Insurance Inquisition ⚖️

**They ask. We dissect. It decides. Welcome to the future of claims processing.**

Tired of sifting through endless policy documents? Buried under ambiguous client queries? Fear not\! ClaimAGI is a multi-agent system that unleashes a team of specialized AI agents to autonomously analyze, investigate, and adjudicate insurance claims with ruthless efficiency.

> This isn't just automation. This is an AI-powered tribunal in a box.

-----

## 🎭 The Cast of Agents: The Three Who Judge

Our system is built on a team of highly-specialized agents, each with a unique personality and a singular purpose. They work in a strict, round-robin sequence.

### 1\. 🧐 The Scribe (`QueryEnhancerAgent`)

The first point of contact. This agent takes messy, jumbled, and often incomplete human queries and forges them into a single, crystal-clear, professional request. It extracts key entities like age, location, and procedure without inventing a single detail.

  * **Motto**: "From chaos, clarity."
  * **Input**: *"46-year-old male, knee surgery in Pune, 3-month-old insurance policy"*
  * **Output**: *"Check whether a 46-year-old male with a 3-month-old insurance policy is eligible for coverage of knee surgery in Pune."*

### 2\. 📚 The Librarian (`ReterivalAgent`)

Armed with the Scribe's clear query, the Librarian dives into the digital catacombs of your document database (our Weaviate Vector Store). Its sole mission is to retrieve every relevant clause and piece of evidence using a powerful semantic search tool (`retriever_tool`). It doesn't think or reason; it just *finds*. The retrieved data is meticulously organized under thematic headings for the final act.

  * **Motto**: "The truth is in here. I will find it."
  * **Tool of Choice**: Maximum Marginal Relevance (MMR) search to find the top 50 most relevant *and* diverse documents.

### 3\. ⚖️ The Judge (`ValidatorAgent`)

The final authority. The Judge receives the enhanced query and the evidence compiled by the Librarian. It operates in two phases:

1.  **Validation**: First, it scrutinizes the evidence. Is it relevant? Is it complete? If not, it can send the case back for more digging.
2.  **Decision**: If the evidence is sufficient, the Judge interprets the policy clauses like a seasoned legal advisor. It then passes the final, unassailable verdict.

The Judge's decision is delivered in a precise JSON format, citing the exact clauses that sealed the claimant's fate. When its work is done, it declares "**STOP**," bringing the inquisition to a close.

  * **Motto**: "In clauses we trust."
  * **Final Output**:
    ```json
    {
      "decision": "Rejected",
      "amount": 0,
      "justification": "Cataract surgery is covered, but Clause 5.3 states a 2-year waiting period applies. The policy is only 2 months old.",
      "clauses_referenced": [
        "Clause 5.3: All cataract and other listed surgeries are covered after a minimum waiting period of 24 months."
      ]
    }
    ```

-----

## 🛠️ The Arsenal: Technology Behind the Inquisition

This system is powered by a stack of cutting-edge AI and data technologies.

  * **The AI Minds**: Google's `gemini-2.5-flash` and `deepseek/deepseek-r1` via OpenRouter provide the cognitive power for our agents.
  * **The Framework**: `autogen_agentchat` is the puppet master, orchestrating the `RoundRobinGroupChat` between the agents.
  * **The Infinite Library**: Weaviate Cloud serves as our vector database, storing and indexing policy documents for lightning-fast semantic retrieval.
  * **The Universal Translator**: Hugging Face's `BAAI/bge-large-en-v1.5` model converts text into high-quality vector embeddings.
  * **The Enforcer**: Pydantic models ensure the ValidatorAgent's final output is always in the correct JSON format.

-----

## 🚀 The Grand Workflow: How a Claim Meets Its Fate

1.  A user submits a query to the `RoundRobinGroupChat` team.
2.  **Turn 1**: The `QueryEnhancerAgent` receives the query, refines it, and passes its clear, enhanced question back to the group.
3.  **Turn 2**: The `ReterivalAgent` takes the enhanced query, activates its `reteriever_tool`, and pulls all relevant clauses from the Weaviate database. It presents its findings, neatly formatted.
4.  **Turn 3**: The `ValidatorAgent` reviews the query and the evidence. It analyzes the clauses, makes a decision (`Approved`, `Rejected`, or `Partially Approved`), and generates the final JSON report.
5.  **Termination**: The `ValidatorAgent` outputs the word "**STOP**". The `TextMentionTermination` condition detects this keyword and gracefully ends the process. The final verdict is served.






For Runing set your keys in .env file 

run : python main.py
