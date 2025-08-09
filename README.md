# Event-Driven-Agentic-RAG

Event-Driven Agentic Document Workflows using Temporal, Qdrant, Gemini. 

## Retrieval-Augmented Generation (RAG) on a Resume Document

### Resume Document Parsing
![Resume Parse](resume_parse_flow.png)

LLamaParse transforms the resume into a list of Document objects.

### Vector Store Index Creation

![Resume Indexing](index_resume_flow.png)

## Installation 
- This uses uv as the package manager. See [installation](https://docs.astral.sh/uv/getting-started/installation/) and [Creating Projects](https://docs.astral.sh/uv/concepts/projects/init/) for more info. 
- See [Temporal CLI inside Docker](https://docs.temporal.io/cli#installation) to have the CLI be accessible from the host system
- You'll need Qdrant locally. See [Local Quickstart](https://qdrant.tech/documentation/quickstart/). In-memory option only requires the qdrant_client (downloaded from `uv sync`).

## Running Project Locally
This depends on whether you want to use Cloud services (Qdrant and Temporal) or local instances. Qdrant also offers an in-memory option which doesn't require any service. All of these are configurable in the code. 

These steps assume a local Qdrant and local Temporal service instance: 
1. Run the Qdrant service (I'm using WSL): `docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant` in 1st terminal
2. Run TemporalIO Service `temporal server start-dev` in 2nd terminal
3. Run Worker `uv run worker.py` in 3rd terminal
4. Run Workflow `uv run client.py` in 4th terminal 

# Improvements
This is just a POC to see how these technologies can work togther. Some things to note:
1. Agentic capabilities. I used LlamaIndex's `FunctionTool` and `FunctionCallingAgent` for ease of use. But Temporal requires Workflow code to be deterministic (of which LLM calls are not). To solve this, you'd need to keep this usage only within Activities (built for non-deterministic behavior) or re-implement them to work with Temporal workflows (allows for more granularity).
2. Conversation History. Currently, this doesn't keep the converation history (durable or not). There should be some ways to use Llama's Memory [component](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/memory/) within an activity to make the conversation history durable. 