# `client/run_workflow.py` uses a Temporal client to start the workflow execution
# See: https://docs.temporal.io/develop/python/temporal-clients

import asyncio
import logging
from temporalio.client import Client
from agent_workflows.rag_workflow import RAGWorkflow
from agent_activities.rag_activity import IngestInput, QueryInput

interrupt_event = asyncio.Event()  # Used to stop the client gracefully TODO: Use


async def main():
    logging.basicConfig(level=logging.INFO)

    # Start client
    client = await Client.connect("localhost:7233", namespace="default")  # Connect to the Temporal server

    # --- Get user input ---
    # Prompt user for their api key
    print("What's your api key for Gemini?")
    usr_api_key = "grabbed from secret_config" # input("Enter api key: ")
    # Prompt user for document path
    print("Where is the document located?")
    doc_path = "./Alberto_Mejia_Resume.pdf" # input("Enter document path: ")
    # Prompt user for their query
    print("What's your query about this document?")
    usr_query = "Who is this person and what do they do?" # input("Enter query: ")
    print("-" * 25)
    print("Starting workflow from user inputs...")
    # --- Create inputs for RAG ---
    ingest_data = IngestInput(
        document=doc_path,  # Path to the document to be ingested
        prompt_guidance="Extract key information and present it in a structured markdown format."
    )
    query_data = QueryInput(
        query=usr_query,  # Query to ask about the ingested document
        api_key=usr_api_key  # API key for Gemini (or other LLM service)
    )

    # While a worker is running, use the client to run the workflow and wait for its result
    handle = await client.start_workflow(  # https://python.temporal.io/temporalio.client.Client.html#start_workflow
        workflow=RAGWorkflow.run,  # Name of the workflow to run
        args=[ingest_data, query_data],  # Args passed to the workflow
        id="rag-workflow-id",  # Workflow ID must be unique (can be reused if completed)
        task_queue="rag_doc_ingest",  # Task queue to use for the workflow
    )
    # https://community.temporal.io/t/java-sdk-any-guidelines-on-when-to-use-workflowclient-start-vs-workflowclient-execute-to-invoke-the-async-workflow-execution/10466/2

    final_result = await handle.result()  # Wait for the workflow to complete and get the result
    print("-" * 25)
    print(f"<Workflow completed> {final_result}")  # Print the result of the workflow execution

if __name__ == "__main__":
    # Run the client using asyncio event loop
    asyncio.run(main())
