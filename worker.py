# Workers listen for tasks from the Temporal server and execute workflows and activities.
# See: https://docs.temporal.io/workers

# from concurrent.futures import ThreadPoolExecutor
import asyncio
import logging
from temporalio.client import Client
from temporalio.worker import Worker
from agent_activities.rag_activity import ingest_and_index_document, query_indexed_document
from agent_workflows.rag_workflow import RAGWorkflow


async def main():
    """
    "async with" + "await asyncio.Future()" recommended over "worker=Worker()" + "await worker.run()"
    This is because "async with" runs a worker with guaranteed cleanup on shutdown
    """
    # Uncomment the line below to see logging
    logging.basicConfig(level=logging.INFO)

    # Create a Temporal client to connect to local Temporal server
    client = await Client.connect("localhost:7233")

    # Create ONE shared thread pool for all workers in this process (for non-async activities)
    # shared_executor = ThreadPoolExecutor(max_workers=5)

    # Run worker (listens for tasks in the "greeting" task queue)
    async with Worker(  # https://python.temporal.io/temporalio.worker.Worker.html
        client=client,  # Allows the worker to reach out and say "I'm here, Temporal Server, give me work!"
        task_queue="rag_doc_ingest",  # Tells Temporal Server, "I am only set up to process tasks from this queue"
        workflows=[RAGWorkflow],  # List of class references, written specifically to handle running activities
        activities=[ingest_and_index_document, query_indexed_document],  # Process the tasks on the task_queue
        # activity_executor=shared_executor,  # Non-async activities require an executor
    ):
        print("Worker started. Listening for tasks...")
        await asyncio.Future()  # Keeps the worker alive so Temporal Server can send tasks to it

    # Note: In a production setup, the worker must always be available (run as a managed service)

if __name__ == "__main__":
    # Run the worker using asyncio event loop
    asyncio.run(main()) 