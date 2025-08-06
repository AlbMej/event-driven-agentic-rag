# Workflows define a sequence of steps. A Conductor/Orchestrator of your business logic (Activities)
# See: https://docs.temporal.io/workflows

from datetime import timedelta
from temporalio import workflow

# The `with` block is a best practice to avoid accidentally using non-deterministic code in the workflow
with workflow.unsafe.imports_passed_through():
    from agent_activities.shared_data_types import DocIngestResult
    from agent_activities.rag_activity import (
        ingest_and_index_document,
        query_indexed_document,
        IngestInput,
        QueryInput,
    )


@workflow.defn
class RAGWorkflow:
    """
    This Temporal Workflow orchestrates the RAG process.
    It ingests a document, indexes it, and allows querying the indexed data.
    """

    @workflow.run
    async def run(self, ingest_input: IngestInput, query_input: QueryInput) -> str:
        # Step 1: Ingest and index the document
        ingestion_result: DocIngestResult = await workflow.execute_activity(
            ingest_and_index_document,
            args=[ingest_input],
            start_to_close_timeout=timedelta(minutes=5),
        )
        workflow.logger.info(f"Step 1 ingest completed: {ingestion_result}")

        # Step 2: Query the indexed document
        query_result = await workflow.execute_activity(
            query_indexed_document,
            args=[query_input],
            start_to_close_timeout=timedelta(minutes=2),
        )
        workflow.logger.info(f"Step 2 query completed: {query_result}")

        return f"Outcome: {query_result}"
