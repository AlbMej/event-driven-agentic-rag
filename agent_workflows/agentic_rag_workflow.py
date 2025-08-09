# Workflows define a sequence of steps. A Conductor/Orchestrator of your business logic (Activities)
# See: https://docs.temporal.io/workflows

from datetime import timedelta
from temporalio import workflow

# The `with` block is a best practice to avoid accidentally using non-deterministic code in the workflow
with workflow.unsafe.imports_passed_through():
    from agent_activities.shared_data_types import DocIngestResult
    from agent_activities.agentic_rag_activity import (
        ingest_and_index_document,
        llama_index_agent,
        IngestInput, QueryInput,
        AgentTask, AgentResult,
    )


@workflow.defn
class AgenticRAGWorkflow:
    """
    This Temporal Workflow orchestrates an Agentic RAG process.
    It ingests a document, indexes it, and allows querying the indexed data.
    """

    @workflow.run
    async def run(self, ingest_input: IngestInput, agent_task: AgentTask) -> str:
        # Step 1: Ingest and index the document
        ingestion_result: DocIngestResult = await workflow.execute_activity(
            ingest_and_index_document,
            args=[ingest_input],
            start_to_close_timeout=timedelta(minutes=5),
        )
        workflow.logger.info(f"Step 1 ingest completed: {ingestion_result}")

        # Step 2: Run the Agent (can query the indexed document)
        agent_result: AgentResult = await workflow.execute_activity(
            llama_index_agent,
            args=[agent_task],
            start_to_close_timeout=timedelta(minutes=2),
        )

        workflow.logger.info(f"Step 2 Agent Returned: {agent_result}")

        return f"Final Outcome, {agent_result.result}"
