import traceback
import secret_config
from temporalio import activity
from agent_activities.shared_data_types import (
    ProgressUpdate, DocIngestResult,
    IngestInput, QueryInput,
    AgentTask, AgentResult
)
# Llama imports
# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_parse import LlamaParse  # For document parsing
from llama_index.llms.google_genai import GoogleGenAI  # For LLM operations
from llama_index.core import StorageContext  #
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding  # For embedding documents
# Vector Database
import qdrant_client  # For vector store indexing
from llama_index.vector_stores.qdrant import QdrantVectorStore
# Agentic imports
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent  


# --- Google Gemini Configuration ---
GEMINI_API_KEY = secret_config.GEMINI_API_KEY
GEMINI_MODEL = "models/gemini-1.5-flash-8b"
EMBEDDING_MODEL_TYPE = "models/embedding-001"
EMBED_MODEL = GoogleGenAIEmbedding(model=EMBEDDING_MODEL_TYPE, api_key=GEMINI_API_KEY)
# Can also use FastEmbedding: https://docs.llamaindex.ai/en/stable/examples/vector_stores/QdrantIndexDemo/

# --- Qdrant Configuration ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "resume_collection"

# Initialize Qdrant client (See: https://qdrant.tech/documentation/database-tutorials/async-api/)
QDRANT_CLIENT = qdrant_client.AsyncQdrantClient(
    # Use in-memory Qdrant for fast and light-weight experiments
    # location=":memory:",
    # If using a Qdrant instance (local or remote) use:
    url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
    # If using Qdrant Cloud, set your Qdrant API KEY
    # api_key="<qdrant-api-key>"
)


async def parse_document(document: str, prompt_guidance: str) -> list:
    """
    Parses a document and returns the parsed data.

    This function uses LlamaParse to transform the document into a structured format.
    """
    activity.heartbeat(ProgressUpdate(msg="Parsing document..."))

    # Parse the document using LlamaParse
    llama_parser = LlamaParse(
        api_key=secret_config.LLAMA_API_KEY,
        result_type="markdown",
        user_prompt=prompt_guidance,  # content_guideline_instruction= is deprecated 
    )

    document_objects = await llama_parser.aload_data(document)

    activity.heartbeat(ProgressUpdate(msg="Document parsing complete."))
    return document_objects


# --- RAG Activities ---
async def create_vector_store_index(document_objects) -> None:
    """
    Indexes the parsed document data for retrieval.

    Reference(s):
        [1] https://docs.llamaindex.ai/en/stable/examples/vector_stores/QdrantIndexDemo/ 
    """
    activity.logger.info(ProgressUpdate(msg="Indexing document..."))

    # Connect to existing Qdrant collection (if it exists) or create a new one
    vector_store = QdrantVectorStore(
        aclient=QDRANT_CLIENT,
        collection_name=COLLECTION_NAME,
    )
    # Create storage context with Qdrant vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # Create the VectorStoreIndex from the parsed documents
    qdrant_index = VectorStoreIndex.from_documents(
        documents=document_objects,
        storage_context=storage_context,
        embed_model=EMBED_MODEL,  # Replace LlamaIndex OpenAIEmbedding default with GeminiEmbedding
        use_async=True,  # Build the VectorStoreIndex asynchronously
    )
    activity.logger.info(ProgressUpdate(msg="Document indexing complete."))

@activity.defn
async def ingest_and_index_document(ingest_input: IngestInput) -> DocIngestResult:
    """
    Ingest and parse a document, then index it for retrieval.

    LLamaParse transforms the resume into a list of Document objects.
    A Document object stores text along with some other attributes:
        - metadata: a dictionary of annotations that can be appended to the text.
        - relationships: a dictionary containing relationships to other Documents.
    """
    file_path = ingest_input.document
    parse_prompt = ingest_input.prompt_guidance
    # Parse the document using LlamaParse
    parsed_data = await parse_document(file_path, parse_prompt)
    # Feed Document objects to VectorStoreIndex
    await create_vector_store_index(parsed_data)
    # TODO: Handle exceptions
    return DocIngestResult(status="Success")


@activity.defn
async def llama_index_agent(task: AgentTask) -> AgentResult:
    """
    LlamaIndex agent for processing tasks.
    """
    activity.logger.info(f"Executing agent task: {task}")

    try:
        api_key = task.api_key

        
        # Define tool functions for the agent
        async def query_indexed_document(query: str) -> str:
            """
            Use this tool to answer questions about the person whose resume has been provided.
            This tool will query the indexed document to retrieve relevant information based on the query.
            """
            activity.logger.info(ProgressUpdate(msg="[Agent Tool] Querying indexed document..."))
            # Connect to existing Qdrant collection (if it exists) or create a new one
            vector_store = QdrantVectorStore(
                aclient=QDRANT_CLIENT,
                collection_name=COLLECTION_NAME,
            )
            # Load the index directly from VectorStore
            qdrant_index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=EMBED_MODEL,
                use_async=True,
            )
            # LLM model for generating query response
            llm = GoogleGenAI(model=GEMINI_MODEL, api_key=api_key)
            query_engine = qdrant_index.as_query_engine(llm=llm, similarity_top_k=5)  # Use top 5 most similar vectors
            # Perform the query
            response = await query_engine.aquery(query)  # Create question vector, perform similarity search
            activity.logger.info(ProgressUpdate(msg="[Agent Tool] Query complete."))
            # TODO: try and except
            return str(response)

        async def general_knowledge_query(query: str) -> str:
            """Answer general questions."""
            activity.logger.info(ProgressUpdate(msg="[Agent Tool] Querying general knowledge..."))
            llm = GoogleGenAI(model=GEMINI_MODEL, api_key=api_key)
            response = await llm.complete(f"Answer this question: {query}")
            return response.text

        # Create tools
        resume_tool = FunctionTool.from_defaults(fn=query_indexed_document)
        general_tool = FunctionTool.from_defaults(fn=general_knowledge_query)

        # Create agent
        llm = GoogleGenAI(model="models/gemini-1.5-flash-8b", api_key=api_key)
        agent = FunctionAgent(
            name="Agent",
            tools=[resume_tool, general_tool],
            llm=llm,
            verbose=True
        )

        # Agent reasoning (non-deterministic)
        response = await agent.run(task.query)

        return AgentResult(
            status="Success",
            result=str(response)
        )

    except Exception as e:
        detailed_error = traceback.format_exc()
        agent_result = AgentResult(
            status="Failed",
            result=f"Agent failed: {detailed_error}"
        )
        activity.logger.error(f"{agent_result.result}")
        raise  # Fail workflow execution
