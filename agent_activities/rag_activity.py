from dataclasses import dataclass
import os

import secret_config
from temporalio import activity
from agent_activities.shared_data_types import ProgressUpdate, DocIngestResult
# Llama imports
# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_parse import LlamaParse  # For document parsing
from llama_index.llms.google_genai import GoogleGenAI  # For LLM operations
from llama_index.core import StorageContext, load_index_from_storage  # For loading indexed data
from llama_index.core import VectorStoreIndex 
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding  # For embedding documents
# Vector Database
import qdrant_client  # For vector store indexing
from llama_index.vector_stores.qdrant import QdrantVectorStore


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


@dataclass
class IngestInput:
    """
    Input data for document ingestion activity.
    """
    document: str
    prompt_guidance: str


@dataclass
class QueryInput:
    """
    Input data for document query activity.
    """
    query: str
    api_key: str


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


async def create_vector_store_index(document_objects) -> None:
    """
    Indexes the parsed document data for retrieval.

    Reference(s):
        [1] https://docs.llamaindex.ai/en/stable/examples/vector_stores/QdrantIndexDemo/ 
    """
    activity.heartbeat(ProgressUpdate(msg="Indexing document..."))
    # Connect to existing Qdrant collection (if it exists) or create a new one
    vector_store = QdrantVectorStore(
        aclient=QDRANT_CLIENT,
        collection_name=COLLECTION_NAME,
    )
    # Create storage context with Qdrant vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Optionally, you can set the embedding model
    
    # Create the VectorStoreIndex from the parsed documents
    qdrant_index = VectorStoreIndex.from_documents(
        documents=document_objects,
        storage_context=storage_context,
        embed_model=EMBED_MODEL,  # Replace LlamaIndex OpenAIEmbedding default with GeminiEmbedding
        use_async=True,  # Build the VectorStoreIndex asynchronously
    )

    activity.heartbeat(ProgressUpdate(msg="Document indexing complete."))


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

    return DocIngestResult(status="Success")


@activity.defn
async def query_indexed_document(query_input: QueryInput) -> str:
    """
    Query the indexed document and return the response.

    This function uses the VectorStoreIndex to retrieve relevant information based on the query.

    Reference(s):
        [1] https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/
        [2] https://docs.llamaindex.ai/en/stable/understanding/querying/querying/
    """
    activity.heartbeat(ProgressUpdate(msg="Querying indexed document..."))
    query = query_input.query
    api_key = GEMINI_API_KEY  # query_input.api_key # TODO: Use passed-in api_key if deployed

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
    activity.heartbeat(ProgressUpdate(msg="Query complete."))

    return str(response)
