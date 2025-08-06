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
from llama_index.core import VectorStoreIndex  # For vector store indexing
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding  # For embedding documents
# Vector Database
# import qdrant_client


GEMINI_API_KEY = secret_config.GEMINI_API_KEY
GEMINI_MODEL = "models/gemini-1.5-flash-8b"
EMBEDDING_MODEL_TYPE = "models/embedding-001"
EMBED_MODEL = GoogleGenAIEmbedding(model=EMBEDDING_MODEL_TYPE, api_key=GEMINI_API_KEY)
STORAGE_DIR = "./storage"


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


async def create_llama_vector_store_index(document_objects) -> None:
    """
    Indexes the parsed document data for retrieval.
    """
    activity.heartbeat(ProgressUpdate(msg="Indexing document..."))

    # Check if the index is stored on disk TODO: implement more robust check
    if os.path.exists(STORAGE_DIR):  # Simply see if directory already created
        # Document index already exists
        return
    
    # Create a VectorStoreIndex from the parsed document objects
    vector_store = VectorStoreIndex.from_documents(
        documents=document_objects,
        embed_model=EMBED_MODEL
    )

    # Store index to disk for persistence (in production, use a hosted vector store)
    vector_store.storage_context.persist(persist_dir=STORAGE_DIR)

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
    await create_llama_vector_store_index(parsed_data)

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

    # Using LLama VectorStoreIndex for querying
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(storage_context, embed_model=EMBED_MODEL)

    # LLM model for querying
    llm = GoogleGenAI(model=GEMINI_MODEL, api_key=api_key)  # TODO: use passed-in api_key if deployed
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)

    # Perform the query
    response = await query_engine.aquery(query)
    activity.heartbeat(ProgressUpdate(msg="Query complete."))

    return str(response)
