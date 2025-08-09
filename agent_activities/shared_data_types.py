from dataclasses import dataclass

# For heatbeat messages and activity results

@dataclass
class ProgressUpdate:
    msg: str


@dataclass
class DocIngestResult:
    status: str

# For input data to activities

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

# For agent tasks and results

@dataclass
class AgentTask:
    query: str
    api_key: str

    def __repr__(self):
        query_preview = (self.query[:50] + '...') if len(self.query) > 50 else self.query
        return (
            f"query='{query_preview}', "  # Use truncated queries for cleaner logs
            f"api_key='****')"  # Avoid leaking API key
        )


@dataclass
class AgentResult:
    status: str
    result: str
