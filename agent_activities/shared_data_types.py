from dataclasses import dataclass

# Input types

# Result types


@dataclass
class ProgressUpdate:
    msg: str


@dataclass
class DocIngestResult:
    status: str

@dataclass
class QueryResult:
    response: str
    metadata: dict = None