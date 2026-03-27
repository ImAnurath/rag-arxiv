from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str         # user query
    top_k: int = 5     # number of chunks to pass to LLM


class SourceInfo(BaseModel):
    # info about a source paper, to be included in the response for provenance.
    arxiv_id: str
    title: str
    authors: list[str]


class QueryResponse(BaseModel):
    '''
    The response returned by the /query endpoint.
    '''
    answer: str
    sources: list[SourceInfo] # provenance info for each source paper
    retrieved_chunks: int # number of chunks retrieved from the database
    usage: dict # usage info from the LLM, e.g. token counts