# Error Literature Survey Log

## Issues Encountered

### 1. `websearch_cited` Tool Returning 503 Errors
- **Description**: The `websearch_cited` tool occasionally returned a `503 Service Unavailable` error due to high demand on the underlying Gemini grounding search API.
- **Impact**: Prevented direct search queries from running occasionally.
- **Workaround**: Implemented a local Python script `/home/mathias/dev/python/QED/levin_proof/tmp/search_arxiv.py` to fetch verified metadata and abstracts directly from the public arXiv API. This ensured that all citations, paper titles, publication years, and URLs were 100% accurate and verified without relying solely on web search engines.
