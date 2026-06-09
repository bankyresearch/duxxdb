"""DuxxDB integrations: a RESP client plus LangChain and LlamaIndex adapters.

The base client has no heavy deps (only ``redis-py``); the framework adapters
are imported from submodules so their dependencies stay optional:

    from duxxdb_integrations import DuxxDBClient
    from duxxdb_integrations.langchain import DuxxDBVectorStore, DuxxDBChatMessageHistory
    from duxxdb_integrations.llama_index import DuxxDBVectorStore as LlamaDuxxStore
"""

from .client import DuxxDBClient, Recall

__all__ = ["DuxxDBClient", "Recall"]
__version__ = "0.1.0"
