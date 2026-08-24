# API package
from src.api.admin import router as admin_router
from src.api.anomaly import router as anomaly_router
from src.api.auth import router as auth_router
from src.api.auto_tag import router as auto_tag_router
from src.api.buddy import router as buddy_router
from src.api.budget import router as budget_router
from src.api.chat import router as chat_router
from src.api.cognitive_state import router as cognitive_state_router
from src.api.connections import router as connections_router
from src.api.conversations import router as conversations_router
from src.api.entities import router as entities_router
from src.api.git_sync import router as git_sync_router
from src.api.graph import router as graph_router
from src.api.import_export import router as import_export_router
from src.api.notes import router as notes_router
from src.api.plugins import router as plugin_router
from src.api.question_gen import router as question_gen_router
from src.api.reembed import router as reembed_router
from src.api.retrieval import router as retrieval_router
from src.api.review import router as review_router
from src.api.schema import router as schema_router
from src.api.search import router as search_router
from src.api.serendipity import router as serendipity_router
from src.api.settings import router as settings_router
from src.api.setup import router as setup_router
from src.api.stats import router as stats_router
from src.api.summarization import router as summarization_router
from src.api.tokens import router as tokens_router
from src.api.websocket import router as websocket_router

__all__ = [
    "admin_router",
    "anomaly_router",
    "auth_router",
    "auto_tag_router",
    "buddy_router",
    "budget_router",
    "chat_router",
    "cognitive_state_router",
    "connections_router",
    "conversations_router",
    "entities_router",
    "git_sync_router",
    "graph_router",
    "import_export_router",
    "notes_router",
    "plugin_router",
    "question_gen_router",
    "reembed_router",
    "retrieval_router",
    "review_router",
    "schema_router",
    "search_router",
    "serendipity_router",
    "settings_router",
    "setup_router",
    "stats_router",
    "summarization_router",
    "tokens_router",
    "websocket_router",
]
