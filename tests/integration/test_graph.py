import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_openai():
    """Mock OpenAI to avoid API calls during tests."""
    with patch('src.models.openai_model.AsyncOpenAI') as mock:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="SEC_FILING"))],
            model="gpt-4"
        ))
        mock.return_value = client
        yield mock


@pytest.fixture
def mock_vector_store():
    """Mock vector store to avoid disk/network access."""
    from src.data.chunking import Chunk
    
    store = MagicMock()
    store.search.return_value = [
        (Chunk(
            text="Apple reported revenue of $383 billion.",
            chunk_id="chunk_1",
            document_id="AAPL-10K-2023",
            metadata={"ticker": "AAPL"}
        ), 0.9)
    ]
    return store


@pytest.mark.asyncio
async def test_graph_creation(mock_openai):
    """Test that the graph can be created."""
    from src.orchestration.graph import create_graph
    
    graph = create_graph()
    assert graph is not None


@pytest.mark.asyncio
async def test_graph_routes_sec_query(mock_openai, mock_vector_store):
    """Test that SEC query is routed to SEC agent."""
    from src.orchestration.graph import get_graph
    from src.orchestration.nodes import classify_intent
    
    # Create initial state
    state = {
        "query": "What are Apple's risk factors?",
        "conversation_id": "test-123",
        "entities": {"ticker": "AAPL"},
        "sec_results": [],
        "openbb_results": [],
        "fred_results": [],
        "iteration_count": 0,
    }
    
    # Test intent classification
    new_state = await classify_intent(state)
    assert new_state["intent"] == "sec_filing"


@pytest.mark.asyncio
async def test_graph_routes_financial_query(mock_openai):
    """Test that financial query is routed to OpenBB agent."""
    # Mock to return FINANCIALS
    with patch('src.models.openai_model.AsyncOpenAI') as mock:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="FINANCIALS"))],
            model="gpt-4"
        ))
        mock.return_value = client
        
        from src.orchestration.nodes import classify_intent
        
        state = {
            "query": "What is Apple's current stock price?",
            "conversation_id": "test-123",
            "entities": {"ticker": "AAPL"},
            "sec_results": [],
            "openbb_results": [],
            "fred_results": [],
            "iteration_count": 0,
        }
        
        new_state = await classify_intent(state)
        assert new_state["intent"] == "financials"


@pytest.mark.asyncio
async def test_graph_routes_macro_query(mock_openai):
    """Test that macro query is routed to FRED agent."""
    with patch('src.models.openai_model.AsyncOpenAI') as mock:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="MACRO"))],
            model="gpt-4"
        ))
        mock.return_value = client
        
        from src.orchestration.nodes import classify_intent
        
        state = {
            "query": "What is the current GDP growth rate?",
            "conversation_id": "test-123",
            "entities": {},
            "sec_results": [],
            "openbb_results": [],
            "fred_results": [],
            "iteration_count": 0,
        }
        
        new_state = await classify_intent(state)
        assert new_state["intent"] == "macro"


@pytest.mark.asyncio
async def test_route_by_intent():
    """Test intent routing logic."""
    from src.orchestration.nodes import route_by_intent
    from src.config.constants import Intent
    
    # Test SEC routing
    assert route_by_intent({"intent": Intent.SEC_FILING}) == "sec_agent"
    
    # Test financial routing
    assert route_by_intent({"intent": Intent.FINANCIALS}) == "openbb_agent"
    
    # Test macro routing
    assert route_by_intent({"intent": Intent.MACRO}) == "fred_agent"
    
    # Test default routing
    assert route_by_intent({}) == "sec_agent"
