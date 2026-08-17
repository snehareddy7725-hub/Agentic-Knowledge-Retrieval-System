"""
Agentic RAG system setup.
Creates the agent with tools and workflow.
"""

import json
import os
from typing import List
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from app.config import LLM_MODEL, PARENT_STORE_PATH, SCORE_THRESHOLD
from app.semantic_cache import SemanticCache
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def setup_agent(vector_store):
    """
    Setup the agentic RAG system.
    
    Args:
        vector_store: Qdrant vector store instance
    
    Returns:
        tuple: (agent_graph, llm_with_tools)
    
    The agent workflow:
        1. User asks a question
        2. Agent decides which tools to use
        3. Agent searches documents
        4. Agent reads results
        5. Agent generates answer
    """
    
    # Create tools
    tools = create_tools(vector_store)

    # Semantic cache: reuses answers for previously-asked (or close
    # paraphrases of) questions, skipping the full search+LLM pipeline.
    cache = SemanticCache(cache_path="data/semantic_cache.json")

    # Initialize LLM (runs locally with Ollama)
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    def _extract_query_text(msg):
        """
        Safely extract plain question text regardless of how the
        message arrived: a LangChain message object (.content),
        a ("user", "text") tuple, or a raw string.
        """
        if hasattr(msg, "content"):
            return msg.content
        if isinstance(msg, tuple) and len(msg) == 2:
            return msg[1]
        return str(msg)

    def _get_original_question(state):
        """Finds the first human message in this turn — the original question."""
        for msg in state["messages"]:
            if hasattr(msg, "type") and msg.type == "human":
                return msg.content
            if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "user":
                return msg[1]
        return None

    def cache_check_node(state):
        """
        Checks the semantic cache before doing any real work. If a
        sufficiently similar question was answered before, returns
        that cached answer immediately — skipping the guardrail,
        search, and LLM call entirely.
        """
        query = _extract_query_text(state["messages"][-1])
        query_embedding = vector_store.embeddings.embed_query(query)
        hit = cache.get(query_embedding)

        if hit:
            cached_msg = AIMessage(
                content=hit["answer"],
                additional_kwargs={"cache_hit": True, "sources": hit["sources"]}
            )
            return {"messages": [cached_msg]}

        return {"messages": []}

    def route_after_cache(state):
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.additional_kwargs.get("cache_hit"):
            return END
        return "guardrail"

    def guardrail_node(state):
        """
        Runs a preliminary relevance check before the agent engages.
        If nothing in the vector store clears the threshold for the
        user's question, short-circuits with a decline message.
        """
        last_user_msg = state["messages"][-1]
        query = _extract_query_text(last_user_msg)

        results = vector_store.similarity_search(query, k=3, score_threshold=SCORE_THRESHOLD)

        if not results:
            decline_msg = AIMessage(
                content="I couldn't find relevant information in the documents to answer this question.",
                additional_kwargs={"guardrail_declined": True}
            )
            return {"messages": [decline_msg]}

        # Relevant content exists — let it pass through unchanged
        return {"messages": []}
    
    def agent_node(state):
        """
        Process user query and decide which tools to use.
        
        The agent:
        1. Takes the user's question
        2. Decides if it needs to search
        3. Chooses the right tools
        4. Calls them with the right parameters
        """
        sys_msg = SystemMessage(content="""
        You are a helpful assistant that answers questions based on documents.
        
        RULES:
        - ALWAYS search the documents BEFORE answering
        - Use search_child_chunks to find relevant information
        - Use retrieve_parent_chunks to get full context when needed
        - Answer based ONLY on retrieved information
        - If you can't find information, say so
        - List the source documents at the end of your answer
        
        WORKFLOW:
        1. Search for relevant information
        2. Review what you found
        3. If needed, get more context
        4. Answer the question
        """)
        
        # Get the user's question
        if not state.get("messages"):
            return {"messages": []}
        
        # Process with the LLM
        response = llm_with_tools.invoke([sys_msg] + state["messages"])
        return {"messages": [response]}

    def route_after_guardrail(state):
        last_msg = state["messages"][-1]
        # If the guardrail flagged a decline, stop here instead of engaging the agent
        if isinstance(last_msg, AIMessage) and last_msg.additional_kwargs.get("guardrail_declined"):
            return END
        return "agent"

    def cache_store_node(state):
        """
        Runs after the agent has produced a final answer (no more tool
        calls pending). Stores the question + answer in the semantic
        cache for future reuse. Guardrail declines never reach this
        node, so declines are never cached.
        """
        last_msg = state["messages"][-1]
        original_question = _get_original_question(state)

        if original_question and hasattr(last_msg, "content") and last_msg.content:
            query_embedding = vector_store.embeddings.embed_query(original_question)
            cache.add(original_question, query_embedding, last_msg.content)

        return {"messages": []}
    # Build the agent graph
    # This creates the workflow: cache_check -> guardrail -> agent -> tools -> agent -> ... -> cache_store -> answer
    agent_builder = StateGraph(AgentState)

    agent_builder.add_node("cache_check", cache_check_node)
    agent_builder.add_node("guardrail", guardrail_node)
    agent_builder.add_node("agent", agent_node)
    agent_builder.add_node("tools", ToolNode(tools))
    agent_builder.add_node("cache_store", cache_store_node)

    agent_builder.add_conditional_edges(
        "cache_check",
        route_after_cache,
        {"guardrail": "guardrail", END: END}
    )

    agent_builder.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"agent": "agent", END: END}
    )

    agent_builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: "cache_store"}
    )
    agent_builder.add_edge("tools", "agent")
    agent_builder.add_edge("cache_store", END)

    agent_builder.set_entry_point("cache_check")   # <-- entry point is now cache_check

    agent_graph = agent_builder.compile()
    print("✅ Agent ready!")
    return agent_graph, llm_with_tools

def create_tools(vector_store):
    """
    Create tools for the agent.
    
    Args:
        vector_store: Qdrant vector store instance
    
    Returns:
        list: List of tools
    """
    
    @tool
    def search_child_chunks(query: str, k: int = 5) -> List[dict]:
        """
        Search for the top K most relevant child chunks.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of dictionaries with content and metadata
        
        This is the primary search tool the agent uses.
        It finds relevant snippets from the vector database.
        """
        # Enforce a minimum k regardless of what the LLM requests, so
        # an overly narrow request (e.g. k=1) can't cause the correct
        # chunk to be missed if it isn't the single top-ranked result.
        MIN_K = 3
        effective_k = max(k, MIN_K)

        try:
            results = vector_store.similarity_search(query, k=effective_k, score_threshold=SCORE_THRESHOLD)
            return [
                {
                    "content": doc.page_content,
                    "parent_id": doc.metadata.get("parent_id", ""),
                    "source": doc.metadata.get("source", "")
                }
                for doc in results
            ]
        except Exception as e:
            print(f"Error searching child chunks: {e}")
            return []
    
    @tool
    def retrieve_parent_chunks(parent_ids: List[str]) -> List[dict]:
        """
        Retrieve full parent chunks by their IDs.
        
        Args:
            parent_ids: List of parent chunk IDs to retrieve
            
        Returns:
            List of dictionaries with full parent content
        
        This tool gets the full context around a search result.
        Parent chunks contain the complete section of the document.
        """
        unique_ids = sorted(list(set(parent_ids)))
        results = []
        
        for parent_id in unique_ids:
            # Add .json if missing
            file_path = os.path.join(
                PARENT_STORE_PATH, 
                parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json"
            )
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        doc_dict = json.load(f)
                        results.append({
                            "content": doc_dict["page_content"],
                            "parent_id": parent_id,
                            "metadata": doc_dict["metadata"]
                        })
                except Exception as e:
                    print(f"Error loading parent chunk {parent_id}: {e}")
        
        return results
    
    return [search_child_chunks, retrieve_parent_chunks]

def get_agent_response(agent_graph, question, thread_id="default"):
    """
    Get a response from the agent.
    
    Args:
        agent_graph: The compiled agent graph
        question: User's question
        thread_id: Conversation thread ID
    
    Returns:
        str: Agent's response
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        result = agent_graph.invoke(
            {"messages": [("user", question)]},
            config=config
        )
        
        # Extract the final answer
        if result and result.get("messages"):
            # Get the last message (assistant's response)
            last_message = result["messages"][-1]
            if hasattr(last_message, 'content'):
                return last_message.content
            else:
                return str(last_message)
        
        return "Sorry, I couldn't generate a response."
    except Exception as e:
        return f"Error: {str(e)}"
