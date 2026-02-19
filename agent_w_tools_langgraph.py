"""
Simple LangGraph agent that uses the research tools directly.
This version imports the functions from research_mcp_server.py directly
instead of using the MCP protocol.
"""

import os
from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Import the research functions directly
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from research_mcp_server import search_papers, extract_info


# Define tools using LangChain's @tool decorator
@tool
def search_research_papers(topic: str, max_results: int = 5) -> list[str]:
    """
    Search for papers on arXiv based on a topic and store their information.
    
    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)
        
    Returns:
        List of paper IDs found in the search
    """
    return search_papers(topic, max_results)


@tool
def get_paper_info(paper_id: str) -> str:
    """
    Get detailed information about a specific paper using its ID.
    
    Args:
        paper_id: The ID of the paper to look for
        
    Returns:
        JSON string with paper information if found, error message if not found
    """
    return extract_info(paper_id)


# Define the agent state
class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[list[BaseMessage], add_messages]


# Create the LangGraph agent
class ResearchAgent:
    """A simple research agent using LangGraph."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", base_url: str = "http://localhost:1234/v1"):
        """
        Initialize the research agent.
        
        Args:
            model_name: Name of the model to use (for LMStudio, this can be anything)
            base_url: Base URL for the LLM API (LMStudio default)
        """
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key="lm-studio",  # LMStudio doesn't need a real API key
            temperature=0.7,
        )
        
        # Define the tools
        self.tools = [search_research_papers, get_paper_info]
        
        # Bind tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Create the graph
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""
        # Create graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Add edges
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            }
        )
        workflow.add_edge("tools", "agent")
        
        # Compile the graph
        return workflow.compile()
    
    def _call_model(self, state: AgentState) -> dict:
        """Call the LLM with the current state."""
        messages = state["messages"]
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """Determine whether to continue or end the workflow."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If there are tool calls, continue
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        
        # Otherwise, end
        return "end"
    
    def run(self, message: str) -> str:
        """
        Run the agent with a message.
        
        Args:
            message: The user's message
            
        Returns:
            The agent's response
        """
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=message)]
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        # Extract the final response
        final_message = result["messages"][-1]
        return final_message.content
    
    def stream(self, message: str):
        """
        Stream the agent's response.
        
        Args:
            message: The user's message
            
        Yields:
            Updates from the agent
        """
        initial_state = {
            "messages": [HumanMessage(content=message)]
        }
        
        for chunk in self.graph.stream(initial_state, stream_mode="values"):
            if "messages" in chunk:
                last_message = chunk["messages"][-1]
                if isinstance(last_message, AIMessage):
                    yield last_message
    
    def chat(self):
        """Run the agent in interactive chat mode."""
        print("=" * 60)
        print("Research Agent - LangGraph Edition")
        print("=" * 60)
        print("\nThis agent can help you search for papers on arXiv")
        print("and extract detailed information about them.")
        print("\nCommands:")
        print("  - Type your question or request")
        print("  - Type 'quit' or 'exit' to exit")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print("\nGoodbye! 👋")
                    break
                
                print("\n🤖 Assistant: ", end="", flush=True)
                
                # Stream the response
                full_response = ""
                for message in self.stream(user_input):
                    if isinstance(message, AIMessage):
                        # Print tool calls if any
                        if hasattr(message, "tool_calls") and message.tool_calls:
                            print("\n")
                            for tool_call in message.tool_calls:
                                print(f"🔧 Calling tool: {tool_call['name']}")
                                print(f"   Arguments: {tool_call['args']}")
                        # Print content if any
                        elif message.content:
                            if message.content != full_response:
                                new_content = message.content[len(full_response):]
                                print(new_content, end="", flush=True)
                                full_response = message.content
                
                print()  # New line after response
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()


def main():
    """Main function to run the research agent."""
    # Create the agent
    agent = ResearchAgent()
    
    # Run in chat mode
    agent.chat()


if __name__ == "__main__":
    main()
