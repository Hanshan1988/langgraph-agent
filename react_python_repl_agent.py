"""
LangGraph ReAct Agent with Python REPL Tool.

This agent implements the ReAct (Reasoning + Acting) framework, which allows
the model to reason about problems and execute Python code iteratively until
it arrives at a final answer.

The ReAct framework:
1. Thought: The agent reasons about what to do next
2. Action: The agent executes a Python command
3. Observation: The agent observes the result
4. Repeat until final answer is reached
"""

import os
from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_experimental.utilities import PythonREPL
from langfuse.langchain import CallbackHandler

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Get keys for your project from the project settings page: https://cloud.langfuse.com
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-***")  # Public key is safe to expose in client-side code
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-***") 
os.environ["LANGFUSE_BASE_URL"] = os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com") # 🇺🇸 US region
 
from langfuse import get_client
 
langfuse = get_client()
 
# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

# # Initialize Langfuse CallbackHandler for Langchain (tracing)
# langfuse_handler = CallbackHandler()

# Initialize Python REPL
python_repl = PythonREPL()


@tool
def python_repl_tool(code: str) -> str:
    """
    Execute Python code and return the output.
    
    Use this tool to run Python code for calculations, data analysis,
    or any computational tasks. The code runs in a persistent Python
    environment, so variables and imports are preserved between calls.
    
    Args:
        code: Python code to execute
        
    Returns:
        The output of the code execution (stdout) or error message
    """
    try:
        result = python_repl.run(code)
        return result if result else "Code executed successfully (no output)"
    except Exception as e:
        return f"Error: {str(e)}"


# Define the agent state
class ReactState(TypedDict):
    """The state of the ReAct agent."""
    messages: Annotated[list[BaseMessage], add_messages]


class ReactPythonAgent:
    """A ReAct agent with Python REPL capability."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", base_url: str = None, api_key: str = None):
        """
        Initialize the ReAct agent with Python REPL.
        
        Args:
            model_name: Name of the model to use
            base_url: Base URL for the LLM API (optional, for local models like LMStudio)
            api_key: API key (optional, uses OPENAI_API_KEY env var by default)
        """
        # Initialize the LLM
        llm_kwargs = {
            "model": model_name,
            "temperature": 0,  # Use 0 for more consistent reasoning
        }
        
        if base_url:
            llm_kwargs["base_url"] = base_url
            llm_kwargs["api_key"] = api_key or "not-needed"
        elif api_key:
            llm_kwargs["api_key"] = api_key
        
        self.llm = ChatOpenAI(**llm_kwargs)
        
        # Define the tools
        self.tools = [python_repl_tool]
        
        # Bind tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # System prompt for ReAct
        self.system_prompt = """You are a helpful assistant with access to a Python REPL.

Use the ReAct (Reasoning + Acting) framework to solve problems:
1. THINK about what you need to do
2. ACT by writing and executing Python code
3. OBSERVE the results
4. REPEAT until you have the final answer

When using the Python REPL:
- Break down complex problems into smaller steps
- Use the tool multiple times if needed - variables persist between calls
- Import libraries as needed (numpy, pandas, matplotlib, etc.)
- For calculations, use Python's computational capabilities
- Print intermediate results to understand what's happening

Once you have the final answer, provide a clear response to the user WITHOUT calling any more tools.
"""
        
        # Create the graph
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph ReAct workflow."""
        workflow = StateGraph(ReactState)
        
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
    
    def _call_model(self, state: ReactState) -> dict:
        """Call the LLM with the current state."""
        messages = state["messages"]
        
        # Prepend system message if this is the first call
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            messages = [SystemMessage(content=self.system_prompt)] + messages
        
        response = self.llm_with_tools.invoke(
            messages,
            # config={"callbacks": [langfuse_handler]}
        )
        return {"messages": [response]}
    
    def _should_continue(self, state: ReactState) -> Literal["continue", "end"]:
        """Determine whether to continue or end the workflow."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If there are tool calls, continue
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        
        # Otherwise, end
        return "end"
    
    def run(self, message: str, verbose: bool = True) -> str:
        """
        Run the agent with a message.
        
        Args:
            message: The user's message/question
            verbose: Whether to print intermediate steps
            
        Returns:
            The agent's final response
        """
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=message)]
        }
        
        # Run the graph with streaming for verbose output
        if verbose:
            print(f"\n{'='*60}")
            print(f"Question: {message}")
            print(f"{'='*60}\n")
            
            step = 1
            for chunk in self.graph.stream(initial_state, stream_mode="values"):
                if "messages" in chunk:
                    last_message = chunk["messages"][-1]
                    
                    # Print tool calls
                    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            print(f"🤔 Step {step} - Thought: Let me execute some Python code")
                            print(f"🔧 Action: python_repl_tool")
                            print(f"📝 Code:\n{tool_call['args']['code']}")
                            step += 1
                    
                    # Print tool results
                    elif hasattr(last_message, "content") and last_message.content:
                        # Check if this is a tool message
                        if hasattr(last_message, "type") and last_message.type == "tool":
                            print(f"👁️  Observation: {last_message.content}\n")
            
            # Get final result
            result = self.graph.invoke(initial_state)
            final_message = result["messages"][-1]
            
            print(f"{'='*60}")
            print(f"✅ Final Answer:")
            print(f"{'='*60}")
            print(final_message.content)
            print()
            
            return final_message.content
        else:
            # Run without verbose output
            result = self.graph.invoke(initial_state)
            final_message = result["messages"][-1]
            return final_message.content
    
    def chat(self):
        """Run the agent in interactive chat mode."""
        print("=" * 60)
        print("ReAct Python REPL Agent")
        print("=" * 60)
        print("\nThis agent can solve problems using Python code.")
        print("It uses the ReAct framework: Reason → Act → Observe → Repeat")
        print("\nExamples:")
        print("  - 'Calculate the first 10 Fibonacci numbers'")
        print("  - 'What is the sum of squares from 1 to 100?'")
        print("  - 'Generate a random password with 16 characters'")
        print("  - 'Plot a sine wave' (if matplotlib is installed)")
        print("\nCommands:")
        print("  - Type your question or request")
        print("  - Type 'quit' or 'exit' to exit")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print("\n👋 Goodbye!")
                    break
                
                # Run the agent
                self.run(user_input, verbose=True)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()


def main():
    """Main function to run the ReAct Python REPL agent."""
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        # Use custom base_url for local models (e.g., LMStudio)
        if sys.argv[1] == "--local":
            base_url = "http://localhost:1234/v1"
            print(f"Using local model at {base_url}")
            agent = ReactPythonAgent(model_name="local-model", base_url=base_url, api_key="not-needed")
        # Use Hugging Face endpoint
        elif sys.argv[1] == "--hf":
            if not os.getenv("HF_TOKEN"):
                print("❌ Error: HF_TOKEN not found in environment variables")
                print("Please set HF_TOKEN to use Hugging Face endpoint")
                return
            
            # Default model or use custom model from command line
            model_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-Coder-Next"
            base_url = "https://router.huggingface.co/v1"
            hf_token = os.getenv("HF_TOKEN")
            
            print(f"Using Hugging Face model: {model_name}")
            agent = ReactPythonAgent(model_name=model_name, base_url=base_url, api_key=hf_token)
        else:
            print("Usage: python react_python_repl_agent.py [--local|--hf [model]]")
            print("  --local: Use local model (LMStudio) at http://localhost:1234/v1")
            print("  --hf [model]: Use Hugging Face endpoint (requires HF_TOKEN env var)")
            print("                Default model: Qwen/Qwen3-Coder-Next")
            return
    else:
        # Use OpenAI by default (requires OPENAI_API_KEY environment variable)
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
            print("Please set it or use --local or --hf flags for alternative providers")
            return
        
        agent = ReactPythonAgent()
    
    # Run in chat mode
    agent.chat()


if __name__ == "__main__":
    main()
