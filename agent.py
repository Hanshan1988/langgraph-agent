from textwrap import dedent
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from functools import partial
import os
import re
import time
import uuid
import asyncio
import json
from dotenv import load_dotenv

# from langchain_openai import ChatOpenAI
# from langchain_huggingface.llms import HuggingFaceEndpoint
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
import openai as openai_module

from langfuse.langchain import CallbackHandler
from langfuse import get_client, observe


class ReasoningChatOpenAI(ChatOpenAI):
    """ChatOpenAI subclass that captures reasoning_content from reasoning models."""

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        # Extract reasoning_content from the raw OpenAI response object
        if (
            isinstance(response, openai_module.BaseModel)
            and getattr(response, "choices", None)
        ):
            message = response.choices[0].message
            reasoning_content = getattr(message, "reasoning_content", None)
            if reasoning_content and result.generations:
                result.generations[0].message.additional_kwargs[
                    "reasoning_content"
                ] = reasoning_content
        return result

load_dotenv()  # Load environment variables from .env file  

os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-***")  # Public key is safe to expose in client-side code
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-***") 
os.environ["LANGFUSE_BASE_URL"] = os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com") # 🇺🇸 US region

langfuse = get_client()
# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")
# langfuse_handler = CallbackHandler()

# # Initialize the Hugging Face model
# hf_model_name = "openai/gpt-oss-120b" # "Qwen/Qwen2.5-72B-Instruct"
# hf_model_provider = "nscale" # "hf-inference"

# llm = HuggingFaceEndpoint(
#     repo_id=hf_model_name, 
#     provider=hf_model_provider,
#     max_new_tokens=8192,
#     do_sample=False,
#     # temperature=0.,
# )

# chat_model = ChatHuggingFace(llm=llm)

# # Equip llm with tools
# tools_list = [
#     fetch_website,
#     get_wiki_full,
#     youtube_transcript,
#     python_repl_tool,
#     duckduckgo_search_results
# ]

# llm_with_tools = chat_model.bind_tools(
#     tools_list
# )

# Define Agent Workflow

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    reasoning_traces: list[dict]  # Store reasoning_content from each LLM call


def assistant(state: AgentState, llm, langfuse_client=None) -> Dict[str, Any]:
    # System message
    textual_description_of_tool = dedent(
        """
        duckduckgo_search_results(query: str) -> list[dict]:
            Perform a web search using DuckDuckGo and return the results.
            Args:
                query: The search query string.
            Returns:
                A list of search results, where each result is a dictionary that includes the snippet, title, and link.

        fetch_website(url: str) -> str:
            Fetch the content of a website.
            Args:
                url: The URL of the website to fetch.
            Returns:
                The title and content of the website.

        get_wiki_full(query: str) -> str:
            Scrape the content of a Wikipedia page based on the user query.
            Args:
                query: The user query to search for on Wikipedia.
            Returns:
                A single string containing the content of the Wikipedia page.

        youtube_transcript(url: str) -> list[dict]:
            Fetch the transcript of a youtube video.
            Args:
                url: input youtube url.
            Returns:
                A list of dictionaries containing the transcript of the youtube videos.
                Each dictionary has 'text', 'start', and 'duration' keys.

        python_repl_tool(code: str) -> str:
            Execute Python code and return the output.
            Args:
                code: A string of Python code to execute.
            Returns:
                The output of the executed code or any error messages.
        """
    )

    sys_msg = SystemMessage(
        content=dedent(
            f"""
            You are a helpful assistant at answering user questions. \
            You would follow the ReAct framework to reason and call tools iteratively until you arrive at a final answer. \
            <think>Thought process:</think>
            <act>Action to take:</act>
            <observe>Observation:</observe>
            Your final answer will be between <answer> and </answer> tags. \
            You can access provided tools:\n{textual_description_of_tool}\n"""
        )
    )

    # Invoke the LLM and capture the full response
    llm_response = llm.invoke([sys_msg] + state["messages"])
    
    # Extract reasoning_content if available (for reasoning models)
    # ReasoningChatOpenAI stores it in additional_kwargs["reasoning_content"]
    reasoning_content = llm_response.additional_kwargs.get("reasoning_content")
    reasoning_traces = state.get("reasoning_traces", [])
    
    # Log the generation to Langfuse with reasoning_content
    if reasoning_content:
        reasoning_trace = {
            "timestamp": time.time(),
            "reasoning_content": reasoning_content,
            "content": llm_response.content,
            "tool_calls": llm_response.tool_calls if hasattr(llm_response, 'tool_calls') else None
        }
        reasoning_traces.append(reasoning_trace)
        
    return {
        "messages": [llm_response],
        "reasoning_traces": reasoning_traces,
    }

# # Build the StateGraph for the agent
# # The graph
# builder = StateGraph(AgentState)

# # Define nodes: these do the work
# builder.add_node("assistant", assistant)
# builder.add_node("tools", ToolNode(tools_list))

# # Define edges: these determine how the control flow moves
# builder.add_edge(START, "assistant")
# builder.add_conditional_edges(
#     "assistant",
#     # If the latest message requires a tool, route to tools
#     # Otherwise, provide a direct response
#     tools_condition,
# )
# builder.add_edge("tools", "assistant")
# agent_graph = builder.compile()

def extract_answer(text):
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return 'None'

class BasicAgent:

    def __init__(self, base_url, api_key, model_name, model_provider, tools_list, **model_kwargs):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.model_provider = model_provider
        self.tools_list = tools_list
        self.model_kwargs = model_kwargs
        print("BasicAgent initialized.")
        # Create agent with all the tools
        self.agent_graph = self.build_agent_graph()

    def build_llm_with_tools(self):
        print("Building Hugging Face model and tools...")
        # Initialize the Hugging Face model
        # llm = HuggingFaceEndpoint(
        #     repo_id=self.model_name, 
        #     provider=self.model_provider,
        #     max_new_tokens=8192,
        #     do_sample=False,
        #     temperature=0.2,
        # )

        # chat_model = ChatHuggingFace(llm=llm)

        chat_model = ReasoningChatOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            model=f"{self.model_name}:{self.model_provider}",
            **self.model_kwargs
        )

        # Equip llm with tools

        llm_with_tools = chat_model.bind_tools(
            self.tools_list
        )
        print("LLM with tools built successfully.")
        return llm_with_tools
    
    def build_agent_graph(self):    
        llm_with_tools = self.build_llm_with_tools()
        # Build the StateGraph for the agent
        builder = StateGraph(AgentState)

        # Define nodes: these do the work
        builder.add_node("assistant", partial(assistant, llm=llm_with_tools))
        builder.add_node("tools", ToolNode(self.tools_list))

        # Define edges: these determine how the control flow moves
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            # If the latest message requires a tool, route to tools
            # Otherwise, provide a direct response
            tools_condition,
        )
        builder.add_edge("tools", "assistant")
        agent_graph = builder.compile()
        print("Agent graph built successfully.")
        return agent_graph

    @observe(name="agent_call")
    async def __call__(self, question: str, task_id: str = None) -> str:
        """
        Run the agent with enhanced Langfuse tracing including reasoning_content.
        """
        run_name = f"agent_question_{task_id or uuid.uuid4().hex[:8]}"

        # Update the current trace with input metadata
        langfuse.update_current_trace(
            name=run_name,
            input={"question": question},
            metadata={
                "task_id": task_id,
                "question_preview": question[:200],
                "model": f"{self.model_name}:{self.model_provider}",
            },
            tags=["agent", "question_answering", "reasoning_model"],
        )

        # CallbackHandler auto-links to the current @observe() trace
        handler = CallbackHandler()

        messages = [
            HumanMessage(content=question)
        ]

        start_time = time.time()

        try:
            response = await self.agent_graph.ainvoke(
                {
                    "messages": messages,
                    "reasoning_traces": [],
                },
                config={
                    "recursion_limit": 8,
                    "callbacks": [handler],
                    "run_name": run_name,
                    "metadata": {
                        "question_preview": question[:200],
                    }
                }
            )

            end_time = time.time()
            response_text = response['messages'][-1].content
            answer = extract_answer(response_text)
            reasoning_traces = response.get('reasoning_traces', [])

            # Update trace with final output
            langfuse.update_current_trace(
                output={
                    "answer": answer,
                    "full_response": response_text,
                    "reasoning_traces": reasoning_traces,
                    "num_messages": len(response['messages']),
                },
                metadata={
                    "duration_seconds": end_time - start_time,
                    "has_reasoning": len(reasoning_traces) > 0,
                    "reasoning_steps": len(reasoning_traces),
                }
            )

        except Exception as e:
            langfuse.update_current_trace(
                output={"error": str(e)},
                metadata={"status": "error"}
            )
            raise
        finally:
            langfuse.flush()

        print(f"Trace logged for task_id: {task_id}, reasoning_steps: {len(reasoning_traces)}")

        return answer
    
    def chat(self):
        """Run the agent in interactive chat mode."""
        print("=" * 60)
        print("ReAct Agent with Python REPL & Wikipedia")
        print("=" * 60)
        # print(f"\nMax iterations: {self.max_iterations} (prevents infinite loops)")
        print("\nThis agent can solve problems using Python code and retrieve")
        print("information from Wikipedia.")
        print("\nIt uses the ReAct framework: Reason → Act → Observe → Repeat")
        print("\nExample questions:")
        print("\n  Python REPL:")
        print("    - 'Calculate the first 10 Fibonacci numbers'")
        print("    - 'What is the sum of squares from 1 to 100?'")
        print("    - 'Generate a random password with 16 characters'")
        print("\n  Wikipedia:")
        print("    - 'Who was Alan Turing?'")
        print("    - 'Tell me about quantum computing'")
        print("    - 'What is the history of Python programming language?'")
        print("\n  Combined:")
        print("    - 'Find the population of Tokyo and calculate its square root'")
        print("    - 'What is Pi? Calculate it to 10 decimal places'")
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
                task_id = str(uuid.uuid4())[:8]
                answer = asyncio.run(self(user_input))
                
                print(f"\n{'='*60}")
                print(f"✅ Answer:")
                print(f"{'='*60}")
                print(answer)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()