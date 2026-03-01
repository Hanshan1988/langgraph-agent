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
import sys
from dotenv import load_dotenv

from agent import BasicAgent
from tools import fetch_website, get_wiki_full, youtube_transcript, python_repl_tool, duckduckgo_search_results

load_dotenv()  # Load environment variables from .env file

model_name_default = "openai/gpt-oss-120b"
provider_default = "novita"
base_url_default = "https://router.huggingface.co/v1"
hf_token = os.getenv("HF_TOKEN")

model_kwargs = {
    "temperature": 0.2,
    "max_tokens": 8192,
    "reasoning_effort": "medium",
}
# Equip llm with tools
tools_list = [
    fetch_website,
    get_wiki_full,
    youtube_transcript,
    python_repl_tool,
    duckduckgo_search_results
]

def main():
    """Main function to run the ReAct Python REPL agent."""
    # Default to Hugging Face with default model

    # Check for command line arguments
    if len(sys.argv) > 1:
        # Use custom base_url for local models (e.g., LMStudio)
        if sys.argv[1] == "--local":
            base_url = "http://localhost:1234/v1"
            print(f"Using local model at {base_url}")
            agent = BasicAgent(base_url=base_url, api_key="not-needed", model_name="local-model", model_provider="local-provider", tools_list=tools_list)
        # Use Hugging Face endpoint with custom model
        elif sys.argv[1] == "--hf":
            if not hf_token:
                print("❌ Error: HF_TOKEN not found in environment variables")
                print("Please set HF_TOKEN to use Hugging Face endpoint")
                return
            
            # Use custom model from command line
            if len(sys.argv) < 3:
                print("❌ Error: Please specify a model name")
                print("Usage: python react_python_repl_agent.py --hf <model_name>")
                print("Example: python react_python_repl_agent.py --hf Qwen/Qwen3-Coder-Next")
                return

            # get model name from command line argument if exists else use default            
            model_name = sys.argv[2] if len(sys.argv) >= 3 else model_name_default 
            print(f"Using Hugging Face model: {model_name}")
            
            # Add reasoning_effort for OpenAI o1 models
            if "openai/gpt-oss" in model_name or "o1" in model_name:
                print("Reasoning effort: high")
                agent = BasicAgent(
                    model_name=model_name,
                    base_url=base_url_default,
                    api_key=hf_token,
                    model_provider=provider_default,
                    tools_list=tools_list
                )
            else:
                agent = BasicAgent(base_url=base_url_default, api_key=hf_token, model_name=model_name, model_provider=provider_default, tools_list=tools_list)
        else:
            print("Usage: python react_python_repl_agent.py [--local|--hf <model>]")
            print("  --local: Use local model (LMStudio) at http://localhost:1234/v1")
            print("  --hf <model>: Use custom Hugging Face model (requires HF_TOKEN env var)")
            print("               Example: --hf Qwen/Qwen3-Coder-Next")
            print("")
            print("  Default: Hugging Face with openai/gpt-oss-120b:ovhcloud")
            return
    else:
        # Use Hugging Face by default (requires HF_TOKEN environment variable)
        if not hf_token:
            print("❌ Error: HF_TOKEN not found in environment variables")
            print("Please set HF_TOKEN to use Hugging Face endpoint")
            print("Or use --local flag for local model")
            return
        
        print(f"Using Hugging Face model (default): {model_name_default}")
        print("Reasoning effort: high")
        print("Tip: Use --hf <model> to specify a different model")
        agent = BasicAgent(
            base_url=base_url_default,
            api_key=hf_token,
            model_name=model_name_default,
            model_provider=provider_default,
            tools_list=tools_list,
        )

    # Run in chat mode
    agent.chat()


if __name__ == "__main__":
    main()
