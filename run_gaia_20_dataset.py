import os
import time
import asyncio
import requests
import inspect
import pandas as pd
import re
from dotenv import load_dotenv

from agent import BasicAgent
from tools import fetch_website, get_wiki_full, youtube_transcript, python_repl_tool, duckduckgo_search_results

load_dotenv()  # Load environment variables from .env file

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

api_url = DEFAULT_API_URL
questions_url = f"{api_url}/questions"
submit_url = f"{api_url}/submit"

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

async def run_and_submit_all(agent):
    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
                print("Fetched questions list is empty.")
                return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
            print(f"Error decoding JSON response from questions endpoint: {e}")
            print(f"Response text: {response.text[:500]}")
            return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = await agent(question_text, task_id=task_id)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)
    return answers_payload, pd.DataFrame(results_log)
    

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    print('Using Hugging Face Model:')
    print(f"   Model Name: {model_name_default}")
    print(f"   Model Provider: {provider_default}")

    print("-"*(60 + len(" App Starting ")) + "\n")
    agent = BasicAgent(
        base_url=base_url_default, 
        api_key=hf_token, 
        model_name=model_name_default, 
        model_provider=provider_default, 
        tools_list=tools_list, 
        **model_kwargs
    )
    asyncio.run(run_and_submit_all(agent))