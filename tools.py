# from langchain.tools import tool
from langchain_core.tools import tool
from langchain_community.document_loaders import YoutubeLoader, WebBaseLoader
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain_experimental.utilities import PythonREPL
from youtube_transcript_api import YouTubeTranscriptApi

import time
import requests
from bs4 import BeautifulSoup

# Initialize Python REPL
python_repl = PythonREPL()

# Initialise Youtube 
youtube_loader = YouTubeTranscriptApi()

@tool
def youtube_transcript(url: str) -> str:    
    """Retrieve transcript from Youtube based url.
    Args:
        url: input youtube url.
    Returns:
        A string containing the transcript of the youtube video.
    """
    try:
        video_id = url.split("watch?v=")[-1]
        transcript = youtube_loader.fetch(video_id).to_raw_data()
        transcript_text = "\n".join([entry['text'] for entry in transcript])
        return transcript_text
    except Exception as e:
        return f"Error retrieving transcript: {str(e)}"

@tool
def duckduckgo_search_results(query: str) -> list[dict]:
    """Perform a DuckDuckGo search for the given query and return the results.
    Args:
        query: The search query string.
    Returns:
        A list of search results, where each result is a dictionary that includes the snippet, title, and link.
    """
    try:
        search = DuckDuckGoSearchResults(output_format="list")
        return search.invoke(query)
    except Exception as e:
        return f"Error performing search: {str(e)}"

@tool
def fetch_website(url:str) -> str:
    """Fetch the content of a website.
    Args:
        url: The URL of the website to fetch.
    Returns:
        The title and content of the website.
    """
    # Use requests directly to avoid browser impersonation warnings from WebBaseLoader
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text content
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text[:32_000]  # Limit length
    except Exception as e:
        return f"Error fetching website: {str(e)}"

# @tool
# def get_wiki_summary(query: str) -> str:
#     """Retrieve information from Wikipedia based on a user query.
#     Args:
#         query: A user query.
#     Returns:
#         A single string containing the retrieved article from Wikipedia.
#     """
#     if not query.strip():
#         return "Please provide a valid query."
#     try:
#         wiki_toolapi_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=16000)
#         wiki_tool = WikipediaQueryRun(api_wrapper=wiki_toolapi_wrapper)
#         result = wiki_tool.run(query)
#         return result
#     except Exception as e:
#         return f"Error retrieving information: {str(e)}"

def get_wiki_title(query: str) -> str:
    """Retrieve Wikipedia page title based on a user query.
    Args:
        query: A user query.
    Returns:
        A single string containing the retrieved article page title from Wikipedia.
    """
    if not query.strip():
        return "Please provide a valid query."
    try:
        # Reduce length of retrieved content as we just need the title
        wiki_toolapi_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
        wiki_tool = WikipediaQueryRun(api_wrapper=wiki_toolapi_wrapper)
        result = wiki_tool.run(query)
        # Extract the title from the result (assuming it's in the format "Page: <title>\nSummary: <summary>")
        title = result.split("\n")[0].replace("Page: ", "")
        return title
    except Exception as e:
        return f"Error retrieving information: {str(e)}"
    
@tool
def get_wiki_full(query: str) -> str:
    """Scrape the content of a Wikipedia page based on the user query.
    
    Args:
        query: The user query to search for on Wikipedia.
    Returns:
        A single string containing the content of the Wikipedia page.
    """
    title = get_wiki_title(query)
    url = f'https://en.wikipedia.org/wiki/{title.replace(" ", "_")}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Get all content from main article
    content = soup.find('div', {'id': 'mw-content-text'})
    
    return content.get_text()[:32_000]  # Limit to 8k tokens to avoid excessive length

# @tool
# def youtube_transcript(url: str) -> str:
#     """Retrieve transcript from Youtube based url.
#     Args:
#         url: input youtube url.
#     Returns:
#         A single string containing the transcript of the youtube videos.
#     """
#     max_attempts = 5  # Set a maximum number of attempts
#     attempts = 0
#     loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
#     while attempts < max_attempts:
#         try:
#             docs  = loader.load()
#             return docs[0].page_content
#         except Exception as e:
#             attempts += 1
#             print(f"Attempt {attempts} failed: {e}")
#             # Optionally add a delay before retrying
#             time.sleep(1) # Import the time module
#     return "Failed to retrieve transcript after multiple attempts."


@tool
def python_repl_tool(code: str) -> str:
    """
    Execute Python code and return the output.
    
    Use this tool to run Python code for calculations, data analysis, \
    or any computational tasks. The code runs in a persistent Python \
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
    