from typing import Any, List, Optional, Sequence
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolCall
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from openai import OpenAI
import os
import json

class CustomOpenAIChat(BaseChatModel):
    """Custom wrapper around native OpenAI client for gpt-oss models with reasoning support."""
    
    client: Any = None
    model: str = "openai/gpt-oss-120b:nscale",
    base_url: str = None,
    api_key: str = None
    temperature: float = 0.3
    top_p: float = 0.9
    reasoning_effort: str = "high"  # 'low', 'medium', or 'high'
    bound_tools: Optional[List[dict]] = None  # Store bound tools in OpenAI format
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response using native OpenAI client with reasoning."""
        
        # Convert LangChain messages to OpenAI format
        openai_messages = self._convert_messages(messages)
        
        # Prepare API call parameters
        api_params = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "reasoning_effort": self.reasoning_effort,
            "stream": False,
        }
        
        # Add tools if bound
        if self.bound_tools:
            api_params["tools"] = self.bound_tools
            api_params["tool_choice"] = "auto"
        
        # Call native OpenAI client
        response = self.client.chat.completions.create(**api_params, **kwargs)
        
        # Extract content and reasoning
        choice = response.choices[0]
        content = choice.message.content or ""  # Handle None content when tool calls exist
        # Note: HuggingFace router returns reasoning as 'reasoning_content', not 'reasoning'
        reasoning = getattr(choice.message, 'reasoning_content', None) or getattr(choice.message, 'reasoning', None)
        
        # Debug log when reasoning is present
        if reasoning and os.getenv("DEBUG_REASONING"):
            print(f"[CustomOpenAI] Reasoning extracted from API response:")
            print(f"  Model: {response.model}")
            print(f"  Finish reason: {choice.finish_reason}")
            print(f"  Reasoning: {reasoning[:200]}...")  # First 200 chars
        
        # Extract tool calls if present
        tool_calls = []
        if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                # Parse arguments if they're a JSON string
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        # If parsing fails, use the string as-is
                        pass
                
                tool_calls.append({
                    "name": tc.function.name,
                    "args": args,
                    "id": tc.id,
                })
        
        # Convert OpenAI response back to LangChain format
        message = AIMessage(
            content=content,
            additional_kwargs={
                "reasoning": reasoning,  # Store reasoning here
                "model": response.model,
                "finish_reason": choice.finish_reason,
                "tool_calls": tool_calls,  # Store tool calls
            },
            response_metadata={
                "usage": response.usage.model_dump() if response.usage else {},
                "model": response.model,
                "system_fingerprint": getattr(response, "system_fingerprint", None),
                "reasoning": reasoning,  # Also store in metadata for easy access
            },
            tool_calls=[
                ToolCall(name=tc["name"], args=tc["args"], id=tc["id"]) 
                for tc in tool_calls
            ] if tool_calls else []
        )
        
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages to OpenAI format."""
        openai_messages = []
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                openai_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
        
        return openai_messages
    
    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "custom-openai-gpt-oss"
    
    @property
    def _identifying_params(self) -> dict:
        """Return identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "reasoning_effort": self.reasoning_effort
        }
    
    def bind_tools(
        self,
        tools: Sequence[BaseTool],
        **kwargs: Any,
    ) -> "CustomOpenAIChat":
        """Bind tools to the model for function calling.
        
        Args:
            tools: A sequence of LangChain tools to bind
            **kwargs: Additional parameters
            
        Returns:
            A new instance with tools bound
        """
        # Convert LangChain tools to OpenAI format
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        
        # Create a new instance with the same parameters but with bound tools
        return self.__class__(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            top_p=self.top_p,
            reasoning_effort=self.reasoning_effort,
            bound_tools=formatted_tools,
            **kwargs
        )
