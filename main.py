import os
from dotenv import load_dotenv
import boto3
from strands.models import BedrockModel
from strands.models.openai import OpenAIModel

from strands import Agent, tool
from strands_tools import calculator, current_time, python_repl
from strands.session.file_session_manager import FileSessionManager
import logging
import sys
from io import StringIO
import asyncio


@tool
def letter_counter(word: str, letter: str) -> int:
    """
    Count occurrences of a specific letter in a word.

    Args:
        word (str): The input word to search in
        letter (str): The specific letter to count

    Returns:
        int: The number of occurrences of the letter in the word
    """
    if not isinstance(word, str) or not isinstance(letter, str):
        return 0

    if len(letter) != 1:
        raise ValueError("The 'letter' parameter must be a single character")

    return word.lower().count(letter.lower())


class MockAgent:
    def __init__(self):
        """Initialize the MockAgent with environment setup and agent configuration."""
        self._load_environment()
        self._configure_logging()
        self._setup_models()
        self._setup_agent()
    
    def _load_environment(self):
        """Load environment variables."""
        load_dotenv(override=True)
    
    def _configure_logging(self):
        """Configure logging to suppress verbose output."""
        # Set specific loggers to CRITICAL to suppress agent-related output
        logging.getLogger("strands").setLevel(logging.CRITICAL)
        logging.getLogger("openai").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)
        logging.getLogger("anthropic").setLevel(logging.CRITICAL)
        logging.getLogger("boto3").setLevel(logging.CRITICAL)
        logging.getLogger("botocore").setLevel(logging.CRITICAL)
        logging.getLogger("urllib3").setLevel(logging.CRITICAL)
        
        # Don't disable all logging - just target specific loggers
        # This allows server logs to still work
    
    def _setup_models(self):
        """Set up the Bedrock and OpenAI models."""
        # AWS Bedrock session
        self.boto_session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )

        self.bedrock_model = BedrockModel(
            model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            boto_session=self.boto_session
        )

        self.openai_model = OpenAIModel(
            client_args={
                "api_key": os.getenv("FIXE_OPENAI_API_KEY"),
            },
            model_id=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            params={
                "temperature": 0.7,
            }
        )
    
    def _setup_agent(self):
        """Set up the agent with tools."""
        self.agent = Agent(
            model=self.openai_model, 
            tools=[calculator, current_time, python_repl, letter_counter]
        )
    
    def _capture_output(self, func, *args, **kwargs):
        """Capture and suppress stdout/stderr during function execution."""
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        captured_output = StringIO()

        sys.stdout = captured_output
        sys.stderr = captured_output

        try:
            result = func(*args, **kwargs)
            # print(result.metrics.get_summary())
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

        return result
    
    async def _capture_output_async(self, async_gen):
        """Capture and suppress stdout/stderr during async generator execution."""
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        captured_output = StringIO()

        # Redirect both stdout and stderr to the StringIO buffer
        sys.stdout = captured_output
        sys.stderr = captured_output

        try:
            async for event in async_gen:
                yield event
        finally:
            # Restore original stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    
    def run_test_scenario(self):
        """Run the test scenario with multiple requests."""
        message = """
                    I have 3 requests:

                    1. What is the time right now?
                    2. Calculate 3111696 / 74088
                    3. Tell me how many letter R's in the word "strawberry" 🍓
                    """
        
        response = self._capture_output(self.agent, message)
        print("\nAGENT MESSAGE CONTENT\n", response.message['content'])
        return response
    
    def query(self, message: str):
        """Send a custom query to the agent."""
        response = self._capture_output(self.agent, message)
        return response
    
    async def query_async(self, message: str):
        """Send a custom query to the agent asynchronously."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            self._capture_output,
            self.agent,
            message
        )
        return response
    
    async def stream_async(self, message: str):
        """Stream agent responses asynchronously using built-in stream_async."""
        # Use the built-in stream_async method from Strands Agent with output capture
        async for event in self._capture_output_async(self.agent.stream_async(message)):
            yield event
    
    async def demo_streaming(self, message: str = None):
        """Demonstrate streaming functionality."""
        if message is None:
            message = "What is the current time and calculate 15 * 23?"
        
        print(f"Streaming response for: {message}")
        print("-" * 50)
        
        async for event in self.stream_async(message):
            print(f"Event: {event}")


# Example usage
if __name__ == "__main__":
    mock_agent = MockAgent()
    mock_agent.run_test_scenario()