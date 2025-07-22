from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import json
from typing import AsyncGenerator
import logging

# Import our MockAgent and tools
from main import MockAgent, letter_counter
from strands_tools import calculator, current_time, python_repl, http_request

app = FastAPI(title="MockAgent API", description="API for streaming responses from MockAgent")

# Global agent instance - initialized once for better performance
agent_instance = None

class PromptRequest(BaseModel):
    prompt: str
    use_bedrock: bool = False  # Option to switch between OpenAI and Bedrock models

def get_agent_instance(use_bedrock: bool = False) -> MockAgent:
    """Get or create agent instance."""
    global agent_instance
    if agent_instance is None:
        agent_instance = MockAgent()
    
    # If requested to use Bedrock, temporarily switch the agent's model
    if use_bedrock and hasattr(agent_instance, 'bedrock_model'):
        # Create a new agent with Bedrock model for this request
        from strands import Agent
        bedrock_agent = Agent(
            model=agent_instance.bedrock_model,
            tools=[calculator, current_time, python_repl, http_request, letter_counter]
        )
        # Create a temporary MockAgent-like object
        class TempAgent:
            def __init__(self, agent):
                self.agent = agent
        return TempAgent(bedrock_agent)
    
    return agent_instance

@app.post("/stream")
async def stream_response(request: PromptRequest):
    """Stream response from MockAgent using native Strands streaming."""
    async def generate():
        agent = get_agent_instance(request.use_bedrock)
        
        try:
            # Use the MockAgent's stream_async method which includes output capture
            async for event in agent.stream_async(request.prompt):
                if "data" in event:
                    # Only stream text chunks to the client
                    yield event["data"]
        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )

@app.post("/chat")
async def chat_response(request: PromptRequest):
    """Get a complete response from MockAgent (non-streaming)."""
    try:
        agent = get_agent_instance(request.use_bedrock)
        
        # Run agent query in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            agent.query if hasattr(agent, 'query') else agent.agent,
            request.prompt
        )
        
        # Extract content from response
        if hasattr(response, 'message') and 'content' in response.message:
            content = response.message['content']
        else:
            content = str(response)
            
        return {
            "response": content,
            "model": "bedrock" if request.use_bedrock else "openai"
        }
        
    except Exception as e:
        logging.error(f"Error in chat_response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Quick check to ensure agent can be initialized
        agent = get_agent_instance()
        return {
            "status": "healthy",
            "agent_initialized": agent is not None,
            "available_models": ["openai", "bedrock"]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "MockAgent API",
        "endpoints": {
            "/stream": "POST - Stream agent responses",
            "/chat": "POST - Get complete agent responses",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        },
        "example_request": {
            "prompt": "What is 2 + 2?",
            "use_bedrock": False
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize agent on startup
    try:
        agent_instance = MockAgent()
        logging.info("MockAgent initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize MockAgent: {e}")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=9000,
        reload=True,
        log_level="info"
    )
