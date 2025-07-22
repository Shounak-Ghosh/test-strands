"""
Test script for the MockAgent FastAPI server.
"""
import requests
import json
from requests.exceptions import ConnectionError, RequestException

# Test prompt
TEST_PROMPT = """
                    I have 3 requests:

                    1. What is the time right now?
                    2. Calculate 3111696 / 74088
                    3. Tell me how many letter R's in the word "strawberry" 🍓
                    """

def test_streaming_endpoint():
    """Test the streaming endpoint."""
    print("Testing streaming endpoint...")
    
    payload = {
        "prompt": TEST_PROMPT,
        "use_bedrock": False
    }
    
    try:
        response = requests.post(
            "http://localhost:9000/stream",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True  # Enable streaming
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print("Streaming response:")
            # Stream the response chunk by chunk
            response_text = ""
            metrics_data = None
            current_event = None
            
            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    # Parse Server-Sent Events
                    lines = chunk.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith('event: '):
                            # This is an event type
                            current_event = line[7:]  # Remove 'event: ' prefix
                        elif line.startswith('data: '):
                            # This is data
                            data = line[6:]  # Remove 'data: ' prefix
                            if current_event == 'metrics':
                                # This is metrics data
                                try:
                                    metrics_data = json.loads(data)
                                except json.JSONDecodeError:
                                    print(f"Error parsing metrics: {data}")
                            else:
                                # This is regular response data
                                if data:
                                    response_text += data
                                    print(data, end='', flush=True)
                        elif line == '':
                            # End of event, reset current_event
                            current_event = None
            
            # Display metrics if available
            if metrics_data:
                print("\n----Metrics----")
                print(json.dumps(metrics_data))
            
            print("\n--- End of stream ---\n")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error during streaming test: {e}")

def print_metrics(metrics):
    """Print metrics in a readable format."""
    print(f"Total Cycles: {metrics.get('total_cycles', 0)}")
    print(f"Total Duration: {metrics.get('total_duration', 0.0):.2f}s")
    print(f"Average Cycle Time: {metrics.get('average_cycle_time', 0.0):.2f}s")
    
    # Display token usage
    accumulated_usage = metrics.get('accumulated_usage', {})
    if accumulated_usage:
        print(f"Input Tokens: {accumulated_usage.get('inputTokens', 0)}")
        print(f"Output Tokens: {accumulated_usage.get('outputTokens', 0)}")
        print(f"Total Tokens: {accumulated_usage.get('totalTokens', 0)}")
    
    # Display latency
    accumulated_metrics = metrics.get('accumulated_metrics', {})
    if accumulated_metrics:
        print(f"Latency: {accumulated_metrics.get('latencyMs', 0)}ms")
    
    # Display tool usage
    tool_usage = metrics.get('tool_usage', {})
    if tool_usage:
        print("Tool Usage:")
        for tool_name, tool_data in tool_usage.items():
            execution_stats = tool_data.get('execution_stats', {})
            print(f"  {tool_name}: {execution_stats}")

def test_chat_endpoint():
    """Test the chat endpoint."""
    print("Testing chat endpoint...")
    
    payload = {
        "prompt": TEST_PROMPT,
        "use_bedrock": False
    }
    
    try:
        response = requests.post(
            "http://localhost:9000/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Display the response content
            print("Response Content:")
            print("-" * 40)
            print(result.get("response", "No response content"))
            print()
            
            # Display metrics in the same format as streaming
            metrics = result.get("metrics", {})
            if metrics:
                print("----Metrics----")
                print(json.dumps(metrics))
            else:
                print("No metrics available")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error during chat test: {e}")

def test_health_endpoint():
    """Test the health endpoint."""
    print("Testing health endpoint...")
    
    try:
        response = requests.get("http://localhost:9000/health")
        print(f"Status: {response.status_code}")
        
        result = response.json()
        print("Health check:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error during health test: {e}")

def test_root_endpoint():
    """Test the root endpoint."""
    print("Testing root endpoint...")
    
    try:
        response = requests.get("http://localhost:9000/")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Root endpoint response:")
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error during root test: {e}")

def main():
    """Run all tests."""
    print("MockAgent API Test Client")
    print("=" * 40)
    
    try:
        # test_health_endpoint()
        # print()
        
        # test_root_endpoint()
        # print()
        
        test_chat_endpoint()
        print()
        
        test_streaming_endpoint()
        
    except ConnectionError:
        print("Error: Could not connect to server. Make sure the server is running on http://localhost:9000")
    except RequestException as e:
        print(f"Request error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
