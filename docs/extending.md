# Extending the FinChain Intelligence Network

This guide explains how to extend the FinChain Intelligence Network (FIN) by creating new specialized agents or enhancing existing ones.

## Creating a New Agent

### 1. Create the Agent Directory Structure

Create a new directory for your agent in the `agents/` directory:

```bash
mkdir -p agents/your_agent_name
```

### 2. Create the Agent Files

Create the following files in your agent directory:

- `__init__.py` - For package initialization and importing
- `your_agent_name.py` - The main agent implementation

### 3. Implement the Agent Class

Your agent class should inherit from the `BaseAgent` class and implement all required methods. Here's a template:

```python
"""
YourAgentName Implementation
"""

import logging
from typing import Dict, List, Any, Optional
from fin.base_agent import BaseAgent


class YourAgentName(BaseAgent):
    """
    YourAgentName description and responsibilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize your agent.
        
        Args:
            config: Optional configuration parameters
        """
        name = "your_agent_name"
        description = "Your agent's description"
        super().__init__(name, description, config)
        # Initialize your agent-specific attributes here
        
    def _initialize_agent(self):
        """Set up your agent with any necessary initializations."""
        super()._initialize_agent()
        # Your initialization code here
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query.
        
        Args:
            query: The user's query string
            
        Returns:
            A dictionary containing insights and recommendations
        """
        self.logger.info(f"Processing query: {query}")
        
        # Your query processing logic here
        
        response = {
            "insights": [],
            "recommendations": [],
            "confidence": 0.0
        }
        
        # Add insights and recommendations based on the query
        
        return response
    
    def get_capabilities(self) -> List[str]:
        """
        Return a list of the agent's capabilities.
        
        Returns:
            A list of capability descriptions
        """
        return [
            "Capability 1",
            "Capability 2",
            "Capability 3"
        ]
    
    # Add any additional agent-specific methods here
```

### 4. Update the Agent's __init__.py File

Update the `__init__.py` file to expose your agent class:

```python
"""
YourAgentName - Brief description.
"""

from .your_agent_name import YourAgentName
```

### 5. Register Your Agent with the Orchestrator

To use your agent with the orchestrator, you need to register it. In your application code:

```python
from fin.orchestrator import FinChainOrchestrator
from agents.your_agent_name.your_agent_name import YourAgentName

# Create the orchestrator
orchestrator = FinChainOrchestrator()

# Create and register your agent
your_agent = YourAgentName()
orchestrator.register_agent(your_agent)
```

## Enhancing Agent Capabilities

### Adding New Capabilities

To add new capabilities to an existing agent:

1. Implement the new functionality in the agent class
2. Update the `get_capabilities()` method to include the new capability
3. Update the query processing logic to handle queries related to the new capability

### Integrating with External Services

To integrate your agent with external services (APIs, databases, etc.):

1. Add the necessary client libraries to the project dependencies
2. Initialize the clients in the `_initialize_agent()` method
3. Use the clients to fetch or process data in your agent's methods

Example:

```python
def _initialize_agent(self):
    super()._initialize_agent()
    # Initialize API client
    self.api_client = ExternalAPIClient(
        api_key=self.config.get("api_key"),
        base_url=self.config.get("api_base_url")
    )
    
def fetch_external_data(self, query_params):
    """Fetch data from external service."""
    return self.api_client.get_data(query_params)
```

## Optimizing Agent Performance

### Caching

If your agent makes expensive API calls or performs complex computations, consider implementing caching:

```python
def __init__(self, config: Optional[Dict[str, Any]] = None):
    # ... existing code ...
    self.cache = {}
    self.cache_ttl = config.get("cache_ttl", 3600)  # 1 hour default
    
def get_data_with_caching(self, key, fetch_function, *args, **kwargs):
    """Get data with caching support."""
    import time
    
    now = time.time()
    if key in self.cache:
        cached_time, cached_data = self.cache[key]
        if now - cached_time < self.cache_ttl:
            return cached_data
    
    # Fetch fresh data
    data = fetch_function(*args, **kwargs)
    self.cache[key] = (now, data)
    return data
```

### Async Processing

For agents that need to handle multiple tasks concurrently, consider using async processing:

```python
import asyncio

async def async_process_query(self, query: str) -> Dict[str, Any]:
    """Process a query asynchronously."""
    tasks = []
    tasks.append(self.async_task1(query))
    tasks.append(self.async_task2(query))
    
    results = await asyncio.gather(*tasks)
    # Process results
    # ...
    return response
```

## Testing Your Agent

Create tests for your agent in the `tests/` directory to ensure it functions correctly:

```python
import unittest
from agents.your_agent_name.your_agent_name import YourAgentName

class TestYourAgent(unittest.TestCase):
    def setUp(self):
        self.agent = YourAgentName()
    
    def test_capabilities(self):
        capabilities = self.agent.get_capabilities()
        self.assertTrue(len(capabilities) > 0)
    
    def test_process_query(self):
        query = "Test query for your agent"
        response = self.agent.process_query(query)
        self.assertIn("insights", response)
        self.assertIn("recommendations", response)
```

## Integration with the Fetch.ai SDK

To register your agent on Agentverse using the Fetch.ai SDK:

```python
from fetchai.sdk import Agent, Registry

def register_on_agentverse(agent_name, agent_description, capabilities):
    # Create Fetch.ai agent
    fetch_agent = Agent(
        name=agent_name,
        description=agent_description,
        capabilities=capabilities
    )
    
    # Register on Agentverse
    registry = Registry()
    registry.register(fetch_agent)
    
    return fetch_agent
```

## Documentation

Remember to document your agent's capabilities, configuration options, and usage examples to help other developers understand how to use it. 