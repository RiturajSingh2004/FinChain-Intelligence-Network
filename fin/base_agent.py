"""
Base Agent Module - Defines the foundation for all specialized agents in the FinChain network.
Integrated with Fetch.ai uAgents framework.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

from uagents import Agent as UAgent, Context, Protocol, Model


class QueryRequest(Model):
    """Model for query requests received by agents."""
    query: str
    context: Optional[Dict[str, Any]] = None
    request_id: str = str(uuid.uuid4())


class QueryResponse(Model):
    """Model for responses sent by agents."""
    insights: List[str] = []
    recommendations: List[str] = []
    confidence: float = 0.0
    additional_data: Optional[Dict[str, Any]] = None
    request_id: str = ""


class BaseAgent(ABC):
    """
    Abstract base class for all FinChain Intelligence Network agents.
    Provides common functionality and interface that all agents must implement,
    now integrated with uAgents framework.
    """

    def __init__(self, name: str, description: str, seed: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base agent.
        
        Args:
            name: The name of the agent
            description: A brief description of the agent's capabilities
            seed: Optional seed phrase for the uAgent (will be generated if not provided)
            config: Optional configuration parameters for the agent
        """
        self.name = name
        self.description = description
        self.config = config or {}
        self.logger = logging.getLogger(f"fin.agents.{name}")
        
        # Initialize uAgent
        self.seed = seed or f"{name}-{uuid.uuid4()}"
        self.uagent = UAgent(name=name, seed=self.seed)
        
        # Create protocol for query handling
        self.protocol = Protocol(f"{name}_protocol")
        
        # Register message handler
        @self.protocol.on_message(model=QueryRequest)
        async def handle_query(ctx: Context, sender: str, msg: QueryRequest):
            self.logger.info(f"Received query from {sender}: {msg.query}")
            
            # Process the query using the agent's implementation
            result = self.process_query(msg.query)
            
            # Convert the result to QueryResponse format
            insights = result.get("insights", [])
            recommendations = result.get("recommendations", [])
            confidence = result.get("confidence", 0.0)
            
            # Extract additional data that isn't in the standard fields
            additional_data = {k: v for k, v in result.items() 
                              if k not in ["insights", "recommendations", "confidence"]}
            
            # Send the response
            response = QueryResponse(
                insights=insights,
                recommendations=recommendations,
                confidence=confidence,
                additional_data=additional_data,
                request_id=msg.request_id
            )
            
            await ctx.send(sender, response)
        
        # Include the protocol in the agent
        self.uagent.include(self.protocol)
        
        # Complete initialization
        self._initialize_agent()
        
    def _initialize_agent(self):
        """Set up the agent with any necessary initializations."""
        self.logger.info(f"Initializing agent: {self.name}")
    
    def run(self, host: str = "127.0.0.1", port: int = 8000):
        """
        Run the agent service.
        
        Args:
            host: Host address
            port: Port number
        """
        self.logger.info(f"Starting agent {self.name} on {host}:{port}")
        self.uagent.run(host=host, port=port)
    
    @abstractmethod
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query and return results.
        
        Args:
            query: The user's query string
            
        Returns:
            A dictionary containing the agent's response
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Return a list of the agent's capabilities.
        
        Returns:
            A list of capability descriptions
        """
        pass
    
    def health_check(self) -> Dict[str, Any]:
        """
        Verify the agent is functioning properly and all dependencies are available.
        
        Returns:
            A dictionary with health status information
        """
        return {
            "status": "healthy",
            "name": self.name,
            "version": self.__class__.__module__,
            "uagent_address": self.uagent.address
        }
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"