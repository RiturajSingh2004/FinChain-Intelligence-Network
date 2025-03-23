"""
FinChain Orchestrator - Coordinates all specialized agents in the FinChain Intelligence Network.
Integrated with Fetch.ai uAgents framework.
"""

import asyncio
import logging
import re
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict

from uagents import Agent as UAgent, Context, Protocol, Model
from uagents.resolver import GlobalResolver

from .base_agent import QueryRequest, QueryResponse, BaseAgent


class OrchestratorRequest(Model):
    """Model for requests received by the orchestrator."""
    query: str
    risk_profile: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    request_id: str = str(uuid.uuid4())


class OrchestratorResponse(Model):
    """Model for responses sent by the orchestrator."""
    query: str
    agents_consulted: List[str] = []
    insights: List[Dict[str, Any]] = []
    recommendations: List[Dict[str, Any]] = []
    confidence: float = 0.0
    additional_data: Optional[Dict[str, Any]] = None
    request_id: str = ""
    timestamp: str = datetime.now().isoformat()


class FinChainOrchestrator:
    """
    The FinChain Orchestrator is responsible for:
    1. Managing all specialized agents in the network
    2. Routing user queries to the appropriate agents
    3. Synthesizing responses from multiple agents
    4. Providing a unified interface for users
    Now integrated with uAgents framework for enhanced agent communication.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the FinChain Orchestrator.
        
        Args:
            config: Optional configuration parameters
        """
        self.logger = logging.getLogger("fin.orchestrator")
        self.config = config or {}
        self.agents = {}
        self.agent_addresses = {}
        self.conversation_history = []
        self.agent_affinities = defaultdict(float)
        
        # Initialize uAgent for the orchestrator
        self.seed = self.config.get("seed", f"orchestrator-{uuid.uuid4()}")
        self.uagent = UAgent(name="finchain_orchestrator", seed=self.seed)
        
        # Create protocol for receiving requests
        self.protocol = Protocol("orchestrator_protocol")
        
        # Register message handler
        @self.protocol.on_message(model=OrchestratorRequest)
        async def handle_orchestrator_request(ctx: Context, sender: str, msg: OrchestratorRequest):
            self.logger.info(f"Received orchestration request from {sender}: {msg.query}")
            
            # Process the request
            response = await self.process_query_async(msg.query, msg.risk_profile, msg.context)
            
            # Create the response object
            orchestrator_response = OrchestratorResponse(
                query=msg.query,
                agents_consulted=response.get("agents_consulted", []),
                insights=response.get("insights", []),
                recommendations=response.get("recommendations", []),
                confidence=response.get("confidence", 0.0),
                additional_data={k: v for k, v in response.items() 
                                if k not in ["query", "agents_consulted", "insights", 
                                             "recommendations", "confidence"]},
                request_id=msg.request_id,
                timestamp=datetime.now().isoformat()
            )
            
            # Send the response
            await ctx.send(sender, orchestrator_response)
        
        # Include the protocol in the orchestrator agent
        self.uagent.include(self.protocol)
        
        # Initialize the resolver for agent discovery
        self.resolver = GlobalResolver()
        
        # Complete initialization
        self._initialize_orchestrator()
    
    def _initialize_orchestrator(self):
        """Set up the orchestrator with default agents and configurations."""
        self.logger.info("Initializing FinChain Orchestrator with uAgents integration")
    
    def register_agent(self, agent: BaseAgent):
        """
        Register a new agent with the orchestrator.
        
        Args:
            agent: The agent instance to register
        """
        self.agents[agent.name] = agent
        self.agent_addresses[agent.name] = agent.uagent.address
        self.logger.info(f"Registered agent: {agent.name} with address {agent.uagent.address}")
    
    async def process_query_async(self, query: str, risk_profile: Optional[str] = None, 
                                  context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user query asynchronously by routing it to the appropriate agents.
        
        Args:
            query: The user's query string
            risk_profile: Optional risk profile for investment-related queries
            context: Optional additional context for the query
            
        Returns:
            A dictionary containing the orchestrated response
        """
        query_timestamp = datetime.now().isoformat()
        context = context or {}
        
        # Append risk profile to query if provided
        if risk_profile and risk_profile.lower() != "not specified":
            full_query = f"{query} (Risk Profile: {risk_profile})"
        else:
            full_query = query
        
        # Store query in conversation history
        self.conversation_history.append({
            "role": "user",
            "content": full_query,
            "timestamp": query_timestamp
        })
        
        # Create a context object
        query_context = self._create_context(full_query)
        query_context.update(context)
        
        # Identify relevant agents
        relevant_agents = self._identify_relevant_agents(full_query)
        self.logger.info(f"Identified relevant agents: {', '.join(relevant_agents)}")
        
        # Create a request ID for this query
        request_id = str(uuid.uuid4())
        
        # Create tasks to query all relevant agents
        agent_tasks = []
        for agent_name in relevant_agents:
            if agent_name in self.agents:
                agent_task = self._query_agent(agent_name, full_query, query_context, request_id)
                agent_tasks.append(agent_task)
        
        # Wait for all agent responses
        agent_responses = {}
        if agent_tasks:
            results = await asyncio.gather(*agent_tasks, return_exceptions=True)
            
            for agent_name, result in zip(relevant_agents, results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error processing query with agent {agent_name}: {result}")
                    agent_responses[agent_name] = {
                        "error": str(result),
                        "insights": [f"Error occurred: {str(result)}"],
                        "recommendations": ["Please try again or reformulate your query"],
                        "confidence": 0.0
                    }
                else:
                    agent_responses[agent_name] = result
                    
                    # Update agent affinity based on confidence score
                    if "confidence" in result:
                        self.agent_affinities[agent_name] += result["confidence"] * 0.1
        
        # Synthesize the responses
        synthesized_response = self._synthesize_responses(full_query, agent_responses, query_context)
        
        # Store response in conversation history
        self.conversation_history.append({
            "role": "system",
            "content": synthesized_response,
            "timestamp": datetime.now().isoformat(),
            "query_timestamp": query_timestamp
        })
        
        return synthesized_response
    
    def process_query(self, query: str, risk_profile: Optional[str] = None, 
                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Synchronous wrapper for process_query_async.
        
        Args:
            query: The user's query string
            risk_profile: Optional risk profile for investment-related queries
            context: Optional additional context for the query
            
        Returns:
            A dictionary containing the orchestrated response
        """
        # Create a new event loop for this thread if there isn't one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async method
        try:
            return loop.run_until_complete(
                self.process_query_async(query, risk_profile, context)
            )
        finally:
            # Clean up if we created a new loop
            if loop != asyncio.get_event_loop():
                loop.close()
    
    async def _query_agent(self, agent_name: str, query: str, context: Dict[str, Any], 
                           request_id: str) -> Dict[str, Any]:
        """
        Query an agent asynchronously using uAgents messaging.
        
        Args:
            agent_name: The name of the agent to query
            query: The query string
            context: Query context
            request_id: Request identifier
            
        Returns:
            Agent response dictionary
        """
        agent = self.agents[agent_name]
        agent_address = self.agent_addresses[agent_name]
        
        # For real uAgents integration, we would send a message to the agent
        # and wait for a response. For this implementation, we're calling the
        # agent's process_query method directly.
        
        # Direct method call (fallback approach)
        try:
            return agent.process_query(query)
        except Exception as e:
            self.logger.error(f"Error querying agent {agent_name}: {e}")
            raise
    
    def _create_context(self, query: str) -> Dict[str, Any]:
        """
        Create a context object for the current query.
        
        Args:
            query: The user's query string
            
        Returns:
            A dictionary containing context information
        """
        # Extract recent conversation (last 5 exchanges)
        recent_conversation = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
        
        # Determine if query is a follow-up question
        is_followup = len(self.conversation_history) > 1
        
        # Extract entities and keywords
        entities = self._extract_entities(query)
        
        return {
            "recent_conversation": recent_conversation,
            "is_followup": is_followup,
            "entities": entities,
            "timestamp": datetime.now().isoformat()
        }
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities and keywords from text.
        This is a simplified implementation - in production, use a proper NER system.
        
        Args:
            text: The text to extract entities from
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = {
            "cryptocurrencies": [],
            "companies": [],
            "regulations": [],
            "technologies": []
        }
        
        # Simple pattern matching
        crypto_patterns = ["bitcoin", "ethereum", "solana", "cardano", "ripple", "xrp", "bnb", "chainlink"]
        for crypto in crypto_patterns:
            if re.search(r'\b' + crypto + r'\b', text.lower()):
                entities["cryptocurrencies"].append(crypto.title())
        
        company_patterns = ["binance", "coinbase", "kraken", "metamask", "uniswap", "aave", "compound"]
        for company in company_patterns:
            if re.search(r'\b' + company + r'\b', text.lower()):
                entities["companies"].append(company.title())
        
        regulation_patterns = ["mifid", "gdpr", "kyc", "aml", "fatf", "mica", "howey"]
        for reg in regulation_patterns:
            if re.search(r'\b' + reg + r'\b', text.lower()):
                entities["regulations"].append(reg.upper())
        
        tech_patterns = ["blockchain", "defi", "nft", "dao", "smart contract", "layer 2", "rollup"]
        for tech in tech_patterns:
            if re.search(r'\b' + tech + r'\b', text.lower()):
                entities["technologies"].append(tech.title())
        
        return entities
    
    def _identify_relevant_agents(self, query: str) -> List[str]:
        """
        Determine which agents are most relevant for the given query.
        
        Args:
            query: The user's query string
            
        Returns:
            A list of agent names that should process the query
        """
        return self._keyword_based_agent_selection(query)
    
    def _keyword_based_agent_selection(self, query: str) -> List[str]:
        """
        Use keyword matching to identify relevant agents.
        
        Args:
            query: The user's query string
            
        Returns:
            A list of relevant agent names
        """
        query_lower = query.lower()
        relevant_agents = set()
        
        # Define keywords for each agent
        agent_keywords = {
            "blockchain_analyst": [
                "blockchain", "transaction", "smart contract", "crypto", "token", "wallet", 
                "address", "block", "hash", "ethereum", "bitcoin", "node", "mining", 
                "consensus", "security", "vulnerability", "audit", "gas"
            ],
            "crypto_economics": [
                "tokenomics", "token economics", "token model", "defi", "yield", "farming", 
                "liquidity", "amm", "lending", "sustainability", "governance", "staking", 
                "inflation", "supply", "demand", "incentive", "tvl", "protocol"
            ],
            "fintech_navigator": [
                "fintech", "trend", "regulation", "market", "payment", "banking", "finance", 
                "news", "innovation", "technology", "digital", "mobile", "api", "platform", 
                "integration", "data", "open banking", "financial", "inclusion"
            ],
            "ml_investment_strategist": [
                "investment", "predict", "forecast", "portfolio", "strategy", "risk", 
                "return", "asset", "allocation", "recommendation", "model", "algorithm", 
                "optimize", "balance", "diversify", "rebalance", "machine learning", "ai"
            ],
            "regulatory_compliance": [
                "regulation", "compliance", "legal", "law", "jurisdiction", "framework", 
                "policy", "governance", "aml", "kyc", "reporting", "requirement", "gdpr", 
                "sec", "mifid", "authority", "license"
            ]
        }
        
        # Find matching keywords
        for agent_name, keywords in agent_keywords.items():
            if agent_name in self.agents:  # Only consider registered agents
                for keyword in keywords:
                    if keyword in query_lower:
                        relevant_agents.add(agent_name)
                        break
        
        # Consider agent affinities for disambiguating similar queries
        if len(relevant_agents) > 3:
            agent_scores = [(agent, self.agent_affinities[agent]) for agent in relevant_agents]
            agent_scores.sort(key=lambda x: x[1], reverse=True)
            relevant_agents = {agent for agent, _ in agent_scores[:3]}
        
        # If no agents were deemed relevant or not enough agents matched, add agents with highest affinity
        if len(relevant_agents) < 2:
            additional_agents = sorted(
                [(a, self.agent_affinities[a]) for a in self.agents if a not in relevant_agents],
                key=lambda x: x[1],
                reverse=True
            )
            for agent, _ in additional_agents[:2 - len(relevant_agents)]:
                relevant_agents.add(agent)
        
        # Still no relevant agents? Use all agents as fallback
        if not relevant_agents:
            return list(self.agents.keys())
        
        return list(relevant_agents)
    
    def _synthesize_responses(self, query: str, agent_responses: Dict[str, Dict[str, Any]], 
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize responses from multiple agents into a coherent answer.
        
        Args:
            query: The original user query
            agent_responses: Dictionary mapping agent names to their responses
            context: The context object with conversation history and entities
            
        Returns:
            A unified response dictionary
        """
        synthesized = {
            "query": query,
            "agents_consulted": list(agent_responses.keys()),
            "insights": [],
            "recommendations": [],
            "confidence": 0.0,
            "entities": context.get("entities", {}),
            "portfolio_allocation": None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Track insights and recommendations to avoid duplicates
        seen_insights = set()
        seen_recommendations = set()
        
        # Extract additional data from responses
        for agent_name, response in agent_responses.items():
            # Handle portfolio allocation if present (from ML Investment Strategist)
            if agent_name == "ml_investment_strategist" and "portfolio_allocation" in response:
                synthesized["portfolio_allocation"] = response["portfolio_allocation"]
            
            # Process all other fields that might contain useful data
            for key, value in response.items():
                if key not in ["insights", "recommendations", "confidence"] and key not in synthesized:
                    synthesized[key] = value
        
        # Extract insights and recommendations from each agent response with deduplication
        for agent_name, response in agent_responses.items():
            if "insights" in response:
                for insight in response["insights"]:
                    # Convert insight to string if it's not already
                    insight_str = insight if isinstance(insight, str) else str(insight)
                    
                    # Check for semantic duplicates (not just exact matches)
                    is_duplicate = False
                    for seen_insight in seen_insights:
                        similarity = self._text_similarity(insight_str, seen_insight)
                        if similarity > 0.8:  # High similarity threshold
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        seen_insights.add(insight_str)
                        synthesized["insights"].append({
                            "content": insight_str,
                            "source": agent_name,
                            "relevance": self._calculate_relevance(insight_str, query)
                        })
                    
            if "recommendations" in response:
                for recommendation in response["recommendations"]:
                    # Convert recommendation to string if it's not already
                    rec_str = recommendation if isinstance(recommendation, str) else str(recommendation)
                    
                    # Check for semantic duplicates
                    is_duplicate = False
                    for seen_rec in seen_recommendations:
                        similarity = self._text_similarity(rec_str, seen_rec)
                        if similarity > 0.8:  # High similarity threshold
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        seen_recommendations.add(rec_str)
                        synthesized["recommendations"].append({
                            "content": rec_str,
                            "source": agent_name,
                            "relevance": self._calculate_relevance(rec_str, query)
                        })
                    
            # Aggregate confidence scores with weighting by agent affinity
            if "confidence" in response:
                agent_weight = 1.0 + self.agent_affinities.get(agent_name, 0)
                synthesized["confidence"] += (response["confidence"] * agent_weight)
        
        # Normalize confidence score
        if agent_responses:
            total_weight = sum(1.0 + self.agent_affinities.get(name, 0) for name in agent_responses.keys())
            synthesized["confidence"] /= total_weight
            synthesized["confidence"] = min(1.0, max(0.0, synthesized["confidence"]))  # Ensure valid range
        
        # Sort insights and recommendations by relevance
        synthesized["insights"].sort(key=lambda x: x.get("relevance", 0), reverse=True)
        synthesized["recommendations"].sort(key=lambda x: x.get("relevance", 0), reverse=True)
        
        return synthesized
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate a simple similarity score between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple word overlap measure (Jaccard similarity)
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_relevance(self, text: str, query: str) -> float:
        """
        Calculate relevance of a text to the original query.
        
        Args:
            text: Text to evaluate
            query: Original query
            
        Returns:
            Relevance score between 0 and 1
        """
        # Simple word overlap with query
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        
        if not query_words or not text_words:
            return 0.5  # Default middle relevance
            
        # Count matches
        matches = len(query_words.intersection(text_words))
        
        # Calculate base relevance score
        if len(query_words) > 0:
            relevance = matches / len(query_words)
        else:
            relevance = 0.5
            
        return relevance
    
    def get_registered_agents(self) -> List[str]:
        """
        Return a list of all registered agent names.
        
        Returns:
            A list of agent names
        """
        return list(self.agents.keys())
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on all registered agents.
        
        Returns:
            A dictionary with health status information for the orchestrator and all agents
        """
        results = {
            "orchestrator": {
                "status": "healthy",
                "agent_count": len(self.agents),
                "address": self.uagent.address
            },
            "agents": {}
        }
        
        for agent_name, agent in self.agents.items():
            results["agents"][agent_name] = agent.health_check()
            
        return results
    
    def run(self, host: str = "127.0.0.1", port: int = 8000):
        """
        Run the orchestrator service.
        
        Args:
            host: Host address
            port: Port number
        """
        self.logger.info(f"Starting FinChain Orchestrator on {host}:{port}")
        self.uagent.run(host=host, port=port)