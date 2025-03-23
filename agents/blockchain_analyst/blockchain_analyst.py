"""
BlockchainAnalyst Agent Implementation with uAgents Framework Integration
"""

import logging
from typing import Dict, List, Any, Optional
import json

from fin.base_agent import BaseAgent
from uagents import Model


class BlockchainTransactionRequest(Model):
    """Model for blockchain transaction analysis requests."""
    address: str
    network: str = "ethereum"


class BlockchainContractRequest(Model):
    """Model for smart contract analysis requests."""
    contract_address: str
    network: str = "ethereum"


class BlockchainAnalyst(BaseAgent):
    """
    The BlockchainAnalyst agent is responsible for:
    1. Monitoring blockchain transactions across multiple networks
    2. Analyzing smart contract activity and identifying potential risks
    3. Providing real-time alerts on suspicious transactions or market anomalies
    
    Now integrated with Fetch.ai uAgents framework.
    """
    
    def __init__(self, name: str = "blockchain_analyst", 
                 description: str = "Monitors blockchain transactions and analyzes smart contracts",
                 seed: Optional[str] = None, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the BlockchainAnalyst agent.
        
        Args:
            name: The name of the agent
            description: A brief description of the agent's capabilities
            seed: Optional seed phrase for the uAgent
            config: Optional configuration parameters
        """
        super().__init__(name, description, seed, config)
        self.supported_networks = self.config.get("supported_networks", 
                                                 ["ethereum", "solana", "avalanche", "polygon"])
        self.transaction_cache = {}
        self.alert_thresholds = self.config.get("alert_thresholds", {
            "large_transaction": 1000,  # ETH/SOL amount that would trigger a large transaction alert
            "suspicious_pattern": 0.85,  # Threshold for suspicious pattern detection
            "contract_risk": 0.7,       # Threshold for smart contract risk assessment
        })
        
        # Register additional message handlers for specific blockchain requests
        @self.protocol.on_message(model=BlockchainTransactionRequest)
        async def handle_transaction_request(ctx, sender, msg):
            self.logger.info(f"Received transaction analysis request from {sender}: {msg.address}")
            
            # Process the transaction analysis
            result = self.monitor_address(msg.address, msg.network)
            
            # Send response back
            await ctx.send(sender, {
                "status": "completed",
                "result": result,
                "address": msg.address,
                "network": msg.network
            })
            
        @self.protocol.on_message(model=BlockchainContractRequest)
        async def handle_contract_request(ctx, sender, msg):
            self.logger.info(f"Received contract analysis request from {sender}: {msg.contract_address}")
            
            # Process the contract analysis
            result = self.analyze_contract(msg.contract_address, msg.network)
            
            # Send response back
            await ctx.send(sender, {
                "status": "completed",
                "result": result,
                "contract_address": msg.contract_address,
                "network": msg.network
            })
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a blockchain-related query.
        
        Args:
            query: The user's query string
            
        Returns:
            A dictionary containing insights and recommendations
        """
        self.logger.info(f"Processing blockchain query: {query}")
        
        # Here we'd use NLP to parse the query and determine the specific blockchain analysis needed
        # For now, we'll use a simplified approach
        
        query_lower = query.lower()
        response = {
            "insights": [],
            "recommendations": [],
            "alerts": [],
            "confidence": 0.0
        }
        
        # Check if query is about transaction monitoring
        if any(term in query_lower for term in ["transaction", "transfer", "wallet", "address"]):
            self._analyze_transactions(query, response)
            
        # Check if query is about smart contract analysis
        if any(term in query_lower for term in ["smart contract", "contract", "code", "audit"]):
            self._analyze_smart_contracts(query, response)
            
        # Check if query is about market anomalies
        if any(term in query_lower for term in ["anomaly", "suspicious", "unusual", "fraud"]):
            self._detect_anomalies(query, response)
            
        # Set confidence based on how well we could answer the query
        response["confidence"] = min(0.9, 0.3 + 0.2 * len(response["insights"]) + 0.1 * len(response["recommendations"]))
        
        return response
    
    def _analyze_transactions(self, query: str, response: Dict[str, Any]):
        """
        Analyze blockchain transactions based on the query.
        
        Args:
            query: The user's query
            response: The response dictionary to update
        """
        # In a real implementation, this would query blockchain nodes for transaction data
        # For now, we'll provide sample insights and recommendations
        
        response["insights"].append("Recent transaction volume on Ethereum has increased by 15% in the last 24 hours")
        response["insights"].append("Average gas prices are currently at 25 gwei, which is lower than the weekly average")
        response["recommendations"].append("Consider batching transactions to reduce gas costs during this period of lower fees")
        
    def _analyze_smart_contracts(self, query: str, response: Dict[str, Any]):
        """
        Analyze smart contracts based on the query.
        
        Args:
            query: The user's query
            response: The response dictionary to update
        """
        # In a real implementation, this would analyze smart contract code and execution patterns
        
        response["insights"].append("The smart contract has passed basic security checks but has not undergone a formal audit")
        response["insights"].append("The contract follows standard ERC-20 implementation patterns with minor modifications")
        response["alerts"].append("Missing input validation in the transfer function could pose a security risk")
        response["recommendations"].append("Recommend a formal security audit before significant funds are committed")
        
    def _detect_anomalies(self, query: str, response: Dict[str, Any]):
        """
        Detect anomalies in blockchain data based on the query.
        
        Args:
            query: The user's query
            response: The response dictionary to update
        """
        # In a real implementation, this would use ML models to detect anomalies
        
        response["insights"].append("No major anomalies detected in recent transaction patterns")
        response["insights"].append("Wallet clustering analysis shows normal distribution of token holdings")
        response["recommendations"].append("Set up automated monitoring for transactions exceeding 100 ETH to detect potential market manipulation")
        
    def get_capabilities(self) -> List[str]:
        """
        Return a list of the agent's capabilities.
        
        Returns:
            A list of capability descriptions
        """
        return [
            "Monitor blockchain transactions across multiple networks",
            "Analyze smart contract code for security vulnerabilities",
            "Detect anomalies in transaction patterns",
            "Provide real-time alerts for suspicious activities",
            "Track gas prices and network congestion",
            "Assess liquidity and trading volume across exchanges"
        ]
    
    def monitor_address(self, address: str, network: str = "ethereum") -> Dict[str, Any]:
        """
        Monitor a specific blockchain address for activity.
        
        Args:
            address: The blockchain address to monitor
            network: The blockchain network (e.g., "ethereum", "solana")
            
        Returns:
            A dictionary containing monitoring results
        """
        # In a real implementation, this would set up monitoring for the address
        
        if network not in self.supported_networks:
            return {"error": f"Unsupported network: {network}"}
        
        self.logger.info(f"Setting up monitoring for address {address} on {network}")
        
        return {
            "status": "monitoring",
            "address": address,
            "network": network,
            "alerts_configured": ["large_transactions", "suspicious_patterns"]
        }
    
    def analyze_contract(self, contract_address: str, network: str = "ethereum") -> Dict[str, Any]:
        """
        Analyze a smart contract for security vulnerabilities and risks.
        
        Args:
            contract_address: The address of the smart contract
            network: The blockchain network
            
        Returns:
            A dictionary containing analysis results
        """
        # In a real implementation, this would analyze the contract code
        
        if network not in self.supported_networks:
            return {"error": f"Unsupported network: {network}"}
        
        self.logger.info(f"Analyzing contract {contract_address} on {network}")
        
        return {
            "risk_score": 0.45,
            "vulnerability_count": 0,
            "warnings": ["High gas consumption in fallback function"],
            "recommendations": ["Optimize storage usage to reduce gas costs"]
        }


# Standalone execution for running the agent as a service
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run the agent
    agent = BlockchainAnalyst()
    
    # Run the agent on localhost:8000
    agent.run()