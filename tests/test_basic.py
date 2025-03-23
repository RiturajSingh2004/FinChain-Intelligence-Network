"""
Basic tests for the FinChain Intelligence Network components.
"""

import unittest
from fin.orchestrator import FinChainOrchestrator
from agents.blockchain_analyst.blockchain_analyst import BlockchainAnalyst
from agents.ml_investment_strategist.ml_investment_strategist import MLInvestmentStrategist


class TestFinChainComponents(unittest.TestCase):
    """Test cases for FinChain Intelligence Network components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.blockchain_analyst = BlockchainAnalyst()
        self.ml_investment_strategist = MLInvestmentStrategist()
        self.orchestrator = FinChainOrchestrator()
        self.orchestrator.register_agent(self.blockchain_analyst)
        self.orchestrator.register_agent(self.ml_investment_strategist)
    
    def test_blockchain_analyst(self):
        """Test BlockchainAnalyst agent functionality."""
        # Test basic capabilities
        capabilities = self.blockchain_analyst.get_capabilities()
        self.assertTrue(len(capabilities) > 0)
        
        # Test query processing
        query = "Analyze smart contract risks in DeFi protocols"
        response = self.blockchain_analyst.process_query(query)
        
        # Verify response structure
        self.assertIn("insights", response)
        self.assertIn("recommendations", response)
        self.assertIn("confidence", response)
        
        # Verify health check
        health = self.blockchain_analyst.health_check()
        self.assertEqual(health["status"], "healthy")
    
    def test_ml_investment_strategist(self):
        """Test MLInvestmentStrategist agent functionality."""
        # Test basic capabilities
        capabilities = self.ml_investment_strategist.get_capabilities()
        self.assertTrue(len(capabilities) > 0)
        
        # Test query processing
        query = "Recommend investment strategy for an aggressive investor"
        response = self.ml_investment_strategist.process_query(query)
        
        # Verify response structure
        self.assertIn("insights", response)
        self.assertIn("recommendations", response)
        self.assertIn("confidence", response)
        
        # Test asset analysis
        analysis = self.ml_investment_strategist.analyze_asset("BTC", "short")
        self.assertIn("asset", analysis)
        self.assertIn("recommendation", analysis)
    
    def test_orchestrator(self):
        """Test FinChainOrchestrator functionality."""
        # Verify agent registration
        registered_agents = self.orchestrator.get_registered_agents()
        self.assertEqual(len(registered_agents), 2)
        self.assertIn("blockchain_analyst", registered_agents)
        self.assertIn("ml_investment_strategist", registered_agents)
        
        # Test query processing and agent selection
        query1 = "Analyze smart contract security"
        response1 = self.orchestrator.process_query(query1)
        self.assertIn("blockchain_analyst", response1["agents_consulted"])
        
        query2 = "Optimize my investment portfolio"
        response2 = self.orchestrator.process_query(query2)
        self.assertIn("ml_investment_strategist", response2["agents_consulted"])
        
        # Test orchestrator health check
        health = self.orchestrator.health_check()
        self.assertEqual(health["orchestrator"]["status"], "healthy")
        self.assertEqual(health["orchestrator"]["agent_count"], 2)


if __name__ == "__main__":
    unittest.main() 