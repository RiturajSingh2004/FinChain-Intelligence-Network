"""
Comprehensive Test Suite for FinChain Intelligence Network

This module contains unit tests for all major components of the FinChain Intelligence Network,
including the orchestrator, specialized agents, and data integration.
"""

import unittest
import os
import json
import logging
from unittest.mock import MagicMock, patch
from typing import Dict, List, Any

# Disable logging during tests
logging.disable(logging.CRITICAL)

# Import components to test
from fin.orchestrator import FinChainOrchestrator
from fin.base_agent import BaseAgent
from fin.data_integration import DataIntegrationManager

# Import agents
from agents.blockchain_analyst.blockchain_analyst import BlockchainAnalyst
from agents.crypto_economics.crypto_economics import CryptoEconomics
from agents.fintech_navigator.fintech_navigator import FinTechNavigator
from agents.ml_investment_strategist.ml_investment_strategist import MLInvestmentStrategist
from agents.regulatory_compliance.regulatory_compliance import RegulatoryCompliance


class TestBaseAgent(unittest.TestCase):
    """Test cases for the BaseAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a concrete implementation of BaseAgent for testing
        class TestAgent(BaseAgent):
            def process_query(self, query):
                return {"response": f"Processed: {query}"}
            
            def get_capabilities(self):
                return ["capability1", "capability2"]
        
        self.agent = TestAgent("test_agent", "Test Agent Description")
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.name, "test_agent")
        self.assertEqual(self.agent.description, "Test Agent Description")
        self.assertIsNotNone(self.agent.logger)
    
    def test_process_query(self):
        """Test the process_query method."""
        result = self.agent.process_query("test query")
        self.assertEqual(result, {"response": "Processed: test query"})
    
    def test_get_capabilities(self):
        """Test the get_capabilities method."""
        capabilities = self.agent.get_capabilities()
        self.assertEqual(capabilities, ["capability1", "capability2"])
    
    def test_health_check(self):
        """Test the health_check method."""
        health = self.agent.health_check()
        self.assertEqual(health["status"], "healthy")
        self.assertEqual(health["name"], "test_agent")
        self.assertIn("version", health)


class TestOrchestrator(unittest.TestCase):
    """Test cases for the FinChainOrchestrator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = FinChainOrchestrator()
        
        # Create mock agents
        self.mock_agent1 = MagicMock(spec=BaseAgent)
        self.mock_agent1.name = "mock_agent1"
        self.mock_agent1.process_query.return_value = {
            "insights": ["Insight 1", "Insight 2"],
            "recommendations": ["Recommendation 1"],
            "confidence": 0.8
        }
        
        self.mock_agent2 = MagicMock(spec=BaseAgent)
        self.mock_agent2.name = "mock_agent2"
        self.mock_agent2.process_query.return_value = {
            "insights": ["Insight 3"],
            "recommendations": ["Recommendation 2", "Recommendation 3"],
            "confidence": 0.6
        }
        
        # Register mock agents
        self.orchestrator.register_agent(self.mock_agent1)
        self.orchestrator.register_agent(self.mock_agent2)
    
    def test_agent_registration(self):
        """Test agent registration."""
        self.assertIn("mock_agent1", self.orchestrator.agents)
        self.assertIn("mock_agent2", self.orchestrator.agents)
        self.assertEqual(len(self.orchestrator.get_registered_agents()), 2)
    
    def test_process_query(self):
        """Test the process_query method."""
        # Mock the _identify_relevant_agents method to control which agents are selected
        with patch.object(self.orchestrator, '_identify_relevant_agents', 
                         return_value=["mock_agent1", "mock_agent2"]):
            
            response = self.orchestrator.process_query("test query")
            
            # Check that both agents were called
            self.mock_agent1.process_query.assert_called_once_with("test query")
            self.mock_agent2.process_query.assert_called_once_with("test query")
            
            # Verify response structure
            self.assertEqual(response["query"], "test query")
            self.assertListEqual(response["agents_consulted"], ["mock_agent1", "mock_agent2"])
            self.assertEqual(len(response["insights"]), 3)
            self.assertEqual(len(response["recommendations"]), 3)
            
            # Verify confidence calculation
            # Average of agent confidences weighted by agent affinities (1.0 by default)
            expected_confidence = (0.8 + 0.6) / 2
            self.assertAlmostEqual(response["confidence"], expected_confidence, places=2)
    
    def test_identify_relevant_agents(self):
        """Test the _identify_relevant_agents method."""
        # Test with blockchain-related query
        agents = self.orchestrator._identify_relevant_agents("Analyze blockchain transactions")
        self.assertIn("mock_agent1", agents)  # Should match all agents since we're using mocks
        
        # Test with finance-related query
        agents = self.orchestrator._identify_relevant_agents("Recommend investment strategy")
        self.assertIn("mock_agent2", agents)  # Should match all agents since we're using mocks
    
    def test_health_check(self):
        """Test the health_check method."""
        # Setup mock returns for health_check
        self.mock_agent1.health_check.return_value = {"status": "healthy", "name": "mock_agent1"}
        self.mock_agent2.health_check.return_value = {"status": "healthy", "name": "mock_agent2"}
        
        health = self.orchestrator.health_check()
        
        # Verify orchestrator health
        self.assertEqual(health["orchestrator"]["status"], "healthy")
        self.assertEqual(health["orchestrator"]["agent_count"], 2)
        
        # Verify agents health
        self.assertIn("mock_agent1", health["agents"])
        self.assertIn("mock_agent2", health["agents"])
        self.assertEqual(health["agents"]["mock_agent1"]["status"], "healthy")
        self.assertEqual(health["agents"]["mock_agent2"]["status"], "healthy")


class TestBlockchainAnalyst(unittest.TestCase):
    """Test cases for the BlockchainAnalyst agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = BlockchainAnalyst()
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.name, "blockchain_analyst")
        self.assertIn("ethereum", self.agent.supported_networks)
        self.assertIsNotNone(self.agent.alert_thresholds)
    
    def test_process_query(self):
        """Test the process_query method."""
        # Test with a transaction-related query
        response = self.agent.process_query("Analyze recent transactions on Ethereum")
        self.assertIn("insights", response)
        self.assertIn("recommendations", response)
        self.assertIn("confidence", response)
        self.assertGreater(len(response["insights"]), 0)
        
        # Test with a smart contract-related query
        response = self.agent.process_query("Check smart contract security")
        self.assertIn("insights", response)
        self.assertIn("recommendations", response)
        self.assertIn("alerts", response)
    
    def test_analyze_contract(self):
        """Test the analyze_contract method."""
        result = self.agent.analyze_contract("0x1234567890abcdef", "ethereum")
        self.assertIn("risk_score", result)
        self.assertIn("recommendations", result)
    
    def test_monitor_address(self):
        """Test the monitor_address method."""
        result = self.agent.monitor_address("0x1234567890abcdef", "ethereum")
        self.assertEqual(result["status"], "monitoring")
        self.assertEqual(result["address"], "0x1234567890abcdef")
        
        # Test with unsupported network
        result = self.agent.monitor_address("0x1234567890abcdef", "unsupported")
        self.assertIn("error", result)
    
    def test_get_capabilities(self):
        """Test the get_capabilities method."""
        capabilities = self.agent.get_capabilities()
        self.assertGreater(len(capabilities), 0)
        self.assertTrue(any("blockchain" in cap.lower() for cap in capabilities))


class TestMLInvestmentStrategist(unittest.TestCase):
    """Test cases for the MLInvestmentStrategist agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = MLInvestmentStrategist()
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.name, "ml_investment_strategist")
        self.assertIn("conservative", self.agent.risk_profiles)
        self.assertIn("stocks", self.agent.asset_classes)
    
    def test_process_query(self):
        """Test the process_query method."""
        # Test with a prediction-related query
        response = self.agent.process_query("Predict market trends for technology stocks")
        self.assertIn("insights", response)
        self.assertIn("recommendations", response)
        self.assertIn("confidence", response)
        
        # Test with a portfolio-related query
        response = self.agent.process_query("Optimize my portfolio for an aggressive risk profile")
        self.assertIn("insights", response)
        self.assertIn("recommendations", response)
        self.assertIn("portfolio_allocation", response)
        self.assertIsNotNone(response["portfolio_allocation"])
    
    def test_determine_risk_profile(self):
        """Test the _determine_risk_profile method."""
        risk_profile = self.agent._determine_risk_profile("I am a conservative investor")
        self.assertEqual(risk_profile, "conservative")
        
        risk_profile = self.agent._determine_risk_profile("I want aggressive growth")
        self.assertEqual(risk_profile, "aggressive")
        
        risk_profile = self.agent._determine_risk_profile("Just a regular investment")
        self.assertEqual(risk_profile, "moderate")
    
    def test_analyze_asset(self):
        """Test the analyze_asset method."""
        result = self.agent.analyze_asset("BTC", "short")
        self.assertEqual(result["asset"], "BTC")
        self.assertIn("time_horizon", result)
        self.assertIn("recommendation", result)
    
    def test_get_capabilities(self):
        """Test the get_capabilities method."""
        capabilities = self.agent.get_capabilities()
        self.assertGreater(len(capabilities), 0)
        self.assertTrue(any("investment" in cap.lower() for cap in capabilities))


class TestDataIntegration(unittest.TestCase):
    """Test cases for the DataIntegrationManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use a minimal configuration for testing
        self.config = {
            "cache_ttl": 60,  # Short TTL for testing
            "blockchain_data": {
                "supported_networks": ["ethereum", "solana"]
            }
        }
        self.data_manager = DataIntegrationManager(self.config)
    
    def test_initialization(self):
        """Test manager initialization."""
        self.assertEqual(self.data_manager.cache_ttl, 60)
        self.assertIsNotNone(self.data_manager.blockchain_data)
        self.assertIsNotNone(self.data_manager.financial_data)
        self.assertIsNotNone(self.data_manager.news_data)
        self.assertIsNotNone(self.data_manager.regulatory_data)
    
    def test_caching(self):
        """Test the caching mechanism."""
        # Mock fetch function
        mock_fetch = MagicMock(return_value={"data": "test"})
        
        # First call should use the fetch function
        result1 = self.data_manager.get_with_cache("test_key", mock_fetch, "arg1", kwarg1="value1")
        mock_fetch.assert_called_once_with("arg1", kwarg1="value1")
        self.assertEqual(result1, {"data": "test"})
        
        # Reset mock
        mock_fetch.reset_mock()
        
        # Second call should use the cache
        result2 = self.data_manager.get_with_cache("test_key", mock_fetch, "arg1", kwarg1="value1")
        mock_fetch.assert_not_called()
        self.assertEqual(result2, {"data": "test"})
    
    def test_cache_key_generation(self):
        """Test the cache key generation."""
        key1 = self.data_manager.generate_cache_key("prefix", {"param1": "value1", "param2": 2})
        key2 = self.data_manager.generate_cache_key("prefix", {"param1": "value1", "param2": 2})
        key3 = self.data_manager.generate_cache_key("prefix", {"param1": "value1", "param2": 3})
        
        # Same parameters should generate same key
        self.assertEqual(key1, key2)
        
        # Different parameters should generate different keys
        self.assertNotEqual(key1, key3)


class TestGradioInterface(unittest.TestCase):
    """Test cases for the Gradio web interface."""
    
    @patch('gradio.Blocks')
    @patch('fin.orchestrator.FinChainOrchestrator')
    def test_process_query(self, mock_orchestrator_class, mock_blocks_class):
        """Test the process_query function in the Gradio interface."""
        # Import the function to test - assumes gradio_app.py exists
        import sys
        sys.path.append('.')  # Add current directory to path
        
        try:
            from gradio_app import process_query, initialize_fin_network
        except ImportError:
            # Skip test if module not found
            self.skipTest("gradio_app.py not found")
            return
        
        # Set up mocks
        mock_orchestrator = mock_orchestrator_class.return_value
        mock_orchestrator.process_query.return_value = {
            "query": "test query",
            "agents_consulted": ["agent1", "agent2"],
            "insights": [{"content": "Insight 1", "source": "agent1"}, {"content": "Insight 2", "source": "agent2"}],
            "recommendations": [{"content": "Recommendation", "source": "agent1"}],
            "confidence": 0.75
        }
        
        # Patch initialize_fin_network to return our mock
        with patch('gradio_app.initialize_fin_network', return_value=mock_orchestrator):
            # Call the function
            html_response, json_response = process_query("test query", "Conservative")
            
            # Verify orchestrator was called with correct query
            mock_orchestrator.process_query.assert_called_once_with("test query (Risk Profile: Conservative)")
            
            # Check that we got HTML and JSON responses
            self.assertIsInstance(html_response, str)
            self.assertIsInstance(json_response, str)
            self.assertIn("FinChain Intelligence Network Analysis", html_response)
            self.assertIn("test query", json_response)


if __name__ == '__main__':
    unittest.main()