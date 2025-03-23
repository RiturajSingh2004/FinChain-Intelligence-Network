#!/usr/bin/env python3
"""
FinChain Intelligence Network (FIN) - Configuration Script

This script provides utilities for configuring and initializing the FinChain Intelligence Network,
including setting up logging, loading configuration from files, and registering agents.
"""

import os
import json
import yaml
import logging
import argparse
from typing import Dict, List, Any, Optional

from fin.orchestrator import FinChainOrchestrator
from agents.blockchain_analyst.blockchain_analyst import BlockchainAnalyst
from agents.crypto_economics.crypto_economics import CryptoEconomics
from agents.fintech_navigator.fintech_navigator import FinTechNavigator
from agents.ml_investment_strategist.ml_investment_strategist import MLInvestmentStrategist
from agents.regulatory_compliance.regulatory_compliance import RegulatoryCompliance


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    log_config = {
        'level': numeric_level,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }
    
    if log_file:
        log_config['filename'] = log_file
        log_config['filemode'] = 'a'  # Append mode
    
    logging.basicConfig(**log_config)
    
    # Create a logger for this script
    logger = logging.getLogger("fin.config")
    logger.info(f"Logging initialized at level {log_level}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON or YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    logger = logging.getLogger("fin.config")
    logger.info(f"Loading configuration from {config_path}")
    
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file {config_path} not found. Using default configuration.")
        return {}
    
    try:
        file_extension = os.path.splitext(config_path)[1].lower()
        
        if file_extension == ".json":
            with open(config_path, 'r') as file:
                config = json.load(file)
        elif file_extension in [".yaml", ".yml"]:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        else:
            logger.warning(f"Unsupported configuration file format: {file_extension}")
            config = {}
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        config = {}
    
    return config


def initialize_fin_network(config: Optional[Dict[str, Any]] = None) -> FinChainOrchestrator:
    """
    Initialize the FinChain Intelligence Network with all specialized agents.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured FinChainOrchestrator instance
    """
    logger = logging.getLogger("fin.config")
    
    # Default configuration
    default_config = {
        "use_nlp": True,
        "enable_agent_affinity": True,
        "default_agents": ["all"]
    }
    
    # Merge with provided config
    config = {**default_config, **(config or {})}
    
    # Create the orchestrator
    logger.info("Creating FinChain Orchestrator")
    orchestrator = FinChainOrchestrator(config)
    
    # Determine which agents to register
    agents_to_register = config.get("default_agents", ["all"])
    register_all = "all" in agents_to_register
    
    # Create and register specialized agents
    if register_all or "blockchain_analyst" in agents_to_register:
        agent_config = config.get("blockchain_analyst", {})
        logger.info("Creating BlockchainAnalyst agent")
        blockchain_analyst = BlockchainAnalyst(agent_config)
        orchestrator.register_agent(blockchain_analyst)
    
    if register_all or "crypto_economics" in agents_to_register:
        agent_config = config.get("crypto_economics", {})
        logger.info("Creating CryptoEconomics agent")
        crypto_economics = CryptoEconomics(agent_config)
        orchestrator.register_agent(crypto_economics)
    
    if register_all or "fintech_navigator" in agents_to_register:
        agent_config = config.get("fintech_navigator", {})
        logger.info("Creating FinTechNavigator agent")
        fintech_navigator = FinTechNavigator(agent_config)
        orchestrator.register_agent(fintech_navigator)
    
    if register_all or "ml_investment_strategist" in agents_to_register:
        agent_config = config.get("ml_investment_strategist", {})
        logger.info("Creating MLInvestmentStrategist agent")
        ml_investment_strategist = MLInvestmentStrategist(agent_config)
        orchestrator.register_agent(ml_investment_strategist)
    
    if register_all or "regulatory_compliance" in agents_to_register:
        agent_config = config.get("regulatory_compliance", {})
        logger.info("Creating RegulatoryCompliance agent")
        regulatory_compliance = RegulatoryCompliance(agent_config)
        orchestrator.register_agent(regulatory_compliance)
    
    logger.info(f"Registered {len(orchestrator.get_registered_agents())} agents")
    return orchestrator


def save_default_config(output_path: str) -> None:
    """
    Save a default configuration file that users can modify.
    
    Args:
        output_path: Path to save the configuration file
    """
    logger = logging.getLogger("fin.config")
    
    default_config = {
        "use_nlp": True,
        "enable_agent_affinity": True,
        "default_agents": ["all"],
        "logging": {
            "level": "INFO",
            "file": "finchain.log"
        },
        "blockchain_analyst": {
            "supported_networks": ["ethereum", "solana", "avalanche", "polygon"]
        },
        "crypto_economics": {
            "defi_protocols_to_track": ["uniswap", "aave", "curve", "compound"]
        },
        "fintech_navigator": {
            "news_sources": ["bloomberg", "reuters", "coindesk"]
        },
        "ml_investment_strategist": {
            "risk_profiles": ["conservative", "moderate", "aggressive"]
        },
        "regulatory_compliance": {
            "jurisdictions_to_track": ["us", "eu", "uk", "sg", "in"]
        }
    }
    
    try:
        file_extension = os.path.splitext(output_path)[1].lower()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        if file_extension == ".json":
            with open(output_path, 'w') as file:
                json.dump(default_config, file, indent=2)
        elif file_extension in [".yaml", ".yml"]:
            with open(output_path, 'w') as file:
                yaml.dump(default_config, file, default_flow_style=False)
        else:
            logger.warning(f"Unsupported configuration file format: {file_extension}")
            return
            
        logger.info(f"Default configuration saved to {output_path}")
    
    except Exception as e:
        logger.error(f"Error saving default configuration: {e}")


def main():
    """Main entry point for the configuration script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FinChain Intelligence Network Configuration")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--log-file", type=str, help="Path to log file")
    parser.add_argument("--generate-config", type=str, help="Generate default configuration file at specified path")
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level, args.log_file)
    
    # Generate default configuration if requested
    if args.generate_config:
        save_default_config(args.generate_config)
        return
    
    # Load configuration
    config = load_config(args.config) if args.config else {}
    
    # Initialize the FIN network
    orchestrator = initialize_fin_network(config)
    
    # Display system information
    logger = logging.getLogger("fin.config")
    logger.info(f"FinChain Intelligence Network initialized with {len(orchestrator.get_registered_agents())} agents")
    logger.info(f"Registered Agents: {', '.join(orchestrator.get_registered_agents())}")


if __name__ == "__main__":
    main()