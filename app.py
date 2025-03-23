#!/usr/bin/env python3
"""
FinChain Intelligence Network (FIN) - Demo Application

This script demonstrates the capabilities of the FinChain Intelligence Network
by setting up the orchestrator and specialized agents and processing sample queries.
"""

import logging
import json
import argparse
from typing import Dict, Any

from fin.orchestrator import FinChainOrchestrator
from agents.blockchain_analyst.blockchain_analyst import BlockchainAnalyst
from agents.ml_investment_strategist.ml_investment_strategist import MLInvestmentStrategist


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def initialize_fin_network() -> FinChainOrchestrator:
    """
    Initialize the FinChain Intelligence Network with all specialized agents.
    
    Returns:
        Configured FinChainOrchestrator instance
    """
    # Create the orchestrator
    orchestrator = FinChainOrchestrator()
    
    # Create and register specialized agents
    blockchain_analyst = BlockchainAnalyst()
    ml_investment_strategist = MLInvestmentStrategist()
    
    # Register agents with the orchestrator
    orchestrator.register_agent(blockchain_analyst)
    orchestrator.register_agent(ml_investment_strategist)
    
    # Additional agents would be created and registered here
    # For a full implementation, create and register all 5 specialized agents
    
    return orchestrator


def format_response(response: Dict[str, Any]) -> str:
    """
    Format the orchestrator response for display.
    
    Args:
        response: The response dictionary from the orchestrator
        
    Returns:
        Formatted string for display
    """
    output = []
    output.append(f"Query: {response.get('query', 'N/A')}")
    output.append(f"Agents consulted: {', '.join(response.get('agents_consulted', []))}")
    output.append(f"Confidence: {response.get('confidence', 0.0):.2f}")
    output.append("\nInsights:")
    
    for idx, insight in enumerate(response.get('insights', []), 1):
        output.append(f"  {idx}. {insight['content']} (Source: {insight['source']})")
    
    output.append("\nRecommendations:")
    for idx, rec in enumerate(response.get('recommendations', []), 1):
        output.append(f"  {idx}. {rec['content']} (Source: {rec['source']})")
    
    return "\n".join(output)


def process_user_query(orchestrator: FinChainOrchestrator, query: str) -> None:
    """
    Process a user query and display the response.
    
    Args:
        orchestrator: The configured orchestrator
        query: The user's query string
    """
    # Process the query through the orchestrator
    response = orchestrator.process_query(query)
    
    # Format and display the response
    formatted_response = format_response(response)
    print("\n" + "="*80)
    print(formatted_response)
    print("="*80 + "\n")


def main():
    """Main entry point for the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FinChain Intelligence Network Demo")
    parser.add_argument("--query", type=str, help="Query to process")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Initialize the FIN network
    orchestrator = initialize_fin_network()
    
    # Display system information
    print("\nFinChain Intelligence Network (FIN) Demo")
    print("="*50)
    print(f"Registered Agents: {', '.join(orchestrator.get_registered_agents())}")
    print("="*50 + "\n")
    
    # Process command line query if provided
    if args.query:
        process_user_query(orchestrator, args.query)
    
    # Interactive mode
    if args.interactive or not args.query:
        print("Interactive Mode: Enter queries or 'exit' to quit.\n")
        while True:
            query = input("Enter your query: ")
            if query.lower() in ["exit", "quit", "q"]:
                break
            
            process_user_query(orchestrator, query)


if __name__ == "__main__":
    main() 