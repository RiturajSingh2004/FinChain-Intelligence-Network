#!/usr/bin/env python3
"""
FinChain Intelligence Network (FIN) - Gradio Web Interface with uAgents Integration

This script creates a user-friendly web interface for the FinChain Intelligence Network
using Gradio, allowing users to interact with the multi-agent system through a browser.
The agents are implemented using the Fetch.ai uAgents framework.
"""

import gradio as gr
import logging
import json
import os
import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
import threading

from fin.orchestrator import FinChainOrchestrator
from agents.blockchain_analyst.blockchain_analyst import BlockchainAnalyst
from agents.crypto_economics.crypto_economics import CryptoEconomics
from agents.fintech_navigator.fintech_navigator import FinTechNavigator
from agents.ml_investment_strategist.ml_investment_strategist import MLInvestmentStrategist
from agents.regulatory_compliance.regulatory_compliance import RegulatoryCompliance


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("fin.gradio_interface")


def initialize_fin_network() -> FinChainOrchestrator:
    """
    Initialize the FinChain Intelligence Network with all specialized agents.
    
    Returns:
        Configured FinChainOrchestrator instance
    """
    # Create the orchestrator
    orchestrator = FinChainOrchestrator()
    
    # Create and register all specialized agents
    blockchain_analyst = BlockchainAnalyst(
        name="blockchain_analyst",
        description="Monitors blockchain transactions and analyzes smart contracts"
    )
    
    crypto_economics = CryptoEconomics(
        name="crypto_economics",
        description="Models tokenomics and provides insights on token valuation and DeFi protocols"
    )
    
    fintech_navigator = FinTechNavigator(
        name="fintech_navigator",
        description="Tracks fintech trends, regulations, and market movements"
    )
    
    ml_investment_strategist = MLInvestmentStrategist(
        name="ml_investment_strategist",
        description="Uses machine learning for investment strategy and portfolio optimization"
    )
    
    regulatory_compliance = RegulatoryCompliance(
        name="regulatory_compliance",
        description="Tracks financial and blockchain regulations and assesses compliance risks"
    )
    
    # Register agents with the orchestrator
    orchestrator.register_agent(blockchain_analyst)
    orchestrator.register_agent(crypto_economics)
    orchestrator.register_agent(fintech_navigator)
    orchestrator.register_agent(ml_investment_strategist)
    orchestrator.register_agent(regulatory_compliance)
    
    return orchestrator


def generate_visualizations(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate visualizations based on the response data.
    
    Args:
        response: The orchestrator response dictionary
        
    Returns:
        Dictionary with visualization file paths or data
    """
    visualizations = {}
    
    # Check if there's portfolio allocation data from ML Investment Strategist
    if "portfolio_allocation" in response and response["portfolio_allocation"]:
        portfolio_data = response["portfolio_allocation"]
        
        # Create a pie chart for portfolio allocation
        plt.figure(figsize=(10, 6))
        plt.pie(
            portfolio_data.values(), 
            labels=portfolio_data.keys(), 
            autopct='%1.1f%%', 
            startangle=90,
            colors=plt.cm.tab10.colors
        )
        plt.axis('equal')
        plt.title('Recommended Portfolio Allocation')
        
        # Save the figure
        portfolio_viz_path = os.path.join(os.getcwd(), "temp_portfolio_viz.png")
        plt.savefig(portfolio_viz_path)
        plt.close()
        
        visualizations["portfolio_allocation"] = portfolio_viz_path
    
    # Check for blockchain transaction data
    if any("transaction" in insight["content"].lower() for insight in response.get("insights", [])):
        # Create a sample blockchain transaction visualization
        # This would be replaced with actual data in a production system
        fig = make_subplots(rows=1, cols=1)
        
        # Sample data - in production this would come from actual blockchain data
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        tx_volume = np.random.normal(loc=1000, scale=200, size=30).cumsum()
        
        fig.add_trace(
            go.Scatter(x=dates, y=tx_volume, mode='lines+markers', name='Transaction Volume'),
            row=1, col=1
        )
        
        fig.update_layout(
            title_text="Blockchain Transaction Volume (30-Day Trend)",
            xaxis_title="Date",
            yaxis_title="Volume",
            height=500
        )
        
        # Save the figure
        blockchain_viz_path = os.path.join(os.getcwd(), "temp_blockchain_viz.html")
        fig.write_html(blockchain_viz_path)
        
        visualizations["blockchain_transactions"] = blockchain_viz_path
    
    return visualizations


def format_response_html(response: Dict[str, Any], visualizations: Dict[str, Any]) -> str:
    """
    Format the orchestrator response as HTML for display in the Gradio interface.
    
    Args:
        response: The response dictionary from the orchestrator
        visualizations: Dictionary with visualization data
        
    Returns:
        HTML formatted string for display
    """
    html = [
        "<div style='max-width:800px; margin:0 auto; font-family:Arial, sans-serif;'>",
        "<h2>FinChain Intelligence Network Analysis</h2>"
    ]
    
    # Add agents consulted section
    html.append("<div style='background:#f5f5f5; padding:15px; border-radius:5px; margin-bottom:20px;'>")
    html.append("<h3>Agents Consulted</h3>")
    agents = response.get("agents_consulted", [])
    html.append("<ul>")
    for agent in agents:
        # Convert agent name to a more readable format
        agent_name = agent.replace("_", " ").title()
        html.append(f"<li>{agent_name}</li>")
    html.append("</ul>")
    html.append(f"<p><strong>Confidence Score:</strong> {response.get('confidence', 0.0):.2f}</p>")
    html.append("</div>")
    
    # Add insights section
    html.append("<div style='margin-bottom:20px;'>")
    html.append("<h3>Key Insights</h3>")
    insights = response.get("insights", [])
    html.append("<ul>")
    for insight in insights:
        source = insight["source"].replace("_", " ").title()
        html.append(f"<li><strong>{source}:</strong> {insight['content']}</li>")
    html.append("</ul>")
    html.append("</div>")
    
    # Add recommendations section
    html.append("<div style='margin-bottom:20px;'>")
    html.append("<h3>Recommendations</h3>")
    recommendations = response.get("recommendations", [])
    html.append("<ul>")
    for rec in recommendations:
        source = rec["source"].replace("_", " ").title()
        html.append(f"<li><strong>{source}:</strong> {rec['content']}</li>")
    html.append("</ul>")
    html.append("</div>")
    
    # Add portfolio visualization if available
    if "portfolio_allocation" in visualizations:
        html.append("<div style='text-align:center; margin:20px 0;'>")
        html.append("<h3>Portfolio Allocation</h3>")
        html.append("<img src='file/" + visualizations["portfolio_allocation"] + "' style='max-width:100%; height:auto;'>")
        html.append("</div>")
    
    # Add blockchain visualization if available
    if "blockchain_transactions" in visualizations:
        html.append("<div style='text-align:center; margin:20px 0;'>")
        html.append("<h3>Blockchain Transaction Analysis</h3>")
        html.append("<iframe src='file/" + visualizations["blockchain_transactions"] + "' style='width:100%; height:500px; border:none;'></iframe>")
        html.append("</div>")
    
    html.append("</div>")
    
    return "".join(html)


def process_query(query: str, risk_profile: str) -> Tuple[str, str]:
    """
    Process a user query through the FIN network and format the results.
    
    Args:
        query: The user's query string
        risk_profile: The user's risk profile (conservative, moderate, aggressive)
        
    Returns:
        Tuple of (formatted HTML response, raw JSON response)
    """
    try:
        # Get the orchestrator instance
        orchestrator = initialize_fin_network()
        
        # Check if we're dealing with an async process_query
        if hasattr(orchestrator, 'process_query_async'):
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Call the async method and run it to completion
            response = loop.run_until_complete(
                orchestrator.process_query_async(query, risk_profile)
            )
            loop.close()
        else:
            # Use the synchronous version
            response = orchestrator.process_query(query, risk_profile)
        
        # Generate visualizations
        visualizations = generate_visualizations(response)
        
        # Format the response as HTML
        html_response = format_response_html(response, visualizations)
        
        # Format the raw JSON response
        raw_json = json.dumps(response, indent=2)
        
        return html_response, raw_json
        
    except Exception as e:
        # Your existing error handling code...
        error_html = f"""
        <div style="color: red; border: 1px solid red; padding: 10px; border-radius: 5px;">
            <h3>Error Processing Query</h3>
            <p><strong>Error message:</strong> {str(e)}</p>
            <p><strong>Query:</strong> {query}</p>
            <p><strong>Risk profile:</strong> {risk_profile}</p>
            <h4>Debugging Information:</h4>
            <p>Check the application logs for more detailed error information.</p>
        </div>
        """
        
        error_json = json.dumps({
            "error": str(e),
            "query": query,
            "risk_profile": risk_profile
        }, indent=2)
        
        return error_html, error_json

def create_web_interface():
    """Create and launch the Gradio web interface."""
    with gr.Blocks(title="FinChain Intelligence Network", css="footer {visibility: hidden}") as interface:
        gr.Markdown(
            """
            # FinChain Intelligence Network (FIN)
            
            An advanced multi-agent system that combines specialized AI agents to deliver comprehensive financial 
            and blockchain intelligence. The system leverages machine learning, natural language processing, 
            and blockchain technology to provide real-time analysis, personalized recommendations, 
            and regulatory compliance guidance.
            
            ## Specialized AI Agents
            
            - **BlockchainAnalyst**: Monitors blockchain transactions and analyzes smart contracts
            - **CryptoEconomics**: Models tokenomics and provides insights on DeFi protocols
            - **FinTechNavigator**: Tracks fintech trends, regulations, and market movements
            - **MLInvestmentStrategist**: Uses machine learning for investment strategy
            - **RegulatoryCompliance**: Tracks financial and blockchain regulations
            """
        )
        
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    label="Enter your query",
                    placeholder="Example: Analyze the investment potential of Ethereum DeFi projects",
                    lines=3
                )
                
                risk_profile = gr.Radio(
                    ["Not specified", "Conservative", "Moderate", "Aggressive"],
                    label="Investment Risk Profile",
                    value="Not specified"
                )
                
                submit_btn = gr.Button("Submit Query", variant="primary")
            
            with gr.Column():
                gr.Markdown("### Example Queries")
                example_queries = gr.Examples(
                    examples=[
                        ["Analyze smart contract security risks in DeFi protocols"],
                        ["What are the current fintech trends in payment systems?"],
                        ["Recommend an investment strategy for blockchain technologies"],
                        ["Evaluate the tokenomics of a new crypto project"],
                        ["What are the key regulatory considerations for a DeFi platform in the EU?"]
                    ],
                    inputs=query_input
                )
        
        with gr.Row():
            with gr.Tab("Analysis Results"):
                html_output = gr.HTML()
            with gr.Tab("Raw JSON Response"):
                json_output = gr.Code(language="json")
        
        submit_btn.click(
            fn=process_query,
            inputs=[query_input, risk_profile],
            outputs=[html_output, json_output]
        )
    
    return interface


if __name__ == "__main__":
    # Create and launch the interface
    interface = create_web_interface()
    interface.launch(share=True)