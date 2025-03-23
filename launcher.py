#!/usr/bin/env python3
"""
FinChain Intelligence Network (FIN) Launcher

This script launches the entire FinChain Intelligence Network system,
including all specialized agents and the web interface.
"""

import os
import sys
import logging
import argparse
import threading
import time
import yaml
import signal
import subprocess
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("fin.launcher")


def load_config(config_path: str = "config/default_config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        else:
            logger.warning(f"Configuration file {config_path} not found. Using default configuration.")
            return {}
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}


def launch_agent(agent_module: str, agent_class: str, name: str, port: int) -> subprocess.Popen:
    """
    Launch an agent as a separate process.
    
    Args:
        agent_module: Python module path to agent
        agent_class: Agent class name
        name: Agent name
        port: Port number for the agent to listen on
        
    Returns:
        The process object
    """
    logger.info(f"Launching agent {name} on port {port}")
    
    # Prepare the command to run the agent
    cmd = [
        sys.executable, "-c",
        f"from {agent_module} import {agent_class}; "
        f"agent = {agent_class}(name='{name}'); "
        f"agent.run(host='127.0.0.1', port={port})"
    ]
    
    # Launch the agent as a separate process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait a moment to ensure the agent has started
    time.sleep(1)
    
    if process.poll() is not None:
        # Process has already terminated
        stdout, stderr = process.communicate()
        logger.error(f"Failed to start agent {name}: {stderr}")
        raise RuntimeError(f"Failed to start agent {name}")
    
    logger.info(f"Agent {name} started successfully")
    return process


def launch_web_interface(port: int = 7860) -> subprocess.Popen:
    """
    Launch the Gradio web interface.
    
    Args:
        port: Port number for the web interface
        
    Returns:
        The process object
    """
    logger.info(f"Launching web interface on port {port}")
    
    # Prepare the command to run the web interface
    cmd = [
        sys.executable, "gradio_app.py"
    ]
    
    # Set environment variable for the port
    env = os.environ.copy()
    env["GRADIO_SERVER_PORT"] = str(port)
    
    # Launch the web interface as a separate process
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait a moment to ensure the web interface has started
    time.sleep(2)
    
    if process.poll() is not None:
        # Process has already terminated
        stdout, stderr = process.communicate()
        logger.error(f"Failed to start web interface: {stderr}")
        raise RuntimeError("Failed to start web interface")
    
    logger.info(f"Web interface started successfully on port {port}")
    return process


def main():
    """Main entry point for the launcher."""
    parser = argparse.ArgumentParser(description="FinChain Intelligence Network Launcher")
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--web-port", type=int, default=7860,
                       help="Port for the web interface")
    parser.add_argument("--agent-base-port", type=int, default=8000,
                       help="Base port for agent services (each agent will use a different port)")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # List of processes to track
    processes = []
    
    try:
        # Launch agents
        agents_config = [
            ("agents.blockchain_analyst.blockchain_analyst", "BlockchainAnalyst", "blockchain_analyst"),
            ("agents.crypto_economics.crypto_economics", "CryptoEconomics", "crypto_economics"),
            ("agents.fintech_navigator.fintech_navigator", "FinTechNavigator", "fintech_navigator"),
            ("agents.ml_investment_strategist.ml_investment_strategist", "MLInvestmentStrategist", "ml_investment_strategist"),
            ("agents.regulatory_compliance.regulatory_compliance", "RegulatoryCompliance", "regulatory_compliance")
        ]
        
        for i, (module, cls, name) in enumerate(agents_config):
            port = args.agent_base_port + i
            process = launch_agent(module, cls, name, port)
            processes.append(process)
        
        # Launch web interface
        web_process = launch_web_interface(args.web_port)
        processes.append(web_process)
        
        logger.info("All components launched successfully!")
        logger.info(f"Access the web interface at http://localhost:{args.web_port}")
        
        # Add a URL to the Fetch.ai explorer for debugging (in a real deployment)
        logger.info("System is running. Press Ctrl+C to stop.")
        
        # Wait for Ctrl+C - using a simple loop instead of signal.pause()
        # This is more cross-platform compatible
        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received stop signal. Shutting down...")
        
    except KeyboardInterrupt:
        logger.info("Received stop signal. Shutting down...")
    except Exception as e:
        logger.error(f"Error during launch: {e}")
    finally:
        # Terminate all processes
        for process in processes:
            if process.poll() is None:  # Process is still running
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        logger.info("All components stopped. Exiting.")


if __name__ == "__main__":
    main()