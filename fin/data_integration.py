"""
FinChain Intelligence Network (FIN) - Data Integration Module

This module provides integration with external APIs and data sources for the specialized agents in the FIN network.
It handles authentication, rate limiting, data formatting, and caching.
"""

import os
import json
import logging
import time
import hashlib
import requests
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from functools import wraps


class APIError(Exception):
    """Exception raised for API errors."""
    pass


class DataIntegrationManager:
    """
    Manager for all external data integrations.
    Provides a unified interface for accessing various data sources.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data integration manager.
        
        Args:
            config: Optional configuration parameters
        """
        self.logger = logging.getLogger("fin.data_integration")
        self.config = config or {}
        self.cache = {}
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # Default 1 hour
        self.api_rate_limits = {}
        self.api_last_called = {}
        
        # Initialize integration modules
        self.blockchain_data = BlockchainDataIntegration(self, self.config.get("blockchain_data", {}))
        self.financial_data = FinancialDataIntegration(self, self.config.get("financial_data", {}))
        self.news_data = NewsDataIntegration(self, self.config.get("news_data", {}))
        self.regulatory_data = RegulatoryDataIntegration(self, self.config.get("regulatory_data", {}))
    
    def get_with_cache(self, key: str, fetch_function: Callable, *args, **kwargs) -> Any:
        """
        Get data with caching support.
        
        Args:
            key: Cache key
            fetch_function: Function to fetch data if not in cache
            args: Arguments to pass to fetch_function
            kwargs: Keyword arguments to pass to fetch_function
            
        Returns:
            Fetched data (from cache if available and valid)
        """
        now = time.time()
        
        # Check cache
        if key in self.cache:
            cached_time, cached_data = self.cache[key]
            if now - cached_time < self.cache_ttl:
                self.logger.debug(f"Cache hit for {key}")
                return cached_data
        
        # Fetch fresh data
        self.logger.debug(f"Cache miss for {key}, fetching fresh data")
        data = fetch_function(*args, **kwargs)
        
        # Update cache
        self.cache[key] = (now, data)
        
        return data
    
    def rate_limited(self, api_name: str, max_calls: int, period: int):
        """
        Decorator for rate-limiting API calls.
        
        Args:
            api_name: Name of the API
            max_calls: Maximum number of calls allowed in the period
            period: Time period in seconds
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                now = time.time()
                
                # Initialize rate limit tracking if not already done
                if api_name not in self.api_rate_limits:
                    self.api_rate_limits[api_name] = {
                        "max_calls": max_calls,
                        "period": period,
                        "calls": []
                    }
                
                # Get rate limit settings
                limits = self.api_rate_limits[api_name]
                
                # Remove old calls from tracking
                limits["calls"] = [call_time for call_time in limits["calls"] if now - call_time < limits["period"]]
                
                # Check if we're over the limit
                if len(limits["calls"]) >= limits["max_calls"]:
                    oldest_call = min(limits["calls"])
                    sleep_time = limits["period"] - (now - oldest_call)
                    if sleep_time > 0:
                        self.logger.warning(f"Rate limit reached for {api_name}, sleeping for {sleep_time:.2f} seconds")
                        time.sleep(sleep_time)
                
                # Record this call
                limits["calls"].append(now)
                
                # Update last called timestamp
                self.api_last_called[api_name] = now
                
                # Call the function
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def generate_cache_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """
        Generate a consistent cache key from a prefix and parameters.
        
        Args:
            prefix: Key prefix
            params: Dictionary of parameters
            
        Returns:
            Cache key string
        """
        # Sort parameters for consistent hashing
        param_str = json.dumps(params, sort_keys=True)
        
        # Create hash of the parameters
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        
        return f"{prefix}:{param_hash}"


class BlockchainDataIntegration:
    """
    Integration with blockchain data sources and APIs.
    """
    
    def __init__(self, manager: DataIntegrationManager, config: Dict[str, Any]):
        """
        Initialize the blockchain data integration.
        
        Args:
            manager: The parent data integration manager
            config: Configuration parameters
        """
        self.manager = manager
        self.logger = logging.getLogger("fin.data_integration.blockchain")
        self.config = config
        
        # API keys and endpoints
        self.api_keys = {
            "etherscan": os.environ.get("ETHERSCAN_API_KEY", config.get("api_keys", {}).get("etherscan", "")),
            "infura": os.environ.get("INFURA_API_KEY", config.get("api_keys", {}).get("infura", "")),
            "solana": os.environ.get("SOLANA_API_KEY", config.get("api_keys", {}).get("solana", "")),
        }
        
        # API endpoints
        self.endpoints = {
            "etherscan": "https://api.etherscan.io/api",
            "infura": f"https://mainnet.infura.io/v3/{self.api_keys['infura']}",
            "solana": "https://api.mainnet-beta.solana.com",
        }
        
        # Supported networks
        self.supported_networks = config.get("supported_networks", ["ethereum", "solana"])
    
    @property
    def _session(self):
        """Get a requests session."""
        if not hasattr(self, "_requests_session"):
            self._requests_session = requests.Session()
        return self._requests_session
    
    def get_latest_blocks(self, network: str, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get the latest blocks from a blockchain network.
        
        Args:
            network: Blockchain network (e.g., "ethereum", "solana")
            count: Number of blocks to retrieve
            
        Returns:
            List of block data dictionaries
        """
        if network not in self.supported_networks:
            raise ValueError(f"Unsupported network: {network}")
        
        cache_key = self.manager.generate_cache_key(
            f"blockchain:latest_blocks:{network}", 
            {"count": count, "timestamp": int(time.time() / 60)}  # Cache by minute
        )
        
        return self.manager.get_with_cache(
            cache_key,
            self._fetch_latest_blocks,
            network,
            count
        )
    
    @DataIntegrationManager.rate_limited("etherscan", 5, 1)  # 5 calls per second
    def _fetch_latest_blocks(self, network: str, count: int) -> List[Dict[str, Any]]:
        """
        Fetch latest blocks from blockchain API.
        
        Args:
            network: Blockchain network
            count: Number of blocks
            
        Returns:
            List of block data
        """
        if network == "ethereum":
            return self._fetch_ethereum_blocks(count)
        elif network == "solana":
            return self._fetch_solana_blocks(count)
        else:
            raise ValueError(f"Fetching blocks not implemented for network: {network}")
    
    def _fetch_ethereum_blocks(self, count: int) -> List[Dict[str, Any]]:
        """Fetch Ethereum blocks using Etherscan or Infura."""
        try:
            # First try Etherscan
            if self.api_keys["etherscan"]:
                params = {
                    "module": "proxy",
                    "action": "eth_blockNumber",
                    "apikey": self.api_keys["etherscan"]
                }
                response = self._session.get(self.endpoints["etherscan"], params=params)
                response.raise_for_status()
                data = response.json()
                
                if data.get("status") == "1" and "result" in data:
                    latest_block_hex = data["result"]
                    latest_block_num = int(latest_block_hex, 16)
                    
                    blocks = []
                    for i in range(count):
                        block_num = latest_block_num - i
                        block_data = self._get_ethereum_block_by_number(block_num)
                        if block_data:
                            blocks.append(block_data)
                    
                    return blocks
            
            # Fallback to Infura
            if self.api_keys["infura"]:
                # Implementation omitted for brevity, would be similar to above
                pass
            
            self.logger.error("Failed to fetch Ethereum blocks: No valid API keys")
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching Ethereum blocks: {e}")
            return []
    
    def _get_ethereum_block_by_number(self, block_number: int) -> Optional[Dict[str, Any]]:
        """Get a specific Ethereum block by number."""
        try:
            params = {
                "module": "proxy",
                "action": "eth_getBlockByNumber",
                "tag": hex(block_number),
                "boolean": "true",  # Include transaction data
                "apikey": self.api_keys["etherscan"]
            }
            
            response = self._session.get(self.endpoints["etherscan"], params=params)
            response.raise_for_status()
            data = response.json()
            
            if "result" in data and data["result"]:
                return data["result"]
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching Ethereum block {block_number}: {e}")
            return None
    
    def _fetch_solana_blocks(self, count: int) -> List[Dict[str, Any]]:
        """Fetch Solana blocks."""
        try:
            # For demonstration purposes - actual implementation would use Solana's JSON RPC API
            # This is a simplified placeholder
            headers = {"Content-Type": "application/json"}
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getRecentBlockhash"
            }
            
            response = self._session.post(self.endpoints["solana"], json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Placeholder data
            blocks = []
            for i in range(count):
                blocks.append({
                    "number": i,
                    "hash": f"sample_hash_{i}",
                    "timestamp": int(time.time()) - i * 30,  # Approx block time
                    "transactions": []
                })
            
            return blocks
            
        except Exception as e:
            self.logger.error(f"Error fetching Solana blocks: {e}")
            return []
    
    def get_token_info(self, network: str, token_address: str) -> Dict[str, Any]:
        """
        Get information about a token.
        
        Args:
            network: Blockchain network
            token_address: Contract address of the token
            
        Returns:
            Token information dictionary
        """
        if network not in self.supported_networks:
            raise ValueError(f"Unsupported network: {network}")
        
        cache_key = self.manager.generate_cache_key(
            f"blockchain:token:{network}", 
            {"address": token_address}
        )
        
        return self.manager.get_with_cache(
            cache_key,
            self._fetch_token_info,
            network,
            token_address
        )
    
    @DataIntegrationManager.rate_limited("etherscan", 5, 1)
    def _fetch_token_info(self, network: str, token_address: str) -> Dict[str, Any]:
        """
        Fetch token information from blockchain API.
        
        Args:
            network: Blockchain network
            token_address: Contract address
            
        Returns:
            Token information
        """
        if network == "ethereum":
            try:
                params = {
                    "module": "token",
                    "action": "tokeninfo",
                    "contractaddress": token_address,
                    "apikey": self.api_keys["etherscan"]
                }
                
                response = self._session.get(self.endpoints["etherscan"], params=params)
                response.raise_for_status()
                data = response.json()
                
                if data.get("status") == "1" and "result" in data:
                    return data["result"]
                
                self.logger.warning(f"Failed to fetch token info: {data.get('message', 'Unknown error')}")
                return {}
                
            except Exception as e:
                self.logger.error(f"Error fetching token info: {e}")
                return {}
        else:
            self.logger.warning(f"Token info fetch not implemented for network: {network}")
            return {}
    
    def analyze_smart_contract(self, network: str, contract_address: str) -> Dict[str, Any]:
        """
        Analyze a smart contract for security risks.
        
        Args:
            network: Blockchain network
            contract_address: Address of the smart contract
            
        Returns:
            Analysis results
        """
        if network not in self.supported_networks:
            raise ValueError(f"Unsupported network: {network}")
        
        # This would typically call a smart contract security analysis service
        # For demonstration, we'll return a simplified analysis
        
        return {
            "network": network,
            "contract_address": contract_address,
            "analysis_timestamp": int(time.time()),
            "risk_score": 0.45,  # Example score
            "vulnerabilities": [
                {
                    "type": "gas_optimization",
                    "severity": "low",
                    "description": "High gas consumption in fallback function"
                }
            ],
            "recommendations": [
                "Optimize storage usage to reduce gas costs"
            ]
        }


class FinancialDataIntegration:
    """
    Integration with financial data sources and APIs.
    """
    
    def __init__(self, manager: DataIntegrationManager, config: Dict[str, Any]):
        """
        Initialize the financial data integration.
        
        Args:
            manager: The parent data integration manager
            config: Configuration parameters
        """
        self.manager = manager
        self.logger = logging.getLogger("fin.data_integration.financial")
        self.config = config
        
        # API keys
        self.api_keys = {
            "alpha_vantage": os.environ.get("ALPHA_VANTAGE_API_KEY", config.get("api_keys", {}).get("alpha_vantage", "")),
            "coingecko": os.environ.get("COINGECKO_API_KEY", config.get("api_keys", {}).get("coingecko", "")),
        }
        
        # API endpoints
        self.endpoints = {
            "alpha_vantage": "https://www.alphavantage.co/query",
            "coingecko": "https://api.coingecko.com/api/v3",
        }
    
    @property
    def _session(self):
        """Get a requests session."""
        if not hasattr(self, "_requests_session"):
            self._requests_session = requests.Session()
        return self._requests_session
    
    def get_stock_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get current and historical price data for a stock.
        
        Args:
            symbol: Stock symbol (e.g., "AAPL" for Apple)
            
        Returns:
            Stock price data
        """
        cache_key = self.manager.generate_cache_key(
            "financial:stock_price", 
            {"symbol": symbol, "date": datetime.now().strftime("%Y-%m-%d")}
        )
        
        return self.manager.get_with_cache(
            cache_key,
            self._fetch_stock_price,
            symbol
        )
    
    @DataIntegrationManager.rate_limited("alpha_vantage", 5, 60)  # 5 calls per minute
    def _fetch_stock_price(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch stock price data from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Stock price data
        """
        try:
            if not self.api_keys["alpha_vantage"]:
                self.logger.error("Alpha Vantage API key not configured")
                return {}
            
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.api_keys["alpha_vantage"]
            }
            
            response = self._session.get(self.endpoints["alpha_vantage"], params=params)
            response.raise_for_status()
            data = response.json()
            
            if "Global Quote" in data and data["Global Quote"]:
                return data["Global Quote"]
            
            self.logger.warning(f"Failed to fetch stock price for {symbol}: {data.get('Note', 'Unknown error')}")
            return {}
            
        except Exception as e:
            self.logger.error(f"Error fetching stock price for {symbol}: {e}")
            return {}
    
    def get_crypto_price(self, coin_id: str) -> Dict[str, Any]:
        """
        Get current and historical price data for a cryptocurrency.
        
        Args:
            coin_id: Cryptocurrency ID (e.g., "bitcoin", "ethereum")
            
        Returns:
            Cryptocurrency price data
        """
        cache_key = self.manager.generate_cache_key(
            "financial:crypto_price", 
            {"coin_id": coin_id, "timestamp": int(time.time() / 300)}  # Cache for 5 minutes
        )
        
        return self.manager.get_with_cache(
            cache_key,
            self._fetch_crypto_price,
            coin_id
        )
    
    @DataIntegrationManager.rate_limited("coingecko", 10, 60)  # 10 calls per minute
    def _fetch_crypto_price(self, coin_id: str) -> Dict[str, Any]:
        """
        Fetch cryptocurrency price data from CoinGecko.
        
        Args:
            coin_id: Cryptocurrency ID
            
        Returns:
            Cryptocurrency price data
        """
        try:
            url = f"{self.endpoints['coingecko']}/coins/{coin_id}"
            params = {
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "false",
                "developer_data": "false"
            }
            
            # Add API key if available (for CoinGecko Pro)
            if self.api_keys["coingecko"]:
                params["x_cg_pro_api_key"] = self.api_keys["coingecko"]
            
            response = self._session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "market_data" in data:
                return {
                    "id": data.get("id", ""),
                    "name": data.get("name", ""),
                    "symbol": data.get("symbol", "").upper(),
                    "current_price": data["market_data"].get("current_price", {}),
                    "market_cap": data["market_data"].get("market_cap", {}),
                    "price_change_24h_percent": data["market_data"].get("price_change_percentage_24h", 0),
                    "price_change_7d_percent": data["market_data"].get("price_change_percentage_7d", 0),
                    "price_change_30d_percent": data["market_data"].get("price_change_percentage_30d", 0),
                    "total_volume": data["market_data"].get("total_volume", {}),
                    "last_updated": data["market_data"].get("last_updated", "")
                }
            
            self.logger.warning(f"Failed to fetch crypto price for {coin_id}")
            return {}
            
        except Exception as e:
            self.logger.error(f"Error fetching crypto price for {coin_id}: {e}")
            return {}


class NewsDataIntegration:
    """
    Integration with financial news and media sources.
    """
    
    def __init__(self, manager: DataIntegrationManager, config: Dict[str, Any]):
        """
        Initialize the news data integration.
        
        Args:
            manager: The parent data integration manager
            config: Configuration parameters
        """
        self.manager = manager
        self.logger = logging.getLogger("fin.data_integration.news")
        self.config = config
        
        # API keys
        self.api_keys = {
            "news_api": os.environ.get("NEWS_API_KEY", config.get("api_keys", {}).get("news_api", "")),
        }
        
        # API endpoints
        self.endpoints = {
            "news_api": "https://newsapi.org/v2",
        }
    
    @property
    def _session(self):
        """Get a requests session."""
        if not hasattr(self, "_requests_session"):
            self._requests_session = requests.Session()
        return self._requests_session
    
    def get_financial_news(self, query: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get financial news articles related to a query.
        
        Args:
            query: Search query
            days: Number of days to look back
            
        Returns:
            List of news articles
        """
        cache_key = self.manager.generate_cache_key(
            "news:financial", 
            {"query": query, "days": days, "date": datetime.now().strftime("%Y-%m-%d")}
        )
        
        return self.manager.get_with_cache(
            cache_key,
            self._fetch_financial_news,
            query,
            days
        )
    
    @DataIntegrationManager.rate_limited("news_api", 100, 86400)  # 100 calls per day
    def _fetch_financial_news(self, query: str, days: int) -> List[Dict[str, Any]]:
        """
        Fetch financial news from News API.
        
        Args:
            query: Search query
            days: Number of days to look back
            
        Returns:
            List of news articles
        """
        try:
            if not self.api_keys["news_api"]:
                self.logger.error("News API key not configured")
                return []
            
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            url = f"{self.endpoints['news_api']}/everything"
            params = {
                "q": f"{query} AND (finance OR economy OR market OR blockchain OR crypto OR investment)",
                "from": from_date,
                "sortBy": "relevancy",
                "language": "en",
                "apiKey": self.api_keys["news_api"]
            }
            
            response = self._session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "articles" in data:
                # Process and clean up articles
                articles = []
                for article in data["articles"]:
                    articles.append({
                        "title": article.get("title", ""),
                        "source": article.get("source", {}).get("name", "Unknown"),
                        "author": article.get("author", ""),
                        "url": article.get("url", ""),
                        "published_at": article.get("publishedAt", ""),
                        "description": article.get("description", ""),
                        "content": article.get("content", "")
                    })
                return articles
            
            self.logger.warning(f"Failed to fetch news: {data.get('message', 'Unknown error')}")
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching news: {e}")
            return []


class RegulatoryDataIntegration:
    """
    Integration with regulatory data sources and APIs.
    """
    
    def __init__(self, manager: DataIntegrationManager, config: Dict[str, Any]):
        """
        Initialize the regulatory data integration.
        
        Args:
            manager: The parent data integration manager
            config: Configuration parameters
        """
        self.manager = manager
        self.logger = logging.getLogger("fin.data_integration.regulatory")
        self.config = config
        
        # Supported jurisdictions
        self.supported_jurisdictions = config.get("jurisdictions_to_track", 
                                                 ["us", "eu", "uk", "sg", "in"])
    
    @property
    def _session(self):
        """Get a requests session."""
        if not hasattr(self, "_requests_session"):
            self._requests_session = requests.Session()
        return self._requests_session
    
    def get_regulations(self, jurisdiction: str) -> List[Dict[str, Any]]:
        """
        Get financial regulations for a specific jurisdiction.
        
        Args:
            jurisdiction: Jurisdiction code (e.g., "us", "eu")
            
        Returns:
            List of regulations
        """
        if jurisdiction not in self.supported_jurisdictions:
            raise ValueError(f"Unsupported jurisdiction: {jurisdiction}")
        
        cache_key = self.manager.generate_cache_key(
            "regulatory:regulations", 
            {"jurisdiction": jurisdiction, "date": datetime.now().strftime("%Y-%m-%d")}
        )
        
        return self.manager.get_with_cache(
            cache_key,
            self._fetch_regulations,
            jurisdiction
        )
    
    def _fetch_regulations(self, jurisdiction: str) -> List[Dict[str, Any]]:
        """
        Fetch regulations for a jurisdiction.
        
        Args:
            jurisdiction: Jurisdiction code
            
        Returns:
            List of regulations
        """
        # Note: In a real implementation, this would call an actual regulatory API
        # For demonstration, we'll return sample data
        
        regulations = {
            "us": [
                {
                    "id": "aml_kyc_us",
                    "name": "Bank Secrecy Act / Anti-Money Laundering Requirements",
                    "enforcing_body": "FinCEN",
                    "key_requirements": [
                        "Customer identification program",
                        "Suspicious activity reporting",
                        "Ongoing monitoring"
                    ],
                    "penalties": "Civil and criminal penalties including fines up to $25,000 per violation",
                    "last_updated": "2023-05-15"
                },
                {
                    "id": "sec_crypto",
                    "name": "SEC Cryptocurrency Enforcement",
                    "enforcing_body": "SEC",
                    "key_requirements": [
                        "Registration of securities offerings",
                        "Disclosure requirements",
                        "Trading compliance"
                    ],
                    "penalties": "Disgorgement, civil penalties, cease and desist",
                    "last_updated": "2023-09-22"
                }
            ],
            "eu": [
                {
                    "id": "mica",
                    "name": "Markets in Crypto-Assets Regulation",
                    "enforcing_body": "ESMA",
                    "key_requirements": [
                        "Licensing for crypto-asset service providers",
                        "Reserve requirements for stablecoins",
                        "Market abuse prevention"
                    ],
                    "penalties": "Administrative measures and fines up to 5M EUR or 3% of annual turnover",
                    "last_updated": "2023-11-10"
                },
                {
                    "id": "gdpr",
                    "name": "General Data Protection Regulation",
                    "enforcing_body": "National Data Protection Authorities",
                    "key_requirements": [
                        "Data minimization",
                        "User consent",
                        "Right to be forgotten"
                    ],
                    "penalties": "Up to 4% of global annual revenue or €20M",
                    "last_updated": "2018-05-25"
                }
            ],
            "uk": [
                {
                    "id": "ukfca_crypto",
                    "name": "UK Cryptoasset Registration",
                    "enforcing_body": "FCA",
                    "key_requirements": [
                        "Registration with FCA",
                        "AML/CTF compliance program",
                        "Fit and proper person test"
                    ],
                    "penalties": "Criminal offenses for operating without registration",
                    "last_updated": "2022-03-31"
                }
            ],
            "sg": [
                {
                    "id": "ps_act",
                    "name": "Payment Services Act",
                    "enforcing_body": "MAS",
                    "key_requirements": [
                        "Licensing for digital payment token services",
                        "AML/CFT requirements",
                        "Technology risk management"
                    ],
                    "penalties": "Fines up to SGD 250,000 and/or imprisonment up to 3 years",
                    "last_updated": "2020-01-28"
                }
            ],
            "in": [
                {
                    "id": "crypto_tax",
                    "name": "Virtual Digital Assets Taxation",
                    "enforcing_body": "Income Tax Department",
                    "key_requirements": [
                        "30% tax on income from VDAs",
                        "1% TDS on transfers",
                        "No offsetting of losses"
                    ],
                    "penalties": "Tax penalties and interest",
                    "last_updated": "2022-04-01"
                }
            ]
        }
        
        return regulations.get(jurisdiction, [])