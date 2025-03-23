"""
Regulatory Compliance Agent Implementation
"""

import logging
from typing import Dict, List, Any, Optional
from fin.base_agent import BaseAgent
import json
from datetime import datetime, timedelta


class RegulatoryCompliance(BaseAgent):
    """
    The Regulatory Compliance agent is responsible for:
    1. Keeping track of financial and blockchain regulations across jurisdictions
    2. Flagging compliance risks in proposed financial transactions
    3. Generating compliance reports for different regulatory frameworks
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Regulatory Compliance agent.
        
        Args:
            config: Optional configuration parameters
        """
        name = "regulatory_compliance"
        description = "Tracks financial and blockchain regulations and assesses compliance risks"
        super().__init__(name, description, config)
        self.regulations = {}
        self.jurisdictions = {}
        self.compliance_frameworks = {}
        self.last_updated = datetime.now() - timedelta(days=1)  # Simulate last update was yesterday
        
    def _initialize_agent(self):
        """Set up the agent with regulatory data sources and compliance frameworks."""
        super()._initialize_agent()
        # In a real implementation, this would connect to regulatory databases and APIs
        self.logger.info("Initializing regulatory data sources and compliance frameworks")
        
        # Initialize sample regulatory data for demo purposes
        self._initialize_sample_regulatory_data()
        
    def _initialize_sample_regulatory_data(self):
        """Initialize sample regulatory data for demo purposes."""
        # Sample jurisdictions
        self.jurisdictions = {
            "us": {
                "name": "United States",
                "key_regulators": ["SEC", "CFTC", "FinCEN", "OCC", "FDIC"],
                "regulatory_approach": "multi-agency fragmented",
                "crypto_stance": "evolving",
                "compliance_complexity": "high"
            },
            "eu": {
                "name": "European Union",
                "key_regulators": ["EBA", "ESMA", "ECB", "National Authorities"],
                "regulatory_approach": "harmonized framework",
                "crypto_stance": "regulated",
                "compliance_complexity": "high"
            },
            "uk": {
                "name": "United Kingdom",
                "key_regulators": ["FCA", "PRA", "Bank of England"],
                "regulatory_approach": "principles-based",
                "crypto_stance": "regulated",
                "compliance_complexity": "medium"
            },
            "sg": {
                "name": "Singapore",
                "key_regulators": ["MAS"],
                "regulatory_approach": "centralized",
                "crypto_stance": "progressive",
                "compliance_complexity": "medium"
            },
            "in": {
                "name": "India",
                "key_regulators": ["RBI", "SEBI", "IFSCA", "FIU-IND"],
                "regulatory_approach": "centralized-progressive",
                "crypto_stance": "evolving",
                "compliance_complexity": "high"
            }
        }
        
        # Sample regulations
        self.regulations = {
            "aml_kyc": {
                "name": "Anti-Money Laundering / Know Your Customer",
                "jurisdictions": ["global", "us", "eu", "uk", "sg", "in"],
                "key_requirements": ["Customer identification", "Transaction monitoring", "Suspicious activity reporting"],
                "penalties": "Severe: criminal charges, heavy fines",
                "compliance_priority": "critical"
            },
            "gdpr": {
                "name": "General Data Protection Regulation",
                "jurisdictions": ["eu", "eea"],
                "key_requirements": ["Data minimization", "User consent", "Right to be forgotten"],
                "penalties": "Up to 4% of global annual revenue or €20M",
                "compliance_priority": "high"
            },
            "mifid_ii": {
                "name": "Markets in Financial Instruments Directive II",
                "jurisdictions": ["eu"],
                "key_requirements": ["Transaction reporting", "Best execution", "Client categorization"],
                "penalties": "Significant financial penalties",
                "compliance_priority": "high"
            },
            "mica": {
                "name": "Markets in Crypto-Assets Regulation",
                "jurisdictions": ["eu"],
                "key_requirements": ["Licensing", "Reserve requirements for stablecoins", "Market abuse prevention"],
                "penalties": "Similar to traditional financial instruments",
                "compliance_priority": "high"
            },
            "sec_regulations": {
                "name": "SEC Cryptocurrency Enforcement",
                "jurisdictions": ["us"],
                "key_requirements": ["Registration of securities offerings", "Disclosure requirements", "Trading compliance"],
                "penalties": "Disgorgement, civil penalties, cease and desist",
                "compliance_priority": "high"
            },
            "pmla": {
                "name": "Prevention of Money Laundering Act",
                "jurisdictions": ["in"],
                "key_requirements": ["KYC procedures", "Reporting suspicious transactions", "Record keeping", "Designated Director appointment"],
                "penalties": "Imprisonment up to 10 years and substantial fines",
                "compliance_priority": "critical"
            },
            "fema": {
                "name": "Foreign Exchange Management Act",
                "jurisdictions": ["in"],
                "key_requirements": ["Cross-border transaction reporting", "Foreign investment rules compliance", "Repatriation requirements"],
                "penalties": "Up to three times the amount involved in the contravention",
                "compliance_priority": "high"
            },
            "sebi_regulations": {
                "name": "SEBI Financial Market Regulations",
                "jurisdictions": ["in"],
                "key_requirements": ["Registration requirements", "Investor protection", "Disclosure norms", "Corporate governance"],
                "penalties": "Substantial monetary penalties and market access restrictions",
                "compliance_priority": "high"
            },
            "dpdp": {
                "name": "Digital Personal Data Protection Act",
                "jurisdictions": ["in"],
                "key_requirements": ["Data consent", "Purpose limitation", "Data security", "Data fiduciary obligations"],
                "penalties": "Up to INR 250 crore for serious breaches",
                "compliance_priority": "high"
            },
            "crypto_regulations_india": {
                "name": "Virtual Digital Assets Regulations",
                "jurisdictions": ["in"],
                "key_requirements": ["30% tax on crypto income", "1% TDS on transfers", "Financial reporting", "No offsetting of losses"],
                "penalties": "Tax penalties and potential legal consequences",
                "compliance_priority": "high"
            }
        }
        
        # Sample compliance frameworks
        self.compliance_frameworks = {
            "iso_27001": {
                "name": "ISO/IEC 27001 - Information Security",
                "focus": "Data security and protection",
                "key_controls": ["Asset management", "Access control", "Incident management"],
                "relevance": "Required for handling sensitive financial data"
            },
            "pci_dss": {
                "name": "Payment Card Industry Data Security Standard",
                "focus": "Payment card data security",
                "key_controls": ["Network security", "Cardholder data protection", "Vulnerability management"],
                "relevance": "Mandatory for handling payment card data"
            },
            "nist": {
                "name": "NIST Cybersecurity Framework",
                "focus": "Cybersecurity risk management",
                "key_controls": ["Identify", "Protect", "Detect", "Respond", "Recover"],
                "relevance": "Widely adopted standard for cybersecurity"
            },
            "rbi_guidelines": {
                "name": "RBI Cybersecurity Framework for Banks",
                "focus": "Banking system security and risk management",
                "key_controls": ["IT governance", "Risk assessment", "Incident response", "Continuous surveillance"],
                "relevance": "Mandatory for banks and financial institutions in India"
            }
        }
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a regulatory compliance-related query.
        
        Args:
            query: The user's query string
            
        Returns:
            A dictionary containing insights and recommendations
        """
        self.logger.info(f"Processing regulatory compliance query: {query}")
        
        # Here we'd use NLP to parse the query and determine the specific regulatory analysis needed
        # For now, we'll use a simplified approach
        
        query_lower = query.lower()
        response = {
            "insights": [],
            "recommendations": [],
            "risks": [],
            "compliance_requirements": [],
            "confidence": 0.0
        }
        
        # Check if query is about specific jurisdictions
        jurisdictions_mentioned = []
        for jcode, jdata in self.jurisdictions.items():
            if jcode in query_lower or jdata["name"].lower() in query_lower:
                jurisdictions_mentioned.append(jcode)
        
        # Check if query is about specific regulations
        regulations_mentioned = []
        for rcode, rdata in self.regulations.items():
            if rcode.lower() in query_lower or rdata["name"].lower() in query_lower:
                regulations_mentioned.append(rcode)
        
        # If specific jurisdictions were mentioned, provide targeted insights
        if jurisdictions_mentioned:
            for jcode in jurisdictions_mentioned:
                self._analyze_jurisdiction(jcode, response)
        else:
            # Provide general jurisdictional insights
            response["insights"].append("Regulatory approaches vary significantly across jurisdictions, requiring tailored compliance strategies")
            response["insights"].append("The EU has the most comprehensive regulatory framework for crypto-assets with MiCA")
            response["insights"].append("Singapore offers a balanced approach with clear regulatory guidance while promoting innovation")
        
        # If specific regulations were mentioned, provide targeted insights
        if regulations_mentioned:
            for rcode in regulations_mentioned:
                self._analyze_regulation(rcode, response)
        else:
            # Check what regulatory domains the query might be about
            domains = {
                "cryptocurrency": ["crypto", "bitcoin", "blockchain", "token", "ico", "defi"],
                "data_privacy": ["data", "privacy", "personal information", "gdpr"],
                "financial_services": ["banking", "payment", "investment", "trading"],
                "aml": ["money laundering", "terrorism financing", "kyc", "customer due diligence"]
            }
            
            for domain, keywords in domains.items():
                if any(kw in query_lower for kw in keywords):
                    self._analyze_domain(domain, response)
        
        # Set confidence based on how well we could answer the query
        response["confidence"] = min(0.9, 0.3 + 0.2 * len(response["insights"]) + 0.1 * len(response["recommendations"]))
        
        return response
    
    def _analyze_jurisdiction(self, jurisdiction_code: str, response: Dict[str, Any]):
        """
        Analyze regulatory environment in a specific jurisdiction.
        
        Args:
            jurisdiction_code: The jurisdiction code to analyze
            response: The response dictionary to update
        """
        jdata = self.jurisdictions.get(jurisdiction_code)
        if not jdata:
            return
        
        jname = jdata["name"]
        response["insights"].append(f"{jname} has a {jdata['regulatory_approach']} approach to financial regulation")
        response["insights"].append(f"Key regulatory bodies in {jname}: {', '.join(jdata['key_regulators'])}")
        response["insights"].append(f"{jname}'s stance on cryptocurrency is {jdata['crypto_stance']}")
        
        complexity = jdata["compliance_complexity"]
        if complexity == "high":
            response["recommendations"].append(f"Allocate significant resources to {jname} compliance due to high complexity")
            response["recommendations"].append(f"Consider specialized legal counsel for {jname} operations")
        elif complexity == "medium":
            response["recommendations"].append(f"Implement structured compliance programs for {jname} with regular updates")
        else:
            response["recommendations"].append(f"Standard compliance measures should be sufficient for {jname}")
        
        # Add relevant regulations for this jurisdiction
        relevant_regs = [rcode for rcode, rdata in self.regulations.items() if jurisdiction_code in rdata["jurisdictions"]]
        if relevant_regs:
            reg_names = [self.regulations[r]["name"] for r in relevant_regs]
            response["insights"].append(f"Key regulations in {jname}: {', '.join(reg_names)}")
            
            # Highlight high priority regulations
            high_priority = [self.regulations[r]["name"] for r in relevant_regs 
                             if self.regulations[r]["compliance_priority"] in ["critical", "high"]]
            if high_priority:
                response["recommendations"].append(f"Prioritize compliance with {', '.join(high_priority)}")
    
    def _analyze_regulation(self, regulation_code: str, response: Dict[str, Any]):
        """
        Analyze a specific regulation.
        
        Args:
            regulation_code: The regulation code to analyze
            response: The response dictionary to update
        """
        rdata = self.regulations.get(regulation_code)
        if not rdata:
            return
        
        rname = rdata["name"]
        applicable_jurisdictions = [self.jurisdictions.get(j, {}).get("name", j) for j in rdata["jurisdictions"]]
        
        response["insights"].append(f"{rname} applies in: {', '.join(applicable_jurisdictions)}")
        response["insights"].append(f"Key requirements: {', '.join(rdata['key_requirements'])}")
        response["insights"].append(f"Penalties for non-compliance: {rdata['penalties']}")
        
        response["recommendations"].append(f"Implement specific controls for {rname} based on its requirements")
        if rdata["compliance_priority"] in ["critical", "high"]:
            response["recommendations"].append(f"Conduct regular audits for {rname} compliance due to its {rdata['compliance_priority']} priority")
        
        # Add compliance requirements
        for req in rdata["key_requirements"]:
            response["compliance_requirements"].append({
                "regulation": rname,
                "requirement": req,
                "priority": rdata["compliance_priority"]
            })
    
    def _analyze_domain(self, domain: str, response: Dict[str, Any]):
        """
        Analyze regulatory issues in a specific domain.
        
        Args:
            domain: The domain to analyze
            response: The response dictionary to update
        """
        if domain == "cryptocurrency":
            response["insights"].append("Cryptocurrency regulations vary widely by jurisdiction but are generally becoming more comprehensive")
            response["insights"].append("The EU's MiCA provides the most comprehensive framework for crypto-asset regulation")
            response["insights"].append("US regulation is evolving with various agencies claiming jurisdiction over different aspects")
            
            response["recommendations"].append("Implement robust AML/KYC procedures as they are universally required for crypto operations")
            response["recommendations"].append("Engage with regulators proactively when launching new crypto products or services")
            
            response["risks"].append({
                "category": "Regulatory",
                "description": "Uncertain classification of tokens as securities, commodities, or currencies",
                "severity": "High",
                "mitigation": "Legal opinion for each token type before launch"
            })
        
        elif domain == "data_privacy":
            response["insights"].append("Data privacy regulations are becoming more stringent globally, with GDPR setting the standard")
            response["insights"].append("Cross-border data transfers face increasing restrictions, especially from EU to non-adequate jurisdictions")
            
            response["recommendations"].append("Implement data minimization and purpose limitation in all systems and processes")
            response["recommendations"].append("Maintain detailed records of processing activities and data protection impact assessments")
            
            response["risks"].append({
                "category": "Compliance",
                "description": "Inadequate user consent mechanisms for data processing",
                "severity": "High",
                "mitigation": "Implement granular consent management system"
            })
        
        elif domain == "financial_services":
            response["insights"].append("Financial services regulations are increasingly focusing on consumer protection and market stability")
            response["insights"].append("Digital-only banks and services face evolving regulatory requirements across jurisdictions")
            
            response["recommendations"].append("Implement robust governance and risk management frameworks that satisfy regulatory expectations")
            response["recommendations"].append("Ensure clear disclosure of fees, risks, and terms to customers")
            
            response["risks"].append({
                "category": "Operational",
                "description": "Inadequate segregation of client funds",
                "severity": "Critical",
                "mitigation": "Implement rigorous accounting controls and regular audits"
            })
        
        elif domain == "aml":
            response["insights"].append("AML regulations are universal with increasing emphasis on beneficial ownership identification")
            response["insights"].append("Transaction monitoring expectations are becoming more sophisticated, requiring advanced analytics")
            
            response["recommendations"].append("Implement risk-based approach to customer due diligence with enhanced measures for high-risk clients")
            response["recommendations"].append("Ensure suspicious activity reporting processes are efficient and meet timing requirements")
            
            response["risks"].append({
                "category": "Compliance",
                "description": "Inadequate screening against sanctions and PEP lists",
                "severity": "Critical",
                "mitigation": "Implement automated screening with regular updates"
            })
    
    def get_capabilities(self) -> List[str]:
        """
        Return a list of the agent's capabilities.
        
        Returns:
            A list of capability descriptions
        """
        return [
            "Track financial and blockchain regulations across jurisdictions",
            "Flag compliance risks in proposed financial transactions",
            "Generate compliance reports for different regulatory frameworks",
            "Analyze cross-border regulatory implications",
            "Monitor regulatory changes and their impact on operations",
            "Provide guidance on regulatory requirements for new products"
        ]
    
    def assess_transaction_compliance(self, transaction_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the compliance risks of a proposed financial transaction.
        
        Args:
            transaction_details: Details of the transaction including:
                                 parties, jurisdictions, amount, asset type, etc.
            
        Returns:
            A dictionary containing compliance assessment
        """
        # Extract key transaction parameters with defaults if not provided
        origin_jurisdiction = transaction_details.get('origin_jurisdiction', 'unknown')
        destination_jurisdiction = transaction_details.get('destination_jurisdiction', 'unknown')
        asset_type = transaction_details.get('asset_type', 'unknown')
        amount = transaction_details.get('amount', 0)
        is_cross_border = origin_jurisdiction != destination_jurisdiction
        party_type = transaction_details.get('party_type', 'individual')
        
        # Initialize risk assessment
        risk_level = "low"
        risk_factors = []
        required_checks = []
        
        # Jurisdictional risk assessment
        high_risk_jurisdictions = ["sanctioned", "high-risk"]
        origin_risk = False
        destination_risk = False
        
        if origin_jurisdiction in high_risk_jurisdictions:
            risk_level = "high"
            risk_factors.append(f"Origin jurisdiction ({origin_jurisdiction}) is high-risk")
            origin_risk = True
            
        if destination_jurisdiction in high_risk_jurisdictions:
            risk_level = "high"
            risk_factors.append(f"Destination jurisdiction ({destination_jurisdiction}) is high-risk")
            destination_risk = True
            
        # Cross-border considerations
        if is_cross_border:
            required_checks.append("Cross-border transfer reporting")
            if risk_level != "high":
                risk_level = "medium"
            risk_factors.append("Cross-border transaction requiring additional scrutiny")
        
        # Asset type considerations
        if asset_type.lower() in ["cryptocurrency", "crypto", "digital asset", "token"]:
            if risk_level != "high":
                risk_level = "medium"
            risk_factors.append("Digital asset transaction with enhanced compliance requirements")
            required_checks.append("Digital asset source of funds verification")
            required_checks.append("Blockchain analytics screening")
        
        # Amount considerations
        threshold_reporting = False
        if asset_type.lower() == "fiat" and amount >= 10000:
            threshold_reporting = True
        elif asset_type.lower() in ["cryptocurrency", "crypto", "digital asset", "token"] and amount >= 3000:
            threshold_reporting = True
            
        if threshold_reporting:
            required_checks.append("Large transaction reporting")
            if risk_level == "low":
                risk_level = "medium"
            risk_factors.append(f"Transaction amount ({amount}) exceeds reporting threshold")
        
        # Party type considerations
        if party_type.lower() in ["business", "corporation", "entity"]:
            required_checks.append("Beneficial ownership verification")
            required_checks.append("Entity purpose and structure assessment")
        
        # Basic checks always required
        required_checks.append("AML/KYC verification")
        required_checks.append("Sanctions screening")
        
        # Compile the assessment
        assessment = {
            "overall_risk": risk_level,
            "risk_factors": risk_factors,
            "required_checks": required_checks,
            "jurisdictional_requirements": []
        }
        
        # Add jurisdiction-specific requirements if known
        for jcode, jdata in self.jurisdictions.items():
            if jcode in [origin_jurisdiction, destination_jurisdiction]:
                regulators = ", ".join(jdata["key_regulators"])
                assessment["jurisdictional_requirements"].append(
                    f"{jdata['name']}: Verify compliance with {regulators} requirements"
                )
        
        # Add recommendations based on risk level
        assessment["recommendations"] = []
        if risk_level == "high":
            assessment["recommendations"].append("Conduct enhanced due diligence on all parties")
            assessment["recommendations"].append("Consider filing suspicious activity report based on risk factors")
            assessment["recommendations"].append("Obtain senior management approval before proceeding")
        elif risk_level == "medium":
            assessment["recommendations"].append("Verify source of funds with appropriate documentation")
            assessment["recommendations"].append("Conduct standard due diligence on all parties")
        else:  # low risk
            assessment["recommendations"].append("Process according to standard procedures")
            assessment["recommendations"].append("Maintain appropriate transaction records")
        
        return assessment
    
    def generate_compliance_report(self, framework: str, scope: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a compliance report for a specific regulatory framework.
        
        Args:
            framework: The compliance framework to use
            scope: Details of the scope for the report
            
        Returns:
            A dictionary containing the compliance report
        """
        # Check if the framework is supported
        framework_data = self.compliance_frameworks.get(framework)
        if not framework_data:
            return {
                "error": f"Framework {framework} not found in available compliance frameworks",
                "recommendations": ["Select a supported framework: " + ", ".join(self.compliance_frameworks.keys())]
            }
        
        # Extract scope parameters
        jurisdiction = scope.get('jurisdiction', 'global')
        business_activities = scope.get('business_activities', [])
        product_types = scope.get('product_types', [])
        
        # Identify applicable regulations based on scope
        applicable_regulations = []
        for rcode, rdata in self.regulations.items():
            if jurisdiction in rdata["jurisdictions"] or "global" in rdata["jurisdictions"]:
                applicable_regulations.append({
                    "code": rcode,
                    "name": rdata["name"],
                    "priority": rdata["compliance_priority"],
                    "key_requirements": rdata["key_requirements"]
                })
        
        # Generate the compliance report
        report = {
            "framework": {
                "code": framework,
                "name": framework_data["name"],
                "focus": framework_data["focus"]
            },
            "scope": {
                "jurisdiction": jurisdiction,
                "business_activities": business_activities,
                "product_types": product_types
            },
            "applicable_regulations": applicable_regulations,
            "key_controls": framework_data["key_controls"],
            "compliance_actions": []
        }
        
        # Generate compliance actions based on framework and applicable regulations
        for regulation in applicable_regulations:
            for requirement in regulation["key_requirements"]:
                report["compliance_actions"].append({
                    "regulation": regulation["name"],
                    "requirement": requirement,
                    "priority": regulation["priority"],
                    "action": f"Implement controls to ensure compliance with {requirement}"
                })
        
        # Add framework-specific controls
        for control in framework_data["key_controls"]:
            report["compliance_actions"].append({
                "regulation": framework_data["name"],
                "requirement": control,
                "priority": "high",
                "action": f"Implement {control} controls according to {framework_data['name']} standards"
            })
        
        # Add recommendations
        report["recommendations"] = [
            f"Prioritize high-priority compliance actions for immediate implementation",
            f"Conduct regular audits against {framework_data['name']} standards",
            f"Maintain documentation of all compliance controls and their effectiveness"
        ]
        
        return report 