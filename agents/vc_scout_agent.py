from config import DEFAULT_MODEL

import os
import sys
import logging
import joblib
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional, Tuple
import json

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from agents.base_agent import BaseAgent

class StartupInfo(BaseModel):
    name: str = Field(..., description="The official name of the startup")
    description: str = Field(..., description="A brief overview of what the startup does")
    market_size: Optional[str] = Field(None, description="The size of the market the startup is targeting")
    growth_rate: Optional[str] = Field(None, description="The growth rate of the market")
    competition: Optional[str] = Field(None, description="Key competitors in the space")
    market_trends: Optional[str] = Field(None, description="Current trends within the market")
    go_to_market_strategy: Optional[str] = Field(None, description="The startup's plan for entering the market")
    product_details: Optional[str] = Field(None, description="Details about the startup's product or service")
    technology_stack: Optional[str] = Field(None, description="Technologies used in the product")
    scalability: Optional[str] = Field(None, description="How the product can scale")
    user_feedback: Optional[str] = Field(None, description="Any feedback received from users")
    product_fit: Optional[str] = Field(None, description="How well the product fits the target market")
    founder_backgrounds: Optional[str] = Field(None, description="Background information on the founders")
    track_records: Optional[str] = Field(None, description="The track records of the founders")
    leadership_skills: Optional[str] = Field(None, description="Leadership skills of the team")
    vision_alignment: Optional[str] = Field(None, description="How the team's vision aligns with the product")
    team_dynamics: Optional[str] = Field(None, description="The dynamics within the startup team")
    web_traffic_growth: Optional[str] = Field(None, description="Information on the growth of web traffic to the startup's site")
    social_media_presence: Optional[str] = Field(None, description="The startup's presence on social media")
    investment_rounds: Optional[str] = Field(None, description="Details of any investment rounds")
    regulatory_approvals: Optional[str] = Field(None, description="Any regulatory approvals obtained")
    patents: Optional[str] = Field(None, description="Details of any patents held by the startup")

class StartupCategorization(BaseModel):
    industry_growth: str = Field(..., description="Is the startup operating in an industry experiencing growth? [Yes/No/N/A]")
    market_size: str = Field(..., description="Is the target market size for the startup's product/service considered large? [Small/Medium/Large/N/A]")
    development_pace: str = Field(..., description="Does the startup demonstrate a fast pace of development compared to competitors? [Slower/Same/Faster/N/A]")
    market_adaptability: str = Field(..., description="Is the startup considered adaptable to market changes? [Not Adaptable/Somewhat Adaptable/Very Adaptable/N/A]")
    execution_capabilities: str = Field(..., description="How would you rate the startup's execution capabilities? [Poor/Average/Excellent/N/A]")
    funding_amount: str = Field(..., description="Has the startup raised a significant amount of funding in its latest round? [Below Average/Average/Above Average/N/A]")
    valuation_change: str = Field(..., description="Has the startup's valuation increased with time? [Decreased/Remained Stable/Increased/N/A]")
    investor_backing: str = Field(..., description="Are well-known investors or venture capital firms backing the startup? [Unknown/Recognized/Highly Regarded/N/A]")
    reviews_testimonials: str = Field(..., description="Are the reviews and testimonials for the startup predominantly positive? [Negative/Mixed/Positive/N/A]")
    product_market_fit: str = Field(..., description="Do market surveys indicate a strong product-market fit for the startup? [Weak/Moderate/Strong/N/A]")
    sentiment_analysis: str = Field(..., description="Does the sentiment analysis of founder and company descriptions suggest high positivity? [Negative/Neutral/Positive/N/A]")
    innovation_mentions: str = Field(..., description="Are terms related to innovation frequently mentioned in the company's public communications? [Rarely/Sometimes/Often/N/A]")
    cutting_edge_technology: str = Field(..., description="Does the startup mention cutting-edge technology in its descriptions? [No/Mentioned/Emphasized/N/A]")
    timing: str = Field(..., description="Considering the startup's industry and current market conditions, is the timing for the startup's product or service right? [Too Early/Just Right/Too Late/N/A]")

class StartupEvaluation(BaseModel):
    market_opportunity: str = Field(..., description="Assessment of the market opportunity")
    product_innovation: str = Field(..., description="Evaluation of product innovation")
    founding_team: str = Field(..., description="Analysis of the founding team")
    potential_risks: str = Field(..., description="Identification of potential risks")
    overall_potential: int = Field(..., description="Overall potential score on a scale of 1 to 10")
    investment_recommendation: str = Field(None, description="Investment recommendation: 'Invest' or 'Pass'")
    confidence: float = Field(None, description="Confidence level in the recommendation (0 to 1)")
    rationale: str = Field(None, description="Brief explanation for the recommendation")

class VCScoutAgent(BaseAgent):
    def __init__(self, model=DEFAULT_MODEL):
        super().__init__(model)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Load the encoder and model
        self.encoder = joblib.load(os.path.join(project_root, 'models/trained_encoder_RF.joblib'))
        self.model_random_forest = joblib.load(os.path.join(project_root,'models/random_forest_classifier.joblib'))

    def parse_record(self, startup_info: str) -> StartupInfo:
        """
        Convert a string description of a startup into a StartupInfo schema.
        """
        self.logger.info("Parsing startup information into StartupInfo schema")
        prompt = """
        Convert the following startup description into a detailed JSON structure that matches the StartupInfo schema. 
        Include as many fields as possible based on the information provided in the description.
        If information for a field is not available, omit that field from the JSON.
        Pay special attention to product details, technology stack, and any information about the product's unique features or market fit.
 
        Startup description:
        {startup_info}
        """  
        try:
            startup_info_dict = self.get_json_response(StartupInfo, prompt, startup_info)
            self.logger.debug(f"Parsed startup info: {startup_info_dict}")
            if isinstance(startup_info_dict, dict):
                return StartupInfo(**startup_info_dict)  # 🟢 convert dict to StartupInfo
            return startup_info_dict  # Return the dictionary directly
        except Exception as e:
            self.logger.error(f"Error parsing startup info: {str(e)}")
            return StartupInfo(name="Error", description="Failed to parse startup info")





    def parse_record(self, startup_info: str) -> StartupInfo:
        """
        Convert a string description of a startup into a StartupInfo schema.
        """
        self.logger.info("Parsing startup information into StartupInfo schema")
        prompt = """
        Convert the following startup description into a detailed JSON structure that matches the StartupInfo schema.
        Include ALL fields listed below. If information for a field is not available, set it to "N/A" or null.

        **Required fields (must be included):**
        - name (str)
        - description (str, default to "N/A" if missing)
        - mission (str, default to "N/A" if missing)
        - vision (str, default to "N/A" if missing)
        - industry (str, default to "N/A" if missing)
        - patents (str, default to "N/A" if none)

        **Optional fields (include if available, otherwise omit):**
        - tagline (str)
        - ticker (str)
        - ...

        **Example Output:**
        ```json
        {
          "name": "Omniscient Neurotechnology",
          "description": "Uses data & machine learning to build advanced brain maps...",
          "mission": "To improve the lives of billions through connectomics...",
          "vision": "Connectomics is a foundational technology of the 21st century...",
          "industry": "Neurotechnology",
          "patents": "N/A",
          "tagline": "THE HUMAN CONNECTOME COMPANY",
          "ticker": "o8t"
        }
        ```

        **Startup description:**
        {startup_info}
        """
        try:
            startup_info_dict = self.get_json_response(StartupInfo, prompt, startup_info)
            self.logger.debug(f"Parsed startup info: {startup_info_dict}")
            if isinstance(startup_info_dict, dict):
                return StartupInfo(**startup_info_dict)
            return startup_info_dict
        except Exception as e:
            self.logger.error(f"Error parsing startup info: {str(e)}")
            # Return a valid StartupInfo object with default values
            return StartupInfo(
                name="N/A",
                description="Failed to parse startup info",
                mission="N/A",
                vision="N/A",
                industry="N/A",
                patents="N/A",
                # ... (include all required fields)
            )



    def parse_record(self, startup_info: str) -> StartupInfo:
        """
        Convert a string description of a startup into a StartupInfo schema.
        """
        self.logger.info("Parsing startup information into StartupInfo schema")
        prompt = """
        You are an expert analyst at a top-tier venture capital firm. Your task is to meticulously extract information from a startup's pitch deck text.

        **CRITICAL INSTRUCTION: Pay special attention to finding ANY information about the founding team.** Founder information is often scattered or implicitly mentioned. Look for:
        - Explicit "Team", "Founders", or "About Us" sections.
        - Names mentioned at the beginning, end, or in signatures.
        - Personal anecdotes ("When I was at Google...", "Our co-founder, a PhD from MIT...").
        - Descriptions of experience ("10 years in fintech", "serial entrepreneur").
        - Educational background mentions ("Harvard MBA", "dropped out of Stanford").
        - Photos with captions (the text might mention "John Doe, CEO").
        - Any use of "I", "we", "our" that refers to the team's background.

        If you cannot find specific names, describe the *type* of experience mentioned (e.g., "technical team with AI background", "finance expert founder").

        Convert the following startup description into a detailed JSON structure that matches the StartupInfo schema.
        Include ALL fields listed below. If information for a field is not available, set it to "N/A" or null.

        **Required fields (must be included):**
        - name (str)
        - description (str, default to "N/A" if missing)
        - mission (str, default to "N/A" if missing)
        - vision (str, default to "N/A" if missing)
        - industry (str, default to "N/A" if missing)
        - patents (str, default to "N/A" if none)
        - founder_backgrounds (str, details about founders' backgrounds, education, experience)
        - track_records (str, details about founders' past achievements or companies)
        - leadership_skills (str, details about founders' leadership capabilities)
        - vision_alignment (str, how the team's vision aligns with the product)
        - team_dynamics (str, information about how the team works together)

        **Optional fields (include if available, otherwise omit):**
        - tagline (str)
        - ticker (str)
        - market_size (str)
        - growth_rate (str)
        - competition (str)
        - market_trends (str)
        - go_to_market_strategy (str)
        - product_details (str)
        - technology_stack (str)
        - scalability (str)
        - user_feedback (str)
        - product_fit (str)
        - web_traffic_growth (str)
        - social_media_presence (str)
        - investment_rounds (str)
        - regulatory_approvals (str)

        **Example Output:**
        ```json
        {
          "name": "Omniscient Neurotechnology",
          "description": "Uses data & machine learning to build advanced brain maps...",
          "mission": "To improve the lives of billions through connectomics...",
          "vision": "Connectomics is a foundational technology of the 21st century...",
          "industry": "Neurotechnology",
          "patents": "N/A",
          "founder_backgrounds": "Dr. Jane Smith - PhD in Neuroscience from Stanford, 10 years research experience; John Doe - MBA from Harvard, former executive at MedTech company",
          "track_records": "Dr. Smith published 20+ papers in neuroscience; Doe led 3 successful product launches at previous company",
          "leadership_skills": "Strong technical leadership combined with business acumen",
          "vision_alignment": "Team shares vision of democratizing brain mapping technology",
          "team_dynamics": "Complementary skills with scientific and business expertise",
          "tagline": "THE HUMAN CONNECTOME COMPANY",
          "ticker": "o8t"
        }
        ```

        **Startup description:**
        {startup_info}
        """
        try:
            startup_info_dict = self.get_json_response(StartupInfo, prompt, startup_info)
            self.logger.debug(f"Parsed startup info: {startup_info_dict}")
            if isinstance(startup_info_dict, dict):
                return StartupInfo(**startup_info_dict)
            return startup_info_dict
        except Exception as e:
            self.logger.error(f"Error parsing startup info: {str(e)}")
            # Return a valid StartupInfo object with default values
            return StartupInfo(
                name="N/A",
                description="Failed to parse startup info",
                mission="N/A",
                vision="N/A",
                industry="N/A",
                patents="N/A",
                founder_backgrounds="N/A",
                track_records="N/A",
                leadership_skills="N/A",
                vision_alignment="N/A",
                team_dynamics="N/A"
            )


    def evaluate(self, startup_info: StartupInfo, mode: str) -> StartupEvaluation:
        self.logger.info(f"Starting startup evaluation in {mode} mode")
        startup_info_str = startup_info.json()
        self.logger.debug(f"Startup info: {startup_info_str}")
        
        if mode == "basic":
            analysis = self.get_json_response(StartupEvaluation, self._get_basic_evaluation_prompt(), startup_info_str)
            self.logger.info("Basic evaluation completed")
        else:  # advanced mode
            analysis = self.get_json_response(StartupEvaluation, self._get_advanced_evaluation_prompt(), startup_info_str)
            self.logger.info("Advanced evaluation completed")
        
        return analysis

    def side_evaluate(self, startup_info: StartupInfo) -> Tuple[str, StartupCategorization]:
        self.logger.info("Starting side evaluation")
        startup_info_str = startup_info.json()
        categorization = self.get_json_response(StartupCategorization, self._get_categorization_prompt(), startup_info_str)
        self.logger.info("Categorization completed")

        # Validate the categorization
        # for field, value in categorization:
        #     expected_values = StartupCategorization.__fields__[field].field_info.description.split('[')[1].split(']')[0].split('/')
        #     if value not in expected_values:
        #         self.logger.warning(f"Unexpected value '{value}' for field '{field}'. Expected one of {expected_values}. Setting to 'N/A'.")
        #         setattr(categorization, field, 'N/A')

        prediction = self._predict(categorization)
        self.logger.info(f"Prediction: {prediction}")

        return prediction, categorization

    def _predict(self, categorization: StartupCategorization) -> str:
        category_mappings = {
            "industry_growth": ["No", "N/A", "Yes"],
            "market_size": ["Small", "Medium", "Large", "N/A"],
            "development_pace": ["Slower", "Same", "Faster", "N/A"],
            "market_adaptability": ["Not Adaptable", "Somewhat Adaptable", "Very Adaptable", "N/A"],
            "execution_capabilities": ["Poor", "Average", "Excellent", "N/A"],
            "funding_amount": ["Below Average", "Average", "Above Average", "N/A"],
            "valuation_change": ["Decreased", "Remained Stable", "Increased", "N/A"],
            "investor_backing": ["Unknown", "Recognized", "Highly Regarded", "N/A"],
            "reviews_testimonials": ["Negative", "Mixed", "Positive", "N/A"],
            "product_market_fit": ["Weak", "Moderate", "Strong", "N/A"],
            "sentiment_analysis": ["Negative", "Neutral", "Positive", "N/A"],
            "innovation_mentions": ["Rarely", "Sometimes", "Often", "N/A"],
            "cutting_edge_technology": ["No", "Mentioned", "Emphasized", "N/A"],
            "timing": ["Too Early", "Just Right", "Too Late", "N/A"]
        }

        feature_order = list(category_mappings.keys())
        encoded_features = self.encoder.transform(pd.DataFrame([categorization.dict()]))
        prediction = self.model_random_forest.predict(encoded_features)

        return "Successful" if prediction[0] == 1 else "Unsuccessful"

    def _get_basic_evaluation_prompt(self):
        return """
        As an experienced VC scout, evaluate the startup based on the following information:
        {startup_info}

        Provide a comprehensive analysis including market opportunity, product innovation, founding team, and potential risks.
        Conclude with an overall potential score from 1 to 10.
        """

    def _get_advanced_evaluation_prompt(self):
        return """
        As an experienced VC scout, provide an in-depth evaluation of the startup based on the following information:
        {startup_info}

        Provide a comprehensive analysis including market opportunity, product innovation, founding team, and potential risks.
        Conclude with an overall potential score from 1 to 10, an investment recommendation (Invest or Pass), 
        a confidence level in your recommendation (0 to 1), and a brief rationale for your decision.
        """




    def _get_categorization_prompt(self):
        return """
        You are a strict JSON generator.
        Output ONLY valid JSON matching the StartupCategorization schema.
        Do not include explanations, instructions, or extra text.
        If information is missing, fill with "N/A".

        Schema with required snake_case keys and allowed values:

        {
          "industry_growth": "Yes/No/N/A",
          "market_size": "Small/Medium/Large/N/A",
          "development_pace": "Slower/Same/Faster/N/A",
          "market_adaptability": "Not Adaptable/Somewhat Adaptable/Very Adaptable/N/A",
          "execution_capabilities": "Poor/Average/Excellent/N/A",
          "funding_amount": "Below Average/Average/Above Average/N/A",
          "valuation_change": "Decreased/Remained Stable/Increased/N/A",
          "investor_backing": "Unknown/Recognized/Highly Regarded/N/A",
          "reviews_testimonials": "Negative/Mixed/Positive/N/A",
          "product_market_fit": "Weak/Moderate/Strong/N/A",
          "sentiment_analysis": "Negative/Neutral/Positive/N/A",
          "innovation_mentions": "Rarely/Sometimes/Often/N/A",
          "cutting_edge_technology": "No/Mentioned/Emphasized/N/A",
          "timing": "Too Early/Just Right/Too Late/N/A"
        }

        Startup Information:
        {startup_info}
        """




if __name__ == "__main__":
    def test_vc_scout_agent():
        # Configure logging for the test function
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        logger.info("Starting VCScoutAgent test")
        # Create a VCScoutAgent instance
        agent = VCScoutAgent()

        # Test startup info
        startup_info_str = """
        HealthTech AI is developing an AI-powered health monitoring wearable device. 
        The global wearable technology market is estimated at $50 billion with a CAGR of 15.9% from 2020 to 2027. 
        Key competitors include Fitbit, Apple Watch, and Garmin. 
        The product offers real-time health tracking with predictive analysis. 
        The founding team consists of experienced entrepreneurs with backgrounds in AI and healthcare. 
        They've raised $2 million in seed funding to date.
        """

        # Test parse_record
        print("Parsing Startup Info:")
        startup_info = agent.parse_record(startup_info_str)
        print(startup_info)
        print()

        # Test basic evaluation
        print("Basic Evaluation:")
        basic_evaluation = agent.evaluate(startup_info, mode="basic")
        print(basic_evaluation)
        print()

        # Test advanced evaluation
        print("Advanced Evaluation:")
        advanced_evaluation = agent.evaluate(startup_info, mode="advanced")
        print(advanced_evaluation)
        print()

        # Test side evaluation
        print("Side Evaluation:")
        prediction, categorization = agent.side_evaluate(startup_info)
        print(f"Prediction: {prediction}")
        print("Categorization:")
        print(categorization)

    # Run the test function
    test_vc_scout_agent()
