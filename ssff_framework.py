import logging
from typing import Dict, Any
from pydantic import BaseModel

from agents.market_agent import MarketAgent
from agents.product_agent import ProductAgent
from agents.founder_agent import FounderAgent
from agents.vc_scout_agent import VCScoutAgent, StartupInfo
from agents.integration_agent import IntegrationAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StartupFramework:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.market_agent = MarketAgent(model)
        self.product_agent = ProductAgent(model)
        self.founder_agent = FounderAgent(model)
        self.vc_scout_agent = VCScoutAgent(model)
        self.integration_agent = IntegrationAgent(model)

    def analyze_startup(self, startup_info_str: str) -> Dict[str, Any]:
        logger.info("Starting startup analysis in advanced mode")

        # Parse the input string into a StartupInfo schema
        startup_info = self.vc_scout_agent.parse_record(startup_info_str)

        print("Parse Record: ", startup_info)

        # Check if parsing was successful
        if isinstance(startup_info, dict):
            startup_info = StartupInfo(**startup_info)
        elif not isinstance(startup_info, StartupInfo):
            logger.error("Failed to parse startup info")
            return {"error": "Failed to parse startup info"}

        # Get prediction and categorization
        prediction, categorization = self.vc_scout_agent.side_evaluate(startup_info)
        logger.info(f"VCScout prediction: {prediction}")

        # Perform agent analyses
        market_analysis = self.market_agent.analyze(startup_info.dict(), "advanced")
        product_analysis = self.product_agent.analyze(startup_info.dict(), "advanced")
        founder_analysis = self.founder_agent.analyze(startup_info.dict(), "advanced")

        # Log the startup_info for debugging
        logger.debug(f"Startup info: {startup_info.dict()}")

        founder_segmentation = self.founder_agent.segment_founder(startup_info.founder_backgrounds)
        founder_idea_fit = self.founder_agent.calculate_idea_fit(startup_info.dict(), startup_info.founder_backgrounds)
        rf_prediction = prediction

        # Integrate analyses
        integrated_analysis = self.integration_agent.integrate_analyses(
            market_info=market_analysis.dict(),
            product_info=product_analysis.dict(),
            founder_info=founder_analysis.dict(),
            prediction=prediction,
            founder_idea_fit=founder_idea_fit[0],
            founder_segmentation=founder_segmentation,
            rf_prediction=rf_prediction,
            categorization=categorization.dict(),
            mode="advanced"
        )

        quant_decision = self.integration_agent.getquantDecision(
            prediction,
            founder_idea_fit[0],  # Assuming this returns a tuple (idea_fit, cosine_similarity)
            founder_segmentation,
            integrated_analysis.dict()
        )

        return {
            'Final Analysis': integrated_analysis.dict(),
            'Market Analysis': market_analysis.dict(),
            'Product Analysis': product_analysis.dict(),
            'Founder Analysis': founder_analysis.dict(),
            'Founder Segmentation': founder_segmentation,
            'Founder Idea Fit': founder_idea_fit[0],
            'Categorical Prediction': prediction,
            'Categorization': categorization.dict(),
            'Quantitative Decision': quant_decision.dict(),
            'Random Forest Prediction': rf_prediction,
            'Startup Info': startup_info.dict()
        }

def main():
    framework = StartupFramework()
    
    startup_info_str = """
    HealthTech AI is developing an AI-powered health monitoring wearable device. 
    The global wearable technology market is estimated at $50 billion with a CAGR of 15.9% from 2020 to 2027. 
    Key competitors include Fitbit, Apple Watch, and Garmin. 
    The product offers real-time health tracking with predictive analysis. 
    The founding team consists of experienced entrepreneurs with backgrounds in AI and healthcare. 
    They've raised $2 million in seed funding to date.
    """

    advanced_result = framework.analyze_startup(startup_info_str)
    print("\nAdvanced Analysis Result:")
    print(advanced_result)

if __name__ == "__main__":
    main()
