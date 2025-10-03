from config import DEFAULT_MODEL

import os
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
import requests   # <-- add this



# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from agents.base_agent import BaseAgent
from utils.api_wrapper import OpenAIAPI
from pydantic import BaseModel, Field

class FounderAnalysis(BaseModel):
    competency_score: int = Field(..., description="Founder competency score on a scale of 1 to 10")
    analysis: str = Field(..., description="Detailed analysis of the founding team, including strengths and challenges.")

class AdvancedFounderAnalysis(FounderAnalysis):
    segmentation: int = Field(..., description="Founder segmentation level. 1-5. 1 is L1, 5 is L5")
    cosine_similarity: float = Field(..., description="The cosine similarity between founder's desc and startup info.")
    idea_fit: float = Field(..., description="Idea fit score")

class FounderSegmentation(BaseModel):
    segmentation: int = Field(..., description="Founder segmentation level. 1-5. 1 is L1, 5 is L5")

class FounderAgent(BaseAgent):
    def __init__(self, model=DEFAULT_MODEL):
        super().__init__(model)
        self.neural_network = load_model(os.path.join(project_root, 'models', 'neural_network.keras'))

        self.api_token = os.getenv("OPENAI_API_KEY")
        self.api_url = "https://api.deepinfra.com/v1/openai/embeddings"
        self.embedding_model = "Qwen/Qwen3-Embedding-4B"   # or 8B/0.6B depending on what you want
        self.embedding_dim = 1536   # change to 4096 if you use Qwen3-Embedding-8B
        self.normalize = True

    def _get_embedding(self, text: str):
        if not text or text.strip() == "":
            return np.zeros(self.embedding_dim)

        response = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_token}"},
            json={
                "model": self.embedding_model,
                "input": text,
                "encoding_format": "float"
            }
        )
        response.raise_for_status()
        emb = response.json()["data"][0]["embedding"]

        emb = np.array(emb, dtype=float)
        if self.normalize and np.linalg.norm(emb) > 0:
            emb = emb / np.linalg.norm(emb)
        return emb




    def analyze(self, startup_info, mode):
        founder_info = self._get_founder_info(startup_info)
        
        print("\n===== DEBUG: Founder Information =====")
        print(founder_info)


        if mode == "advanced":
            basic_analysis = self.get_json_response(FounderAnalysis, self._get_analysis_prompt(), founder_info)
            segmentation = self.segment_founder(founder_info)
            idea_fit, cosine_similarity = self.calculate_idea_fit(startup_info, founder_info)
            
            return AdvancedFounderAnalysis(
                **basic_analysis.dict(),
                segmentation=segmentation,
                cosine_similarity=cosine_similarity,
                idea_fit=idea_fit,
            )
        else:
            return self.get_json_response(FounderAnalysis, self._get_analysis_prompt(), founder_info)

    def _get_founder_info(self, startup_info):
        return f"Founders' Backgrounds: {startup_info.get('founder_backgrounds', '')}\n" \
               f"Track Records: {startup_info.get('track_records', '')}\n" \
               f"Leadership Skills: {startup_info.get('leadership_skills', '')}\n" \
               f"Vision and Alignment: {startup_info.get('vision_alignment', '')}"

    def segment_founder(self, founder_info):
        return self.get_json_response(FounderSegmentation, self._get_segmentation_prompt(), founder_info).segmentation

    def segment_founder(self, founder_info):
        if founder_info is None or founder_info.strip() == "":
            founder_info = "No founder information available"
        return self.get_json_response(FounderSegmentation, self._get_segmentation_prompt(), founder_info).segmentation


    def calculate_idea_fit(self, startup_info, founder_info):
        # Get embeddings
        startup_embedding = self._get_embedding(startup_info.get("description", ""))
        founder_embedding = self._get_embedding(founder_info)
#        founder_embedding = self._get_embedding(founder_info.get("background", ""))


    def calculate_idea_fit(self, startup_info, founder_info):
        # Get embeddings for the startup idea
        idea_embedding = self.embed_text(startup_info.get("idea", ""))

        # Get embeddings for the founder info (string)
        founder_embedding = self.embed_text(founder_info)

        # Calculate cosine similarity
        cosine_similarity = self.cosine_similarity(idea_embedding, founder_embedding)

        return {
            "idea_fit_score": cosine_similarity,
            "founder_strength": founder_info,  # keep founder info as plain string
        }



    def calculate_idea_fit(self, startup_info, founder_info):
        # Get embeddings for the startup idea
        idea_embedding = self._get_embedding(startup_info.get("idea", ""))

        # Get embeddings for the founder info (string)
        founder_embedding = self._get_embedding(founder_info)

        # Defensive check: validate embeddings
        def validate_embedding(emb, label):
            arr = np.array(emb, dtype=float)
            if arr.size == 0:
                self.logger.warning(f"{label} embedding is empty. Using zeros as fallback.")
                return np.zeros(self.embedding_dim)
            if np.isnan(arr).any() or np.isinf(arr).any():
                self.logger.warning(f"{label} embedding contains NaN/Inf. Fixing.")
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            return arr

        idea_embedding = validate_embedding(idea_embedding, "Startup Idea")
        founder_embedding = validate_embedding(founder_embedding, "Founder")

        # Calculate cosine similarity safely
        cosine_sim = self._calculate_cosine_similarity(founder_embedding, idea_embedding)

        return {
            "idea_fit_score": cosine_sim,
            "founder_strength": founder_info.strip(),  # keep as plain string
        }, cosine_sim


        # Validate embeddings
        def validate_embedding(emb, label):
            arr = np.array(emb, dtype=float)
            if arr.size == 0:
                self.logger.warning(f"{label} embedding is empty. Using zeros as fallback.")
                return np.zeros(768)  # match embedding dimension (adjust if your model uses another size)
            if np.isnan(arr).any() or np.isinf(arr).any():
                self.logger.warning(f"{label} embedding contains NaN/Inf. Fixing.")
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            return arr

        startup_embedding = validate_embedding(startup_embedding, "Startup")
        founder_embedding = validate_embedding(founder_embedding, "Founder")

        # Calculate similarity safely
        cosine_sim = self._calculate_cosine_similarity(founder_embedding, startup_embedding)

        # For downstream logic
        idea_fit = {
            "founder_strength": founder_info.get("strengths", ""),
            "startup_focus": startup_info.get("focus", ""),
            "cosine_similarity": cosine_sim
        }

        return idea_fit, cosine_sim

    def calculate_idea_fit(self, startup_info, founder_info):
        idea_embedding = self._get_embedding(startup_info.get("idea", ""))
        founder_embedding = self._get_embedding(founder_info)

        idea_embedding = self._validate_embedding(idea_embedding, "Startup Idea")
        founder_embedding = self._validate_embedding(founder_embedding, "Founder")

        cosine_sim = self._calculate_cosine_similarity(founder_embedding, idea_embedding)

        return cosine_sim, cosine_sim  # first is idea_fit (float), second is cosine_sim




    def calculate_idea_fit(self, startup_info, founder_info):
        idea_embedding = self._get_embedding(startup_info.get("idea", ""))
        founder_embedding = self._get_embedding(founder_info)

        idea_embedding = self._validate_embedding(idea_embedding, "Startup Idea")
        founder_embedding = self._validate_embedding(founder_embedding, "Founder")

        cosine_sim = self._calculate_cosine_similarity(founder_embedding, idea_embedding)

        # Prepare input for the neural network
        X_new_embeddings = np.array(founder_embedding).reshape(1, -1)
        X_new_embeddings_2 = np.array(idea_embedding).reshape(1, -1)
        X_new_cosine = np.array([[cosine_sim]])
        X_new = np.concatenate([X_new_embeddings, X_new_embeddings_2, X_new_cosine], axis=1)

        # Predict using the neural network
        idea_fit = self.neural_network.predict(X_new)[0][0]

        return float(idea_fit), float(cosine_sim)





    def calculate_idea_fit(self, startup_info, founder_info):
            """Calculate idea fit between founder and startup."""
            try:
                # Get embeddings
                idea_text = startup_info.get("description", "") or startup_info.get("idea", "")
                idea_embedding = self._get_embedding(idea_text)
                founder_embedding = self._get_embedding(founder_info)

                # Validate embeddings
                idea_embedding = self._validate_embedding(idea_embedding, "Startup Idea")
                founder_embedding = self._validate_embedding(founder_embedding, "Founder")

                # Calculate cosine similarity
                cosine_sim = self._calculate_cosine_similarity(founder_embedding, idea_embedding)

                # Calculate idea fit using neural network if available
                if self.neural_network is not None:
                    try:
                        # Prepare input for neural network
                        X_new_embeddings = np.array(founder_embedding).reshape(1, -1)
                        X_new_embeddings_2 = np.array(idea_embedding).reshape(1, -1)
                        X_new_cosine = np.array([[cosine_sim]])
                        X_new = np.concatenate([X_new_embeddings, X_new_embeddings_2, X_new_cosine], axis=1)

                        # Predict using neural network
                        idea_fit = float(self.neural_network.predict(X_new, verbose=0)[0][0])
                    except Exception as e:
                        self.logger.error(f"Neural network prediction failed: {e}")
                        # Fallback to cosine similarity
                        idea_fit = float(cosine_sim)
                else:
                    # Fallback calculation based on cosine similarity
                    idea_fit = float(cosine_sim)

                return idea_fit, cosine_sim

            except Exception as e:
                self.logger.error(f"Idea fit calculation failed: {e}")
                return 0.5, 0.5  # Return neutral values on failure




    def _calculate_cosine_similarity(self, vec1, vec2):
        import numpy as np
        v1 = np.array(vec1, dtype=float).reshape(1, -1)
        v2 = np.array(vec2, dtype=float).reshape(1, -1)

        # Defensive checks
        if v1.shape[1] != v2.shape[1]:
            self.logger.warning(f"Embedding dimension mismatch: {v1.shape} vs {v2.shape}. Returning 0.0")
            return 0.0

        if np.isnan(v1).any() or np.isnan(v2).any():
            self.logger.warning("NaN detected in embeddings, returning 0.0 similarity")
            return 0.0

        if np.isinf(v1).any() or np.isinf(v2).any():
            self.logger.warning("Inf detected in embeddings, returning 0.0 similarity")
            return 0.0

        try:
            return cosine_similarity(v1, v2)[0][0]
        except Exception as e:
            self.logger.error(f"Cosine similarity failed: {e}")
            return 0.0



    def _get_analysis_prompt(self):
        return """
        As a highly qualified analyst specializing in startup founder assessment, evaluate the founding team based on the provided information.
        Consider the founders' educational background, industry experience, leadership capabilities, and their ability to align and execute on the company's vision.
        Provide a competency score, key strengths, and potential challenges. Please write in great details.
        """


    def _get_analysis_prompt(self):
        return """
        You are a professional VC founder analyst.
        Analyze the founding team and respond ONLY with a valid JSON object matching this schema:
        {
            "competency_score": <int 1-10>,
            "analysis": "Detailed analysis of the founding team, including strengths and challenges."
        }

        Founder Information:
        {founder_info}

        Make sure your response is valid JSON and includes ALL fields.
        """




    def _get_segmentation_prompt(self):
        return """
        Categorize the founder into one of these levels: L1, L2, L3, L4, L5.
        L5: Entrepreneur who has built a $100M+ ARR business or had a major exit.
        L4: Entrepreneur with a small to medium-size exit or executive at a notable tech company.
        L3: 10-15 years of technical and management experience.
        L2: Entrepreneurs with a few years of experience or accelerator graduates.
        L1: Entrepreneurs with negligible experience but large potential.

        Based on the following information, determine the appropriate level:
        {founder_info}
        """


    def _get_segmentation_prompt(self):
        return """
        Categorize the founder strictly into one of these levels: L1, L2, L3, L4, or L5.
        Respond ONLY with a valid JSON object matching this schema:
        {
            "segmentation": <int 1-5>
        }

        Founder Information:
        {founder_info}

        Ensure your response is valid JSON and includes ALL fields.
        """



if __name__ == "__main__":
    def test_founder_agent():
        # Create a FounderAgent instance
        agent = FounderAgent()

        # Test startup info
        startup_info = {
            "founder_backgrounds": "MBA from Stanford, 5 years at Google as Product Manager",
            "track_records": "Successfully launched two products at Google, one reaching 1M users",
            "leadership_skills": "Led a team of 10 engineers and designers",
            "vision_alignment": "Strong passion for AI and its applications in healthcare",
            "description": "AI-powered health monitoring wearable device"
        }

        # Test basic analysis
        print("Basic Analysis:")
        basic_analysis = agent.analyze(startup_info, mode="basic")
        print(basic_analysis)
        print()

        # Test advanced analysis
        print("Advanced Analysis:")
        advanced_analysis = agent.analyze(startup_info, mode="advanced")
        print(advanced_analysis)

    # Run the test function
    test_founder_agent()
