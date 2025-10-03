import os
# import streamlit as st # Comment out or remove streamlit import if not essential for non-streamlit contexts
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import serpapi # Assuming this is needed elsewhere, keep it
import logging

# Configure basic logging if not already configured by the main script
# This is a failsafe; ideally, the main script configures logging.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s')

# Load environment variables from .env file in the project root
# Assumes .env is in the parent directory of 'utils'
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    logging.info(f"Loaded .env file from: {dotenv_path}")
else:
    logging.info(".env file not found at project root, relying on system environment variables or other secrets management.")

class OpenAIAPI:
    def __init__(self, model_name):
        """
        Initialize the OpenAIAPI with the given model name.
        """
        self.model_name = model_name
        self.model = model_name
        self.logger = logging.getLogger(__name__)
        
        api_key = None
        api_key_source = "Unknown"

        # 1. Try os.getenv() first (which load_dotenv() should populate if .env exists)
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_URL")
        
        if api_key:
            api_key_source = "os.getenv (potentially from .env)"
        
        # 2. Fallback to Streamlit secrets if not found via os.getenv() AND if Streamlit is available
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("OPENAI_API_KEY")
                if api_key:
                    api_key_source = "Streamlit secrets"
            except ImportError:
                self.logger.debug("Streamlit is not installed or not in a Streamlit environment, skipping Streamlit secrets.")
            except Exception as e:
                self.logger.debug(f"Error trying to access Streamlit secrets: {e}")

        self.logger.info(f"Attempting to use OpenAI API Key from {api_key_source}. Key: {'*' * (len(api_key) - 4) + api_key[-4:] if api_key else 'Not Found'}")

        if not api_key:
            self.logger.error("OPENAI_API_KEY not found through os.getenv, .env, or Streamlit secrets.")
            raise ValueError("OPENAI_API_KEY not found.")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def get_completion(self, system_content, user_content):
        """
        Get a completion from the OpenAI API.
        """
        self.logger.debug(f"Requesting completion. Model: {self.model_name}, System: '{system_content[:50]}...', User: '{user_content[:50]}...'")
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
            )
            response_content = completion.choices[0].message.content
            self.logger.debug(f"Completion received: '{response_content[:100]}...'")
            return response_content
        except Exception as e:
            self.logger.error(f"An error occurred during get_completion: {e}", exc_info=True)
            return None






    def _extract_content_with_fallback(self, message) -> str:
        """
        Extract content from message object with fallback to reasoning_content
        
        Args:
            message: OpenAI message object
            
        Returns:
            Extracted content string
        """
        # Try main content field first
        content = getattr(message, 'content', None)
        if content and content.strip():
            self.logger.debug(f"Using 'content' field: {len(content)} characters")
            return content
        
        # Fallback to reasoning_content
        reasoning_content = getattr(message, 'reasoning_content', None)
        if reasoning_content and reasoning_content.strip():
            self.logger.debug(f"Found 'reasoning_content' field: {len(reasoning_content)} characters")
            
            # Try to extract JSON from reasoning content
            extracted_json = self._extract_json_from_reasoning(reasoning_content)
            if extracted_json:
                self.logger.info("Successfully extracted JSON from reasoning_content")
                return extracted_json
            
            # If no JSON found, return the reasoning content as-is
            # (might contain instructions or explanations)
            self.logger.warning("No valid JSON found in reasoning_content, returning as-is")
            return reasoning_content
        
        self.logger.error("Both 'content' and 'reasoning_content' are empty or missing")
        return ""


    def _extract_json_from_reasoning(self, reasoning_content: str) -> str:
        """
        Extract JSON object from reasoning content that contains mixed text and JSON
        
        Args:
            reasoning_content: The reasoning content string
            
        Returns:
            Extracted JSON string or empty string if not found
        """
        import re
        import json
        
        # Look for JSON object patterns in the reasoning content
        # Pattern 1: Look for { ... } structures
        json_object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_object_pattern, reasoning_content, re.DOTALL)
        
        for match in matches:
            try:
                # Test if it's valid JSON
                json.loads(match)
                self.logger.debug(f"Found valid JSON object: {len(match)} chars")
                return match
            except json.JSONDecodeError:
                continue
        
        # Pattern 2: Look for array structures
        json_array_pattern = r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'
        matches = re.findall(json_array_pattern, reasoning_content, re.DOTALL)
        
        for match in matches:
            try:
                json.loads(match)
                self.logger.debug(f"Found valid JSON array: {len(match)} chars")
                return match
            except json.JSONDecodeError:
                continue
        
        # Pattern 3: Look for the specific case we see in logs
        # JSON starting with specific pattern
        specific_patterns = [
            r'\{\s*"name":\s*"[^"]*".*?\}',
            r'\[\s*\{\s*"name":\s*"[^"]*".*?\}\s*\]',
        ]
        
        for pattern in specific_patterns:
            matches = re.findall(pattern, reasoning_content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    json.loads(match)
                    self.logger.debug(f"Found JSON with specific pattern: {len(match)} chars")
                    return match
                except json.JSONDecodeError:
                    continue
        
        # Pattern 4: Try to find JSON by looking for balanced braces
        brace_positions = []
        brace_count = 0
        
        for i, char in enumerate(reasoning_content):
            if char == '{':
                if brace_count == 0:
                    start_pos = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    potential_json = reasoning_content[start_pos:i+1]
                    try:
                        json.loads(potential_json)
                        self.logger.debug(f"Found JSON by brace matching: {len(potential_json)} chars")
                        return potential_json
                    except json.JSONDecodeError:
                        continue
        
        self.logger.warning("Could not extract valid JSON from reasoning_content")
        return ""


    def _clean_json_content(self, content: str) -> str:
        """
        Clean JSON content by removing common formatting issues
        
        Args:
            content: Raw JSON content string
            
        Returns:
            Cleaned JSON content string
        """
        import re
        
        # Remove leading/trailing whitespace
        cleaned = content.strip()
        
        # Remove any text before the first { or [
        json_start = min(
            (cleaned.find('{') if cleaned.find('{') != -1 else len(cleaned)),
            (cleaned.find('[') if cleaned.find('[') != -1 else len(cleaned))
        )
        
        if json_start < len(cleaned):
            cleaned = cleaned[json_start:]
        
        # Remove any text after the last } or ]
        last_brace = max(cleaned.rfind('}'), cleaned.rfind(']'))
        if last_brace != -1:
            cleaned = cleaned[:last_brace + 1]
        
        # Fix common JSON formatting issues
        # Remove trailing commas before closing braces/brackets
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        
        # Fix unescaped quotes in strings (basic attempt)
        # This is a simplified approach - more sophisticated parsing might be needed
        
        return cleaned



    def _clean_json_content(self, content: str) -> str:
        """
        Clean JSON content by removing common formatting issues

        Args:
            content: Raw JSON content string

        Returns:
            Cleaned JSON content string
        """
        import re

        # Try to find JSON within the text
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            cleaned = json_match.group(0)
        else:
            # If no JSON found, use original content
            cleaned = content

        # Remove leading/trailing whitespace
        cleaned = cleaned.strip()

        # Remove any text before the first { or [
        json_start = min(
            (cleaned.find('{') if cleaned.find('{') != -1 else len(cleaned)),
            (cleaned.find('[') if cleaned.find('[') != -1 else len(cleaned))
        )

        if json_start < len(cleaned):
            cleaned = cleaned[json_start:]

        # Remove any text after the last } or ]
        last_brace = max(cleaned.rfind('}'), cleaned.rfind(']'))
        if last_brace != -1:
            cleaned = cleaned[:last_brace + 1]

        # Fix common JSON formatting issues
        # Remove trailing commas before closing braces/brackets
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)

        # Fix unescaped quotes in strings (basic attempt)
        # This is a simplified approach - more sophisticated parsing might be needed

        return cleaned

    def get_structured_output(self, schema_class: BaseModel, system_message: str, user_message: str, **kwargs) -> BaseModel:
        """
        Get structured output from the LLM using the provided schema.
        FIXED VERSION - handles empty content and reasoning_content fields
        


        Args:
            schema_class: Pydantic model class for validation
            system_message: System prompt
            user_message: User prompt
            **kwargs: Additional parameters for the API call
            
        Returns:
            Instance of schema_class with validated data
        """
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
            # Set default parameters for structured output
            api_params = {
                "model": self.model,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "temperature": kwargs.get("temperature", 0.1),
                "max_tokens": kwargs.get("max_tokens", 4000),
            }
            
            # Add any additional parameters
            api_params.update({k: v for k, v in kwargs.items() 
                              if k not in ["temperature", "max_tokens"]})
            
            self.logger.debug(f"Requesting structured output. Model: {self.model}, Schema: {schema_class.__name__}, "
                             f"System: '{system_message[:50] if system_message else None}...', User: '{user_message[:50] if user_message else None}...'")
                            # f"System: '{system_message[:50]}...', User: '{user_message[:50]}...'")
            
            # Make the API call
            response = self.client.chat.completions.create(**api_params)
            
            # DEBUG: Log the full response structure
            self.logger.debug(f"Raw response type: {type(response)}")
            self.logger.debug(f"Response attributes: {dir(response)}")
            
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                self.logger.debug(f"Choice type: {type(choice)}, attributes: {dir(choice)}")
                
                if hasattr(choice, 'message'):
                    message = choice.message
                    self.logger.debug(f"Message type: {type(message)}, attributes: {dir(message)}")
                    
                    # Extract content with fallback to reasoning_content
                    raw_content = self._extract_content_with_fallback(message)
                    
                    if not raw_content or not raw_content.strip():
                        error_msg = ("Empty response content. This might indicate:\n"
                                   "1. API configuration issues\n"
                                   "2. Model not following JSON format instructions\n"
                                   "3. Content filtering or safety measures\n"
                                   "4. Token limit exceeded")
                        self.logger.error(error_msg)
                        raise ValueError("Empty response content - check API configuration")
                    
                    self.logger.debug(f"Extracted content length: {len(raw_content)}")
                    self.logger.debug(f"Content preview: {raw_content[:200]}...")
                    
                    # Validate and parse JSON
                    try:
                        parsed = schema_class.model_validate_json(raw_content)
                        self.logger.debug(f"Successfully parsed structured output: {schema_class.__name__}")
                        return parsed
                        
                    except Exception as parse_error:
                        self.logger.error(f"JSON parsing failed: {parse_error}")
                        self.logger.error(f"Raw content: {raw_content[:500]}")
                        
                        # Try to clean and re-parse the JSON
                        cleaned_content = self._clean_json_content(raw_content)
                        if cleaned_content != raw_content:
                            try:
                                parsed = schema_class.model_validate_json(cleaned_content)
                                self.logger.info("Successfully parsed after JSON cleaning")
                                return parsed
                            except Exception as clean_parse_error:
                                self.logger.error(f"Cleaning attempt also failed: {clean_parse_error}")
                        
                        raise parse_error
            
            raise ValueError("Response structure is invalid - no choices or message found")
            
        except Exception as e:
            self.logger.error(f"An error occurred during get_structured_output: {e}")
            raise



    def get_embeddings(self, text):
        """
        Get embeddings for the given text.
        """
        self.logger.debug(f"Requesting embeddings for text: '{text[:50]}...'")
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-large",  # You might want to make this configurable
                dimensions = 100,
            )
            self.logger.debug(f"Embedding response: {response}")
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"An error occurred while getting embeddings: {e}", exc_info=True)
            return None

# The GoogleSearchAPI class remains unchanged
class GoogleSearchAPI:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        serpapi_key = None
        key_source = "Unknown"

        # 1. Try os.getenv()
        serpapi_key = os.getenv("SERPAPI_API_KEY")
        if serpapi_key:
            key_source = "os.getenv (potentially from .env)"
        
        # 2. Fallback to Streamlit secrets if not found and Streamlit is available
        if not serpapi_key:
            try:
                import streamlit as st
                serpapi_key = st.secrets.get("SERPAPI_API_KEY")
                if serpapi_key:
                    key_source = "Streamlit secrets"
            except ImportError:
                self.logger.debug("Streamlit is not installed or not in a Streamlit environment, skipping Streamlit secrets for SERPAPI_API_KEY.")
            except Exception as e: # Broad exception for other st.secrets issues
                self.logger.debug(f"Error trying to access Streamlit secrets for SERPAPI_API_KEY: {e}")

        self.logger.info(f"Attempting to use SerpAPI Key from {key_source}. Key: {'**********' + serpapi_key[-4:] if serpapi_key else 'Not Found'}")

        if not serpapi_key:
            self.logger.error("SERPAPI_API_KEY not found through os.getenv, .env, or Streamlit secrets.")
            raise ValueError("SERPAPI_API_KEY not found.")
        self.api_key = serpapi_key

    def search(self, query, num_results=5):
        
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key,
            "num": num_results
        }
        search = serpapi.search(params)
        results = search.as_dict()
        return results.get('organic_results', [])

if __name__ == "__main__":
    
    # Setup basic logging for the __main__ block, if not already set
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG)

    print("Testing starts")

    # Test OpenAIAPI
    openai_api = OpenAIAPI("openai/gpt-oss-120b")  # Use an appropriate model name
    
    # Test get_completion
    system_content = "You are a helpful assistant."
    user_content = "What's the capital of France?"
    completion = openai_api.get_completion(system_content, user_content)
    print("OpenAI Completion Test:")
    print(completion)
    print()

    # Test get_structured_output
    from pydantic import BaseModel, Field

    class WeatherResponse(BaseModel):
        temperature: float = Field(..., description="Temperature in Celsius")
        conditions: str = Field(..., description="Weather conditions (e.g., sunny, rainy)")

    system_prompt = "You are a weather reporting system. Provide weather information based on the user's query."
    user_prompt = "What's the weather like in Paris today?"
    structured_output = openai_api.get_structured_output(WeatherResponse, user_prompt, system_prompt)
    print("OpenAI Structured Output Test:")
    print(structured_output)
    print()

    # Test get_embeddings
    text = "This is a test sentence for embeddings."
    embeddings = openai_api.get_embeddings(text)
    print("OpenAI Embeddings Test:")
    print(f"Embedding vector length: {len(embeddings)}")
    print(f"First 5 values: {embeddings[:5]}")
    print()

    # Test GoogleSearchAPI
    google_api = GoogleSearchAPI()
    search_results = google_api.search("Python programming")
    print("Google Search API Test:")
    for i, result in enumerate(search_results[:3], 1):  # Print first 3 results
        print(f"{i}. {result['title']}")
        print(f"   {result['link']}")
        print()
