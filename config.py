from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()

# No fallback — force user to define DEFAULT_MODEL in .env
DEFAULT_MODEL = os.environ["DEFAULT_MODEL"]
