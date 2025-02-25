import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Extract the variables from the .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
WOLFRAM_ALPHA_APPID = os.environ.get("WOLFRAM_ALPHA_APPID")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")


# Function to set environment variables (optional, if you want to ensure they are set in the system environment)
def set_environment():
    variable_dict = globals().items()
    for key, value in variable_dict:
        if "API" in key or "TOKEN" in key:
            os.environ[key] = value


# How to use the set_environment function
# from config import set_environment
# set_environment()
