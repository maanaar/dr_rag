import os
from google import generativeai as genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Function declaration for doctor search
search_doctors_function = {
    "name": "search_doctors",
    "description": "Search doctors by name, specialty, or business unit",
    "parameters": {
        "type": "object",
        "properties": {
            "doctor_name": {"type": "string"},
            "speciality": {"type": "string"},
            "business_unit": {"type": "string"},
        },
        "required": []  # all optional
    }
}

# Create tool
search_tool = types.Tool(function_declarations=[search_doctors_function])
config = types.GenerateContentConfig(tools=[search_tool])


def get_gemini_client():
    """Return Gemini client and tool config"""
    return client, config
