# llm/gemini_client.py
import os
from google import generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_gemini_client():
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        tools=[
            {
                "name": "search_doctor",
                "description": "Search doctors by name, specialty, or business unit",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "doctor_name": {"type": "string"},
                        "specialty": {"type": "string"},
                        "business_unit": {"type": "string"},
                    },
                },
            }
        ]
    )
