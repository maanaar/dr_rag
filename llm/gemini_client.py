# llm/gemini_client.py
import os
from google import generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --------------------------
#   DEFINE TOOL FUNCTION
# --------------------------
def search_doctors(
    doctor_name: str = None,
    speciality: str = None,
    business_unit: str = None
):
    """
    Gemini will call this function automatically.
    Your real implementation will be inside app.py or another handler.
    """
    return {
        "doctor_name": doctor_name,
        "speciality": speciality,
        "business_unit": business_unit
    }


# --------------------------
#   RETURN MODEL WITH TOOLS
# --------------------------
def get_gemini_client():
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        tools=[search_doctors]   
    )
