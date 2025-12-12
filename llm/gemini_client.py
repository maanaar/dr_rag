import os
from google.genai import Client, types
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"


class GeminiClient:
    """Wrapper for Gemini API client with tool configuration"""
    
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.client = Client(api_key=api_key)
        self.search_tool = self._create_search_tool()
        self.config = types.GenerateContentConfig(tools=[self.search_tool])
    
    def _create_search_tool(self) -> types.Tool:
        """Create the search_doctors function declaration"""
        search_doctors_function = {
            "name": "search_doctors",
            "description": "Search doctors by name, specialty, or business unit in Arabic",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_name": {
                        "type": "string",
                        "description": "اسم الطبيب للبحث عنه"
                    },
                    "speciality": {
                        "type": "string",
                        "description": "التخصص الطبي للبحث عنه"
                    },
                    "business_unit": {
                        "type": "string",
                        "description": "الوحدة الطبية للبحث عنها"
                    },
                },
                "required": []
            }
        }
        return types.Tool(function_declarations=[search_doctors_function])
    
    def generate_content(self, contents, config=None):
        """Generate content using Gemini"""
        return self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=config or self.config
        )


def get_gemini_client():
    """Factory function to get Gemini client instance (for backward compatibility)"""
    return GeminiClient()
