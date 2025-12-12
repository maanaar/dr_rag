import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file from the dr_rag directory (parent of llm directory)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Fireworks AI model - you can change this to any model available on Fireworks
# Available models for your account:
# - accounts/fireworks/models/llama-v3p3-70b-instruct (recommended)
# - accounts/fireworks/models/deepseek-v3-0324
# - accounts/fireworks/models/flux-1-dev-fp8
# Check all available models at: https://fireworks.ai/models
FIREWORKS_MODEL = "accounts/fireworks/models/llama-v3p3-70b-instruct"


class FireworksClient:
    """Wrapper for Fireworks AI client with function calling support"""
    
    def __init__(self):
        api_key = os.getenv("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY not found in environment variables")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.fireworks.ai/inference/v1"
        )
        self.model = FIREWORKS_MODEL
        self.search_tool = self._create_search_tool()
        self.booking_tool = self._create_booking_tool()
    
    def _create_search_tool(self):
        """Create the search_doctors function declaration for OpenAI-style function calling"""
        return {
            "type": "function",
            "function": {
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
                        "scope_of_service": {
                            "type": "string",
                            "description": "نطاق الخدمة أو الخدمة المطلوبة (مثل: تنظير، جراحة، علاج، تشخيص)"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "عدد النتائج المراد تخطيها (للعرض التالي). استخدم 0 للصفحة الأولى، 10 للثانية، 20 للثالثة، إلخ."
                        },
                    },
                    "required": []
                }
            }
        }
    
    def _create_booking_tool(self):
        """Create the book_appointment function declaration"""
        return {
            "type": "function",
            "function": {
                "name": "book_appointment",
                "description": "حجز موعد مع طبيب. استخدم هذه الوظيفة عندما يطلب المستخدم حجز موعد.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doctor_name": {
                            "type": "string",
                            "description": "اسم الطبيب المطلوب الحجز معه"
                        },
                        "patient_name": {
                            "type": "string",
                            "description": "اسم المريض"
                        },
                        "patient_phone": {
                            "type": "string",
                            "description": "رقم هاتف المريض"
                        },
                        "appointment_date": {
                            "type": "string",
                            "description": "تاريخ الموعد بصيغة YYYY-MM-DD (مثل: 2024-12-25)"
                        },
                        "appointment_time": {
                            "type": "string",
                            "description": "وقت الموعد بصيغة HH:MM (مثل: 14:30)"
                        },
                        "speciality": {
                            "type": "string",
                            "description": "تخصص الطبيب (اختياري)"
                        },
                        "business_unit": {
                            "type": "string",
                            "description": "الوحدة الطبية (اختياري)"
                        },
                        "notes": {
                            "type": "string",
                            "description": "ملاحظات إضافية (اختياري)"
                        }
                    },
                    "required": ["doctor_name", "patient_name", "patient_phone", "appointment_date", "appointment_time"]
                }
            }
        }
    
    def generate_content(self, messages, tools=None, tool_choice="auto"):
        """
        Generate content using Fireworks AI.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: List of tool definitions. If None, uses search_tool. If empty list [], no tools.
            tool_choice: "auto", "none", or specific tool name
        """
        # Prepare the request
        request_params = {
            "model": self.model,
            "messages": messages,
        }
        
        # Handle tools: None means use default, [] means no tools
        if tools is None:
            # Default: use both search and booking tools
            request_params["tools"] = [self.search_tool, self.booking_tool]
            request_params["tool_choice"] = tool_choice
        elif tools == []:
            # Empty list: explicitly no tools
            # Don't add tools parameter at all
            pass
        else:
            # Custom tools provided
            request_params["tools"] = tools
            request_params["tool_choice"] = tool_choice
        
        response = self.client.chat.completions.create(**request_params)
        return response

