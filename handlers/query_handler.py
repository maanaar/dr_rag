# handlers/query_handler.py

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from llm.fireworks_client import FireworksClient
from llm.llm_router import route_llm
from handlers.function_parser import parse_function_call
from handlers.search_handler import handle_search
from handlers.booking_request_handler import handle_booking


def is_function_call_content(content: str) -> bool:
    """Check if content looks like a function call JSON"""
    if not content:
        return False
    content_str = str(content).strip()
    return (
        ("search_doctors" in content_str or "book_appointment" in content_str) and
        ('"type": "function"' in content_str or '"name":' in content_str)
    )


def handle_user_query(client: FireworksClient, df: pd.DataFrame, query: str,
                     specialties: list, doctor_names: list, 
                     business_units: list, conversation_history: List[Dict[str, str]] = None) -> Tuple[str, List[Dict[str, str]], Optional[str]]:
    """
    Main handler for user queries.
    Returns (response_string, updated_history, first_speciality)
    """
    try:
        # Step 1: Send query to LLM with conversation history
        response = route_llm(client, query, specialties, doctor_names, business_units, 
                           conversation_history=conversation_history or [])
        
        if not response.choices:
            error_msg = "⚠️ لم يتم الحصول على رد من النموذج. يرجى المحاولة مرة أخرى."
            new_history = (conversation_history or []) + [
                {"role": "user", "content": query},
                {"role": "assistant", "content": error_msg}
            ]
            return error_msg, new_history, None
        
        message = response.choices[0].message
        
        # Step 2: Parse function call if any
        function_name, args = parse_function_call(message)
        
        # Step 3: Handle booking function
        if function_name == "book_appointment" and args:
            response_text, new_history = handle_booking(df, query, args, conversation_history or [])
            return response_text, new_history, None
        
        # Step 4: Handle search function
        if function_name == "search_doctors" and args:
            # Extract speciality from search args
            speciality = args.get("speciality") or args.get("specialty")
            response_text, new_history = handle_search(client, df, query, args, conversation_history or [])
            return response_text, new_history, speciality
        
        # Step 5: No function call - return text response directly
        if message.content:
            content_str = str(message.content).strip()
            
            # Check if content looks like a function call that was filtered out
            if is_function_call_content(content_str):
                print("⚠️ Detected function call in content but it was filtered - generating casual response")
                # Generate a friendly casual response
                response_text = "أهلاً وسهلاً! كيف يمكنني مساعدتك اليوم؟ يمكنني مساعدتك في إيجاد الأطباء المناسبين حسب التخصص أو الأعراض."
            else:
                response_text = content_str
            
            new_history = (conversation_history or []) + [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response_text}
            ]
            return response_text, new_history, None
        
        # If we reach here, something went wrong
        error_msg = "⚠️ لم يتم الحصول على رد نصي من النموذج."
        new_history = (conversation_history or []) + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": error_msg}
        ]
        return error_msg, new_history, None
    
    except ValueError as e:
        # API key or configuration errors
        error_msg = str(e)
        if "API" in error_msg or "مفتاح" in error_msg:
            print(f"\n{error_msg}\n")
        new_history = (conversation_history or []) + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": error_msg}
        ]
        return error_msg, new_history, None
    except Exception as e:
        error_msg = str(e)
        # Check for API key errors
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            msg = (
                "🔒 خطأ في المصادقة.\n\n"
                "📝 الحل:\n"
                "1. اذهب إلى https://fireworks.ai/\n"
                "2. أنشئ مفتاح API جديد\n"
                "3. قم بتحديث FIREWORKS_API_KEY في ملف .env\n"
                "4. أعد تشغيل التطبيق"
            )
            print(f"\n{msg}\n")
            new_history = (conversation_history or []) + [
                {"role": "user", "content": query},
                {"role": "assistant", "content": msg}
            ]
            return msg, new_history, None
        
        print(f"❌ Error in handle_user_query: {e}")
        import traceback
        traceback.print_exc()
        error_response = f"⚠️ حدث خطأ أثناء معالجة استفسارك: {str(e)}"
        new_history = (conversation_history or []) + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": error_response}
        ]
        return error_response, new_history, None