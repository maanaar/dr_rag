import pandas as pd
import json
import re
from typing import Dict, Any
from data_handling.data_loader import (
    load_doctor_data,
    extract_context_specialties,
    extract_context_doctors,
    extract_context_business_units,
)
from llm.fireworks_client import FireworksClient
from llm.llm_router import route_llm, generate_final_response
from search.search_functions import execute_search
from booking.booking_handler import book_appointment


def handle_user_query(client: FireworksClient, df: pd.DataFrame, query: str,
                     specialties: list, doctor_names: list, 
                     business_units: list, conversation_history: list = None,
                     last_search_args: dict = None, last_search_offset: int = 0) -> tuple:
    """
    Main handler for user queries.
    Returns tuple: (response_string, updated_history, search_args, search_offset)
    """
    try:
        # Check if user is asking for more results
        more_keywords = ["المزيد", "مزيد", "أكثر", "نتائج أخرى", "عرض المزيد", "أطباء آخرين", 
                        "more", "other", "another", "next", "التالي"]
        is_more_request = any(keyword in query.lower() for keyword in more_keywords)
        
        # Step 1: Send query to LLM with conversation history
        response = route_llm(client, query, specialties, doctor_names, business_units, 
                           conversation_history=conversation_history or [])
        
        if not response.choices:
            return "⚠️ لم يتم الحصول على رد من النموذج. يرجى المحاولة مرة أخرى."
        
        message = response.choices[0].message
        
        # Step 2: Check if LLM called any function
        tool_calls = getattr(message, "tool_calls", None)
        args = None
        function_name = None
        
        # Check for tool_calls first (proper function calling)
        if tool_calls:
            print(f"🔍 Found {len(tool_calls)} tool call(s)")
            # Process all tool calls
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                    print(f"🔍 Parsed {function_name} function arguments: {args}")
                    break  # Process first function call
                except json.JSONDecodeError as e:
                    print(f"⚠️ Error parsing function arguments: {e}")
                    args = {}
                    function_name = None
        
        # If no tool_calls, check if function call is in content (fallback)
        if not args and message.content:
            # Try to parse function call from content if it's JSON
            content_str = str(message.content).strip()
            if "search_doctors" in content_str or "book_appointment" in content_str:
                print(f"🔍 Function call found in content, attempting to parse...")
                print(f"   Content: {content_str}")
                
                # Try to parse the entire content as JSON first
                try:
                    func_data = json.loads(content_str)
                    if isinstance(func_data, dict) and "name" in func_data:
                        function_name = func_data["name"]
                        if "parameters" in func_data:
                            args = func_data["parameters"]
                            print(f"🔍 Extracted {function_name} arguments from JSON content: {args}")
                except json.JSONDecodeError:
                    # Try to extract JSON from content using regex
                    for func_name in ["search_doctors", "book_appointment"]:
                        json_match = re.search(r'\{[^{}]*"name"\s*:\s*"' + func_name + r'"[^{}]*\}', content_str)
                        if json_match:
                            try:
                                func_data = json.loads(json_match.group())
                                if "parameters" in func_data:
                                    args = func_data["parameters"]
                                    function_name = func_name
                                    print(f"🔍 Extracted {function_name} arguments from regex match: {args}")
                                    break
                            except json.JSONDecodeError:
                                pass
        
        # Handle booking function
        if function_name == "book_appointment" and args:
            print(f"📅 Processing booking request...")
            
            # Extract booking parameters
            doctor_name = args.get("doctor_name", "")
            patient_name = args.get("patient_name", "")
            patient_phone = args.get("patient_phone", "")
            appointment_date = args.get("appointment_date", "")
            appointment_time = args.get("appointment_time", "")
            speciality = args.get("speciality", "")
            business_unit = args.get("business_unit", "")
            notes = args.get("notes", "")
            
            # Validate required fields
            if not any([doctor_name, patient_name, patient_phone, appointment_date, appointment_time]):
                error_msg = "⚠️ يرجى تقديم جميع المعلومات المطلوبة للحجز: اسم الطبيب، اسم المريض، رقم الهاتف، التاريخ، والوقت."
                new_history = (conversation_history or []) + [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": error_msg}
                ]
                return error_msg, new_history, None, 0
            
            # Execute booking
            booking_result = book_appointment(
                doctor_name=doctor_name,
                patient_name=patient_name,
                patient_phone=patient_phone,
                appointment_date=appointment_date,
                appointment_time=appointment_time,
                df=df,
                speciality=speciality,
                business_unit=business_unit,
                notes=notes
            )
            
            # Format response
            if booking_result["success"]:
                response_text = f"""✅ {booking_result['message']}

📋 تفاصيل الحجز:
- رقم الحجز: {booking_result['appointment_id']}
- الطبيب: {booking_result['doctor_name']}
- المريض: {booking_result['patient_name']}
- التاريخ: {booking_result['appointment_date']}
- الوقت: {booking_result['appointment_time']}"""
                
                if booking_result.get('dr_notes'):
                    response_text += f"\n\n📝 ملاحظات الطبيب:\n{booking_result['dr_notes']}"
            else:
                # Check if it's a validation error
                if booking_result.get('error') == 'validation_failed':
                    response_text = f"⚠️ {booking_result['message']}"
                    if booking_result.get('suggestions'):
                        response_text += "\n\n💡 يمكنك اختيار أحد الخيارات المتاحة أعلاه."
                else:
                    response_text = f"❌ {booking_result['message']}"
            
            new_history = (conversation_history or []) + [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response_text}
            ]
            
            return response_text, new_history, None, 0
        
        # Execute search if we found arguments
        if function_name == "search_doctors" and args:
            # Normalize parameter names (handle both "specialty" and "speciality")
            normalized_args = {}
            for key, value in args.items():
                # Skip empty strings and offset (we handle offset separately)
                if key == "offset":
                    continue
                if not value or (isinstance(value, str) and value.strip() == ""):
                    continue
                    
                if key == "specialty":
                    normalized_args["speciality"] = value
                elif key == "speciality":
                    normalized_args["speciality"] = value
                elif key == "scope_of_service" or key == "scope_of_services" or key == "service":
                    normalized_args["scope_of_service"] = value
                else:
                    normalized_args[key] = value
            
            # Extract and handle offset separately (it might be in args)
            offset_raw = args.get("offset", 0)
            # Convert offset to int (handle both string and int)
            try:
                offset = int(offset_raw) if offset_raw else 0
            except (ValueError, TypeError):
                offset = 0
            
            # Handle pagination for "more results" requests
            search_args_for_storage = None
            if is_more_request and last_search_args:
                # Use previous search args with increased offset
                normalized_args = last_search_args.copy()
                offset = last_search_offset + 10
                search_args_for_storage = last_search_args  # Keep same args
                print(f"📄 Loading more results (offset: {offset})...")
            else:
                # New search - use offset from args or default to 0
                if offset > 0:
                    print(f"📄 Loading results from offset: {offset}...")
                else:
                    offset = 0
                # Store search args for future "more" requests (without offset)
                search_args_for_storage = normalized_args.copy()
            
            print(f"🔍 Executing search with args: {normalized_args}")
            
            # Execute the search with user query for similarity ranking
            search_results = execute_search(df, normalized_args, user_query=query, offset=offset, limit=10)
            
            # Extract metadata before removing it
            total_results = 0
            has_more = False
            if search_results:
                total_results = search_results[0].get('_total_results', len(search_results))
                has_more = search_results[0].get('_has_more', False)
                
                # Remove metadata from results before displaying
                for result in search_results:
                    result.pop('_total_results', None)
                    result.pop('_current_offset', None)
                    result.pop('_has_more', None)
            
            print(f"✅ Found {len(search_results)} results (showing {offset + 1}-{offset + len(search_results)} of {total_results})")
            
            # Step 3: Send results back to LLM for natural formatting
            if search_results:
                # Add pagination info to prompt
                pagination_info = ""
                if has_more:
                    pagination_info = f"\n\nملاحظة: يوجد {total_results - offset - len(search_results)} طبيب/أطباء إضافيين. يمكن للمستخدم طلب المزيد."
                
                final_response = generate_final_response(client, query, search_results, pagination_info)
                
                # Update conversation history
                new_history = (conversation_history or []) + [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": final_response}
                ]
                
                # Store search args for next "more" request
                # Use the stored args (without offset) for future pagination
                return final_response, new_history, search_args_for_storage, offset
            else:
                error_msg = "⚠️ لم يتم العثور على أطباء مطابقين لبحثك. يرجى المحاولة بكلمات مختلفة."
                new_history = (conversation_history or []) + [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": error_msg}
                ]
                return error_msg, new_history, None, 0
        
        # Step 4: No function call - return text response directly
        if message.content:
            response_text = message.content
            new_history = (conversation_history or []) + [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response_text}
            ]
            return response_text, new_history, None, 0
        
        error_msg = "⚠️ لم يتم الحصول على رد نصي من النموذج."
        new_history = (conversation_history or []) + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": error_msg}
        ]
        return error_msg, new_history, None, 0
    
    except ValueError as e:
        # API key or configuration errors
        error_msg = str(e)
        if "API" in error_msg or "مفتاح" in error_msg:
            print(f"\n{error_msg}\n")
            new_history = (conversation_history or []) + [
                {"role": "user", "content": query},
                {"role": "assistant", "content": error_msg}
            ]
            return error_msg, new_history, None, 0
        raise
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
            return msg, new_history, None, 0
        
        print(f"❌ Error in handle_user_query: {e}")
        import traceback
        traceback.print_exc()
        error_response = f"⚠️ حدث خطأ أثناء معالجة استفسارك: {str(e)}"
        new_history = (conversation_history or []) + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": error_response}
        ]
        return error_response, new_history, None, 0


def main():
    """Main execution function"""
    print("🔄 Loading data...")
    df = load_doctor_data()
    
    if df.empty:
        print("❌ No data loaded. Please implement load_doctor_data() in data_handling/data_loader.py")
        print("   Expected columns: Doctor Name, Speciality Description Arabic, BU Arabic List")
        return
    
    print("📊 Extracting context...")
    specialties = extract_context_specialties(df)
    doctor_names = extract_context_doctors(df)
    business_units = extract_context_business_units(df)
    
    print(f"✅ Loaded {len(df)} doctors, {len(specialties)} specialties, {len(business_units)} business units")
    
    print("🤖 Initializing Fireworks AI client...")
    try:
        client = FireworksClient()
    except ValueError as e:
        print(f"❌ {e}")
        print("   Please set FIREWORKS_API_KEY in your .env file")
        print("   Get your API key from: https://fireworks.ai/")
        return
    
    print("\n" + "="*60)
    print("مساعد البحث عن الأطباء - Doctor Search Assistant")
    print("اكتب 'exit' أو 'quit' للخروج")
    print("اكتب 'المزيد' أو 'نتائج أخرى' لعرض المزيد من النتائج")
    print("="*60 + "\n")
    
    # Session state
    conversation_history = []
    last_search_args = None
    last_search_offset = 0
    
    while True:
        try:
            query = input("👤 أنت: ").strip()
            
            if query.lower() in ["exit", "quit", "خروج"]:
                print("👋 مع السلامة!")
                break
            
            if not query:
                continue
            
            print("🤔 جاري المعالجة...")
            response, conversation_history, last_search_args, last_search_offset = handle_user_query(
                client, df, query, specialties, doctor_names, business_units,
                conversation_history=conversation_history,
                last_search_args=last_search_args,
                last_search_offset=last_search_offset
            )
            print(f"\n🤖 المساعد:\n{response}\n")
            
        except KeyboardInterrupt:
            print("\n\n مع السلامة")
            break
        except Exception as e:
            print(f"\n❌ خطأ: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()