import pandas as pd
from typing import Dict, Any
from data_handling.data_loader import (
    load_doctor_data,
    extract_context_specialties,
    extract_context_doctors,
    extract_context_business_units,
)
from llm.gemini_client import GeminiClient
from llm.llm_router import route_llm, generate_final_response
from search.search_functions import execute_search


def handle_user_query(client: GeminiClient, df: pd.DataFrame, query: str,
                     specialties: list, doctor_names: list, 
                     business_units: list) -> str:
    """
    Main handler for user queries.
    Returns the final response as a string.
    """
    try:
        # Step 1: Send query to LLM
        response = route_llm(client, query, specialties, doctor_names, business_units)
        
        if not response.candidates:
            return "⚠️ لم يتم الحصول على رد من النموذج. يرجى المحاولة مرة أخرى."
        
        candidate = response.candidates[0]
        
        # Check if we have content
        if not candidate.content or not candidate.content.parts:
            return "⚠️ لم يتم الحصول على محتوى في الرد."
        
        part = candidate.content.parts[0]
        
        # Step 2: Check if LLM called the search function
        function_call = getattr(part, "function_call", None)
        
        if function_call and function_call.name == "search_doctors":
            print(f"🔍 Executing search with args: {dict(function_call.args)}")
            
            # Execute the search
            search_results = execute_search(df, dict(function_call.args))
            print(f"✅ Found {len(search_results)} results")
            
            # Step 3: Send results back to LLM for natural formatting
            final_response = generate_final_response(client, query, search_results)
            return final_response
        
        # Step 4: No function call - return text response directly
        if hasattr(part, "text") and part.text:
            return part.text
        
        return "⚠️ لم يتم الحصول على رد نصي من النموذج."
    
    except ValueError as e:
        # API key or configuration errors
        error_msg = str(e)
        if "API" in error_msg or "مفتاح" in error_msg:
            print(f"\n{error_msg}\n")
            return error_msg
        raise
    except Exception as e:
        error_msg = str(e)
        # Check for API key leaked error in the message
        if "403" in error_msg and ("leaked" in error_msg.lower() or "مسرب" in error_msg):
            msg = (
                "🔒 مفتاح API الخاص بك تم الإبلاغ عنه كمسرب.\n\n"
                "📝 الحل:\n"
                "1. اذهب إلى https://aistudio.google.com/app/apikey\n"
                "2. أنشئ مفتاح API جديد\n"
                "3. قم بتحديث GEMINI_API_KEY في ملف .env\n"
                "4. أعد تشغيل التطبيق"
            )
            print(f"\n{msg}\n")
            return msg
        
        print(f"❌ Error in handle_user_query: {e}")
        import traceback
        traceback.print_exc()
        return f"⚠️ حدث خطأ أثناء معالجة استفسارك: {str(e)}"


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
    
    print("🤖 Initializing Gemini client...")
    try:
        client = GeminiClient()
    except ValueError as e:
        print(f"❌ {e}")
        print("   Please set GEMINI_API_KEY in your .env file")
        return
    
    print("\n" + "="*60)
    print("مساعد البحث عن الأطباء - Doctor Search Assistant")
    print("اكتب 'exit' أو 'quit' للخروج")
    print("="*60 + "\n")
    
    while True:
        try:
            query = input("👤 أنت: ").strip()
            
            if query.lower() in ["exit", "quit", "خروج"]:
                print("👋 مع السلامة!")
                break
            
            if not query:
                continue
            
            print("🤔 جاري المعالجة...")
            response = handle_user_query(
                client, df, query, specialties, doctor_names, business_units
            )
            print(f"\n🤖 المساعد:\n{response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 مع السلامة!")
            break
        except Exception as e:
            print(f"\n❌ خطأ: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()