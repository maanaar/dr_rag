# app.py - Main entry point

import pandas as pd
import uuid
from datetime import datetime
from data_handling.data_loader import (
    load_doctor_data,
    extract_context_specialties,
    extract_context_doctors,
    extract_context_business_units,
)
from llm.fireworks_client import FireworksClient
from handlers.query_handler import handle_user_query
from database.conversation_db import save_conversation_summary, init_database
from database.conversation_summarizer import summarize_conversation, extract_conversation_metadata


def main():
    """Main execution function"""
    print("🔄 Loading data...")
    df = load_doctor_data()
    
    if df.empty:
        print("❌ No data loaded. Please check data_handling/data_loader.py")
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
    print("="*60 + "\n")
    
    # Initialize database
    init_database()
    
    # Session state - conversation history and session ID
    conversation_history = []
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    first_speciality = None  # Track the first speciality chosen by LLM
    
    while True:
        try:
            query = input("👤 أنت: ").strip()
            
            if query.lower() in ["exit", "quit", "خروج"]:
                # Save conversation summary before exiting
                if conversation_history:
                    try:
                        print("\n📝 جاري حفظ ملخص المحادثة...")
                        summary = summarize_conversation(client, conversation_history)
                        metadata = extract_conversation_metadata(conversation_history)
                        
                        # Use tracked first_speciality or extract from metadata
                        speciality_to_save = first_speciality or metadata.get("first_speciality")
                        
                        # Extract last bot response
                        last_response = None
                        for msg in reversed(conversation_history):
                            if msg.get("role") == "assistant":
                                last_response = msg.get("content", "")
                                break
                        
                        record_id = save_conversation_summary(
                            session_id=session_id,
                            summary=summary,
                            conversation_history=conversation_history,
                            metadata=metadata,
                            last_bot_response=last_response,
                            speciality=speciality_to_save
                        )
                        print(f"✅ تم حفظ الملخص في قاعدة البيانات (ID: {record_id})")
                    except Exception as e:
                        print(f"⚠️ فشل حفظ الملخص: {e}")
                
                print("👋 مع السلامة!")
                break
            
            if not query:
                continue
            
            print("🤔 جاري المعالجة...")
            response, conversation_history, speciality_from_query = handle_user_query(
                client, df, query, specialties, doctor_names, business_units,
                conversation_history=conversation_history
            )
            
            # Track first speciality from search function calls
            if not first_speciality and speciality_from_query:
                first_speciality = speciality_from_query
            
            print(f"\n🤖 المساعد:\n{response}\n")
            
        except KeyboardInterrupt:
            # Save conversation summary before exiting
            if conversation_history:
                try:
                    print("\n\n📝 جاري حفظ ملخص المحادثة...")
                    summary = summarize_conversation(client, conversation_history)
                    metadata = extract_conversation_metadata(conversation_history)
                    
                    # Use tracked first_speciality or extract from metadata
                    speciality_to_save = first_speciality or metadata.get("first_speciality")
                    
                    # Extract last bot response
                    last_response = None
                    for msg in reversed(conversation_history):
                        if msg.get("role") == "assistant":
                            last_response = msg.get("content", "")
                            break
                    
                    record_id = save_conversation_summary(
                        session_id=session_id,
                        summary=summary,
                        conversation_history=conversation_history,
                        metadata=metadata,
                        last_bot_response=last_response,
                        speciality=speciality_to_save
                    )
                    print(f"✅ تم حفظ الملخص في قاعدة البيانات (ID: {record_id})")
                except Exception as e:
                    print(f"⚠️ فشل حفظ الملخص: {e}")
            
            print("\n\n👋 مع السلامة!")
            break
        except Exception as e:
            print(f"\n❌ خطأ: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
