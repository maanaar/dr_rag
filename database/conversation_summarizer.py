# database/conversation_summarizer.py

from typing import List, Dict, Any
from llm.fireworks_client import FireworksClient
from openai import OpenAIError


def summarize_conversation(client: FireworksClient, 
                           conversation_history: List[Dict[str, str]]) -> str:
    """
    Generate a summary of the conversation history using the LLM.
    
    Args:
        client: FireworksClient instance
        conversation_history: List of messages with 'role' and 'content' keys
    
    Returns:
        A summary string in Arabic
    """
    if not conversation_history:
        return "لا توجد محادثة لتلخيصها."
    
    # Format conversation for summarization
    conversation_text = ""
    for msg in conversation_history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "user":
            conversation_text += f"المستخدم: {content}\n\n"
        elif role == "assistant":
            conversation_text += f"المساعد: {content}\n\n"
    
    # Create summarization prompt
    system_prompt = """أنت مساعد ذكي مهمتك تلخيص المحادثات الطبية. قم بإنشاء ملخص موجز وواضح باللغة العربية يتضمن:
1. المواضيع الرئيسية التي تمت مناقشتها
2. أي عمليات بحث عن أطباء تمت
3. أي حجوزات مواعيد تمت
4. المعلومات المهمة الأخرى

اجعل الملخص واضحًا ومفيدًا للرجوع إليه لاحقًا."""
    
    user_prompt = f"""يرجى تلخيص المحادثة التالية:

{conversation_text}

قم بإنشاء ملخص موجز وواضح باللغة العربية."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        # Generate summary without tools
        response = client.generate_content(messages, tools=[], tool_choice="none")
        
        if response.choices and response.choices[0].message.content:
            summary = response.choices[0].message.content.strip()
            return summary
        else:
            return "فشل في إنشاء الملخص."
    
    except OpenAIError as e:
        return f"خطأ في إنشاء الملخص: {str(e)}"
    except Exception as e:
        return f"حدث خطأ غير متوقع: {str(e)}"


def extract_conversation_metadata(conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Extract metadata from conversation history.
    
    Args:
        conversation_history: List of messages
    
    Returns:
        Dictionary with metadata (topics, actions, etc.)
    """
    import json
    import re
    
    metadata = {
        "total_messages": len(conversation_history),
        "user_messages": sum(1 for msg in conversation_history if msg.get("role") == "user"),
        "assistant_messages": sum(1 for msg in conversation_history if msg.get("role") == "assistant"),
        "has_search": False,
        "has_booking": False,
        "first_speciality": None,
        "topics": []
    }
    
    # Check for search and booking actions, and extract first speciality
    for msg in conversation_history:
        content = str(msg.get("content", "")).lower()
        
        # Check for search actions
        if "search" in content or "بحث" in content or "دكتور" in content:
            metadata["has_search"] = True
            
            # Try to extract speciality from function call JSON in content
            if not metadata["first_speciality"]:
                # Look for JSON function calls with speciality parameter
                try:
                    # Try to parse as JSON first
                    if "{" in content and "speciality" in content:
                        json_match = re.search(r'\{[^{}]*"speciality"[^{}]*\}', content)
                        if json_match:
                            try:
                                func_data = json.loads(json_match.group())
                                speciality = func_data.get("speciality") or func_data.get("specialty")
                                if speciality:
                                    metadata["first_speciality"] = speciality
                            except json.JSONDecodeError:
                                pass
                except Exception:
                    pass
        
        # Check for booking actions
        if "booking" in content or "حجز" in content or "موعد" in content:
            metadata["has_booking"] = True
    
    return metadata

