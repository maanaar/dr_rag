
from typing import List, Dict, Any
from google.genai import types
from google.genai.errors import ClientError
from llm.gemini_client import GeminiClient


def create_system_instructions(specialties: List[str], doctor_names: List[str], 
                               business_units: List[str]) -> str:
    """Create system instructions for the LLM"""
    return f"""أنت مساعد طبي ذكي لمساعدة المرضى في العثور على الأطباء المناسبين.

**البيانات المتاحة:**
- التخصصات المتاحة: {specialties}
- عدد الأطباء: {len(doctor_names)}
- الوحدات الطبية: {business_units}

**التعليمات:**
1. إذا ذكر المستخدم اسم طبيب، تخصص، أو وحدة طبية → استدعِ وظيفة search_doctors مباشرة
2. إذا ذكر المستخدم عرضًا طبيًا (مثل: ألم في الصدر، صداع، مشاكل في الهضم):
   - حدد التخصص المناسب من القائمة أعلاه
   - استدعِ search_doctors مع التخصص المناسب
3. إذا كان الاستفسار عامًا جدًا أو غير طبي → أجب مباشرة دون استدعاء وظيفة
4. لا تطلب تأكيدًا من المستخدم، اتخذ القرار مباشرة

**أمثلة على تصنيف الأعراض:**
- "قلبي يؤلمني" → التخصص: "القلب" أو "أمراض القلب"
- "عندي صداع مستمر" → التخصص: "المخ والأعصاب" أو "طب الأعصاب"
- "مشاكل في الهضم" → التخصص: "الجهاز الهضمي"
- "ألم في المفاصل" → التخصص: "العظام" أو "الروماتيزم"

رد دائمًا باللغة العربية بشكل طبيعي ومفيدة."""


def route_llm(client: GeminiClient, query: str, specialties: List[str], 
              doctor_names: List[str], business_units: List[str]) -> types.GenerateContentResponse:
    """
    Route user query to Gemini LLM.
    Returns the raw response for processing.
    """
    instructions = create_system_instructions(specialties, doctor_names, business_units)
    
    system_content = types.Content(
        parts=[types.Part(text=instructions)], 
        role="model"
    )
    user_content = types.Content(
        parts=[types.Part(text=query)], 
        role="user"
    )
    
    try:
        response = client.generate_content(
            contents=[system_content, user_content]
        )
        return response
    except ClientError as e:
        # Check for API key issues
        error_msg = str(e)
        if "403" in error_msg and "leaked" in error_msg.lower():
            raise ValueError(
                "🔒 مفتاح API الخاص بك تم الإبلاغ عنه كمسرب. يرجى إنشاء مفتاح API جديد من:\n"
                "   https://aistudio.google.com/app/apikey\n"
                "   ثم قم بتحديث GEMINI_API_KEY في ملف .env"
            ) from e
        elif "403" in error_msg:
            raise ValueError(
                "🔒 خطأ في الصلاحيات (403). يرجى التحقق من صحة مفتاح API الخاص بك.\n"
                "   تأكد من أن GEMINI_API_KEY في ملف .env صحيح."
            ) from e
        print(f"❌ Error calling Gemini API: {e}")
        raise
    except Exception as e:
        print(f"❌ Error calling Gemini API: {e}")
        raise


def format_results_for_llm(results: List[Dict[str, Any]]) -> str:
    """Format search results into Arabic text for LLM to present naturally"""
    if not results:
        return "لم يتم العثور على أطباء مطابقين للبحث."
    
    formatted = f"تم العثور على {len(results)} طبيب/أطباء:\n\n"
    
    for i, doctor in enumerate(results[:10], 1):  # Limit to top 10
        formatted += f"{i}. **{doctor.get('Doctor Name', 'غير متوفر')}**\n"
        formatted += f"   - التخصص: {doctor.get('Speciality Description Arabic', 'غير متوفر')}\n"
        
        bus = doctor.get('BU Arabic List', [])
        if bus and isinstance(bus, list):
            formatted += f"   - الوحدات: {', '.join(bus)}\n"
        
        formatted += "\n"
    
    if len(results) > 10:
        formatted += f"... وهناك {len(results) - 10} طبيب/أطباء آخرين.\n"
    
    return formatted


def generate_final_response(client: GeminiClient, query: str, 
                           search_results: List[Dict[str, Any]]) -> str:
    """
    Send search results back to LLM for natural Arabic formatting.
    """
    formatted_results = format_results_for_llm(search_results)
    
    prompt = f"""استفسار المستخدم الأصلي: "{query}"

نتائج البحث:
{formatted_results}

قدم هذه النتائج للمستخدم بطريقة طبيعية ومفيدة باللغة العربية. يمكنك إضافة نصائح أو معلومات إضافية إذا كانت مناسبة."""

    try:
        response = client.generate_content(
            contents=[types.Content(parts=[types.Part(text=prompt)], role="user")],
            config=None  # Don't use tools for final response
        )
        
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        
        return formatted_results  # Fallback
    except ClientError as e:
        error_msg = str(e)
        if "403" in error_msg and "leaked" in error_msg.lower():
            print("⚠️ مفتاح API مسرب - يرجى تحديثه في ملف .env")
        print(f"⚠️ Error generating final response: {e}")
        return formatted_results  # Fallback
    except Exception as e:
        print(f"⚠️ Error generating final response: {e}")
        return formatted_results  # Fallback
