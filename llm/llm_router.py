from typing import List, Dict, Any, Optional
from openai import OpenAIError
from llm.fireworks_client import FireworksClient
from data_handling.business_unit_map import map_business_unit


def create_system_instructions(specialties: List[str], doctor_names: List[str], 
                               business_units: List[str]) -> str:
    """Create system instructions for the LLM"""
    return f"""أنت مساعد طبي ذكي لمساعدة المرضى في العثور على الأطباء المناسبين.


**البيانات المتاحة:**
- التخصصات المتاحة: {specialties}
- عدد الأطباء: {len(doctor_names)}
- الوحدات الطبية: {business_units}

**قواعد استخدام الوظائف - اقرأ بعناية:**

⚠️ **لا تستخدم أي وظيفة في هذه الحالات:**
- التحيات: أهلا، مرحبا، السلام عليكم، صباح الخير، مساء الخير
- الشكر: شكراً، يعطيك العافية، تسلم
- الوداع: باي، وداعاً، مع السلامة
- أسئلة عامة: ما اسمك، كيف حالك، مين أنت
- أي محادثة عامة غير طبية

 **استخدم search_doctors فقط عندما:**
- يذكر المستخدم صراحة: اسم طبيب، تخصص طبي، وحدة طبية
- يذكر المستخدم عرضاً طبياً واضحاً (ألم، مرض، شكوى صحية)
- يطلب توصية طبيب لخدمة معينة

 **استخدم book_appointment فقط عندما:**
- يطلب المستخدم صراحة حجز موعد

**أمثلة واضحة:**
 "أهلا" → رد مباشرة: "أهلاً وسهلاً! كيف يمكنني مساعدتك؟" (لا وظيفة)
 "شكراً" → رد مباشرة: "العفو!" (لا وظيفة)
 "كيف حالك" → رد مباشرة: "بخير، شكراً! كيف أساعدك؟" (لا وظيفة)
 "عندي ألم في قلبي" → استدعِ search_doctors مع التخصص: "القلب"
 "أريد دكتور أسنان" → استدعِ search_doctors مع التخصص: "الأسنان"

**مهم جداً:** إذا لم يكن الاستفسار واضحاً طبياً، لا تستخدم الوظائف!

رد دائمًا باللغة العربية بشكل طبيعي."""

class FireworksResponse:
    """Wrapper to make Fireworks response compatible with existing code"""
    def __init__(self, response):
        self.response = response
        self.choices = response.choices if hasattr(response, 'choices') else []
    
    @property
    def candidates(self):
        """Compatibility property for existing code"""
        return self.choices
    
    def get_message(self):
        """Get the message from the first choice"""
        if self.choices:
            return self.choices[0].message
        return None


def route_llm(client: FireworksClient, query: str, specialties: List[str], 
              doctor_names: List[str], business_units: List[str], 
              conversation_history: List[Dict[str, str]] = None) -> FireworksResponse:
    """
    Route user query to Fireworks LLM.
    Returns the raw response for processing.
    """
    instructions = create_system_instructions(specialties, doctor_names, business_units)
    
    messages = [
        {"role": "system", "content": instructions}
    ]
    
    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history)
    
    # Add current query
    messages.append({"role": "user", "content": query})
    
    try:
        response = client.generate_content(messages, tools=[client.search_tool], tool_choice="auto")
        return FireworksResponse(response)
    except OpenAIError as e:
        # Check for API key issues
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            raise ValueError(
                "🔒 خطأ في المصادقة (401). يرجى التحقق من صحة مفتاح API الخاص بك.\n"
                "   تأكد من أن FIREWORKS_API_KEY في ملف .env صحيح.\n"
                "   احصل على مفتاح من: https://fireworks.ai/"
            ) from e
        elif "403" in error_msg or "forbidden" in error_msg.lower():
            raise ValueError(
                "🔒 خطأ في الصلاحيات (403). يرجى التحقق من صحة مفتاح API الخاص بك.\n"
                "   تأكد من أن FIREWORKS_API_KEY في ملف .env صحيح."
            ) from e
        print(f"❌ Error calling Fireworks API: {e}")
        raise
    except Exception as e:
        print(f"❌ Error calling Fireworks API: {e}")
        raise


def format_results_for_llm(results: List[Dict[str, Any]]) -> str:
    """Format search results into Arabic text for LLM to present naturally"""
    if not results:
        return "لم يتم العثور على أطباء مطابقين للبحث."
    
    formatted = f"تم العثور على {len(results)} طبيب/أطباء:\n\n"
    
    for i, doctor in enumerate(results, 1):  # Limit to top 10
        formatted += f"{i}. **{doctor.get('Doctor Name', 'غير متوفر')}**\n"
        formatted += f"   - التخصص: {doctor.get('Speciality Description Arabic', 'غير متوفر')}\n"
        
        bus = doctor.get('BU Arabic List', [])
        if bus and isinstance(bus, list):
            # Ensure all business units are mapped (apply mapping to any unmapped codes)
            mapped_bus = [map_business_unit(str(bu)) if isinstance(bu, str) and len(bu) <= 10 and bu.isalnum() else str(bu) for bu in bus]
            # Remove duplicates while preserving order
            seen = set()
            unique_bus = []
            for bu in mapped_bus:
                if bu not in seen:
                    seen.add(bu)
                    unique_bus.append(bu)
            if unique_bus:
                formatted += f"   - الوحدات: {', '.join(unique_bus)}\n"
        
        # Add scope of service if available
        scope = doctor.get('Scope of Service Arabic', '')
        if scope and str(scope).strip():
            # Limit scope text length for display
            scope_text = str(scope).strip()
            if len(scope_text) > 150:
                scope_text = scope_text[:150] + "..."
            formatted += f"   - نطاق الخدمة: {scope_text}\n"
        
        formatted += "\n"
    
    if len(results) > 10:
        formatted += f"... وهناك {len(results) - 10} طبيب/أطباء آخرين.\n"
    
    return formatted


def generate_final_response(client: FireworksClient, query: str, 
                           search_results: List[Dict[str, Any]], 
                           pagination_info: str = "") -> str:
    """
    Send search results back to LLM for natural Arabic formatting.
    """
    formatted_results = format_results_for_llm(search_results)
    
    prompt = f"""أنت مساعد طبي. قدم نتائج البحث عن الأطباء للمستخدم بطريقة طبيعية ومفيدة باللغة العربية.

استفسار المستخدم: "{query}"

نتائج البحث:
{formatted_results}
{pagination_info}

قم بتقديم هذه النتائج للمستخدم بشكل واضح ومنظم. لا تستدعي أي وظائف، فقط قدم النتائج كنص عادي."""

    try:
        messages = [
            {"role": "system", "content": "أنت مساعد طبي. قدم المعلومات بشكل واضح ومفيد بالعربية. لا تستخدم وظائف، فقط قدم النص."},
            {"role": "user", "content": prompt}
        ]
        
        # Explicitly disable tools for final response
        response = client.generate_content(
            messages, 
            tools=[],  # Empty list instead of None
            tool_choice="none"
        )
        
        message = response.choices[0].message if response.choices else None
        
        # Check if response contains function call (shouldn't happen, but handle it)
        if message:
            # Check for tool_calls first
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                print("⚠️ Warning: LLM tried to call function in final response, using fallback")
                return formatted_results
            
            # Check if content contains function call JSON
            if message.content:
                content_str = str(message.content).strip()
                if "search_doctors" in content_str or '"type": "function"' in content_str:
                    print("⚠️ Warning: Function call found in final response content, using fallback")
                    return formatted_results
                
                return message.content
        
        return formatted_results  # Fallback
    except Exception as e:
        print(f"⚠️ Error generating final response: {e}")
        import traceback
        traceback.print_exc()
        return formatted_results  # Fallback