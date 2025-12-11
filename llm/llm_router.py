from llm.gemini_client import get_gemini_client
from google.genai import types
from search.search_functions import execute_search

def route_llm(query, specialties, doctor_names, business_units, tool_result=None):
    """
    Handles:
    1) Normal user queries -> LLM decides whether to call a function
    2) Returning tool results -> LLM generates final Arabic output
    """
    client, config = get_gemini_client()

    # Step 1: If tool_result exists, send it to LLM for final Arabic response
    if tool_result is not None:
        tool_msg = types.Content(
            text=str(tool_result), role="tool"
        )
        return client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[tool_msg],
            config=config
        )

    # Step 2: Normal query
    instructions = f"""
أنت مساعد طبي ذكي.
التخصصات المتاحة: {specialties}
الأطباء: {doctor_names}
الوحدات الطبية: {business_units}

1) إذا ذكر المستخدم اسم طبيب، تخصص، أو وحدة → استدعِ search_doctors تلقائيًا.
2) إذا ذكر المستخدم عرضًا طبيًا أو شكوى ولم يذكر تخصصًا، صنف العرض إلى أقرب تخصص من القائمة أعلاه.
3) إذا الاستفسار عام جدًا → لا تستدعي أي وظيفة.
4) لا تطلب من المستخدم تأكيد أو إعادة صياغة، استدعِ search_doctors مباشرة عند الحاجة.
5) رد باللغة العربية بشكل طبيعي.

مثال:
- المستخدم: "قلبي نغزت"
- استنتاجك: التخصص = "القلب"
- استدعاء search_doctors مع {{"speciality": "القلب"}}
"""

    sys_content = types.Content(parts=[types.Part(text=instructions)], role="model")
    user_content = types.Content(parts=[types.Part(text=query)], role="user")

    return client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[sys_content, user_content],
        config=config
    )