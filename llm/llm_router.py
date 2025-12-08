# llm/llm_router.py

from llm.gemini_client import get_gemini_client


def route_llm(query, specialties, doctor_names, business_units):
    instructions = f"""
    You are an assistant that must classify Arabic user queries.

    You have 3 things to detect:

    1) Doctor Name → from this list:
    {doctor_names}

    2) Specialty → from this list:
    {specialties}

    3) Business Unit → from this list:
    {business_units}

    If the query contains a symptom or medical complaint but NO specialty,
    classify the complaint to the closest specialty.

    ALWAYS call the function `search_doctors`
    if ANY doctor/specialty/BU is detected or inferred.

    If the query is general (e.g., شكراً – ازيك – تمام؟)
    → DO NOT call any function.
    """

    client = get_gemini_client()

    response = client.generate_content(
        [instructions, query],
    )

    return response
