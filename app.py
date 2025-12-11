from data_handling.data_loader import (
    load_doctor_data,
    extract_context_specialties,
    extract_context_doctors,
    extract_context_business_units,
)
from llm.llm_router import route_llm
from search.search_functions import execute_search

# Load and prepare data
df = load_doctor_data()
specialties = extract_context_specialties(df)
doctor_names = extract_context_doctors(df)
business_units = extract_context_business_units(df)


def handle_user_query(query):
    response = route_llm(query, specialties, doctor_names, business_units)

    # Check if candidates exist
    if not response.candidates:
        return "⚠ لا يوجد رد من النموذج."

    candidate = response.candidates[0]
    print("DEBUG - LLM Response Candidate:", candidate)
    # Check if content exists
    if candidate.content and candidate.content.parts:
        part = candidate.content.parts[0]
        print("DEBUG - LLM Response Part:", part)
        # Check if the model called a function
        fc = getattr(part, "function_call", None)
        if fc:
            fn_name = fc.name
            args = fc.args
            if fn_name == "search_doctors":
                print("DEBUG - Executing search_doctors with args:", args)
                print(execute_search(df, args) ,'resss')
                return execute_search(df, args)

        # Otherwise, return text
        if getattr(part, "text", None):
            return part.text

    return "⚠ لا يوجد نص في الرد."



if __name__ == "__main__":
    while True:
        q = input("User: ")
        if q.lower() in ["exit", "quit"]:
            break
        print("Bot:", handle_user_query(q))
