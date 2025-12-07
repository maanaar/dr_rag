# app.py

from data_handling.data_loader import (
    load_doctor_data,
    extract_context_specialties,
    extract_context_doctors,
    extract_context_business_units,
)
from data_handling.data_preprocess import normalize_query
from llm.llm_router import route_llm
from search.search_functions import execute_search

# Load and prepare data
df = load_doctor_data()
specialties = extract_context_specialties(df)
doctor_names = extract_context_doctors(df)
business_units = extract_context_business_units(df)


def handle_user_query(query):
    query_norm = normalize_query(query)

    response = route_llm(query_norm, specialties, doctor_names, business_units)

    # No function call → LLM normal reply
    if not response.candidates[0].content.parts:
        return response.text

    part = response.candidates[0].content.parts[0]

    if part.function_call:
        fn_name = part.function_call.name
        args = part.function_call.args

        results = execute_search(df, args)
        return results

    return response.text


# Example
if __name__ == "__main__":
    q = input("User: ")
    print(handle_user_query(q))
