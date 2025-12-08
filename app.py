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
print(specialties)
doctor_names = extract_context_doctors(df)
business_units = extract_context_business_units(df)


def handle_user_query(query):
    response = route_llm(query, specialties, doctor_names, business_units)
    print(response)
    # 1) If the model is calling a function
    fc = response.candidates[0].content.parts[0].function_call
    if fc:
        fn_name = fc.name
        args = fc.args

        if fn_name == "search_doctors":
            return execute_search(df,args)

    # 2) Otherwise return model text
    try:
        print(response.text)
        return response.text
    except:
        return "⚠ لا يوجد نص في الرد."


# def run_search_doctors(args):
#     doctor_name = args.get("doctor_name")
#     speciality = args.get("speciality")
#     business_unit = args.get("business_unit")

#     # Call your search logic
#     return execute_search(doctor_name, speciality, business_unit)

# Example
if __name__ == "__main__":
    q = input("User: ")
    print(handle_user_query(q))
