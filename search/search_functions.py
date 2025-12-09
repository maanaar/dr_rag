from rapidfuzz import fuzz
import pandas as pd
from data_handling.data_preprocess import clean_text

# Threshold for fuzzy matching (0-100)
FUZZY_THRESHOLD = 80

def fuzzy_filter(df, column, value, threshold=FUZZY_THRESHOLD):
    value = clean_text(value)
    matches = []
    for idx, row in df.iterrows():
        row_value = clean_text(str(row[column]))
        if fuzz.partial_ratio(value, row_value) >= threshold:
            matches.append(row)
    return pd.DataFrame(matches)

def search_by_doctor(df, name):
    return fuzzy_filter(df, "Doctor Name", name).to_dict(orient="records")

def search_by_specialty(df, specialty):
    return fuzzy_filter(df, "Speciality Description Arabic", specialty).to_dict(orient="records")

def search_by_business_unit(df, bu_ar):
    # BU Arabic List is a list of strings
    df_matches = []
    for idx, row in df.iterrows():
        for bu in row["BU Arabic List"]:
            if fuzz.partial_ratio(clean_text(bu_ar), clean_text(bu)) >= FUZZY_THRESHOLD:
                df_matches.append(row)
                break
    return pd.DataFrame(df_matches).to_dict(orient="records")

def execute_search(df, args):
    """
    Handles function call payload.
    Applies all provided filters sequentially using fuzzy matching.
    """
    filtered_df = df.copy()

    if args.get("doctor_name"):
        filtered_df = fuzzy_filter(filtered_df, "Doctor Name", args["doctor_name"])

    if args.get("speciality"):
        filtered_df = fuzzy_filter(filtered_df, "Speciality Description Arabic", args["speciality"])

    if args.get("business_unit"):
        # BU is a list column, handle separately
        bu_matches = []
        for idx, row in filtered_df.iterrows():
            for bu in row["BU Arabic List"]:
                if fuzz.partial_ratio(clean_text(args["business_unit"]), clean_text(bu)) >= FUZZY_THRESHOLD:
                    bu_matches.append(row)
                    break
        filtered_df = pd.DataFrame(bu_matches)

    return filtered_df
