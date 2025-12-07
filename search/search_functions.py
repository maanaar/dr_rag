# search/search_functions.py

def search_by_doctor(df, name):
    results = df[df["Doctor Name"].str.contains(name, case=False, na=False)]
    return results.to_dict(orient="records")


def search_by_specialty(df, specialty):
    results = df[df["Speciality Description Arabic"] == specialty]
    return results.to_dict(orient="records")


def search_by_business_unit(df, bu_ar):
    """Match BU Arabic name inside list -> df['BU Arabic List']"""
    results = df[df["BU Arabic List"].apply(lambda x: bu_ar in x)]
    return results.to_dict(orient="records")


def execute_search(df, args):
    """Handles the function call payload"""

    if args.get("doctor_name"):
        return search_by_doctor(df, args["doctor_name"])

    if args.get("specialty"):
        return search_by_specialty(df, args["specialty"])

    if args.get("business_unit"):
        return search_by_business_unit(df, args["business_unit"])

    return []
