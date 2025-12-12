# data_handling/data_loader.py

import pandas as pd
from .data_preprocess import clean_text, expand_business_units

def load_doctor_data(path="data/KSA_Doctors.xlsx"):
    df = pd.read_excel(path)

    # Basic clean
    df = df.fillna("")

    df["Doctor Name"] = df["Doctor Name Ar"].apply(clean_text)
    df["Speciality Description Arabic"] = df["Specialty: ArabicName"].apply(clean_text)
    # Business Unit Expansion
    df["BU Arabic List"] = df["Business Unit"].apply(expand_business_units)
    # Scope of Service (Arabic)
    df["Scope of Service Arabic"] = df["Scope of Service(AR)"].apply(clean_text)
    # Keep DR Notes as is (for booking)
    if "DR Notes" in df.columns:
        df["DR Notes"] = df["DR Notes"].fillna("")

    return df


def extract_context_specialties(df):
    """Return clean list of specialties without the ID"""
    specialties = df["Speciality Description Arabic"].unique().tolist()
    
    cleaned = []
    for s in specialties:
        if not s.strip():
            continue
        # خذ الجزء قبل ;
        main_speciality = s.split(";")[0].strip()
        cleaned.append(main_speciality)
    return cleaned


def extract_context_doctors(df):
    return df["Doctor Name"].unique().tolist()


def extract_context_business_units(df):
    bu_list = []

    for row in df["BU Arabic List"]:
        for bu in row:
            bu_list.append(bu)

    return list(set(bu_list))
