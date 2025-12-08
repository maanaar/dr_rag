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

    return df


def extract_context_specialties(df):
    """Return unique list of specialties for LLM"""
    specialties = df["Speciality Description Arabic"].unique().tolist()
    return [s for s in specialties if s.strip()]


def extract_context_doctors(df):
    return df["Doctor Name"].unique().tolist()


def extract_context_business_units(df):
    bu_list = []

    for row in df["BU Arabic List"]:
        for bu in row:
            bu_list.append(bu)

    return list(set(bu_list))
