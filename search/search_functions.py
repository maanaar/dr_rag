# ============================================================================
# FILE: search/search_functions.py
# ============================================================================

import pandas as pd
from typing import Dict, Any, List
from rapidfuzz import fuzz
from data_handling.data_preprocess import clean_text

FUZZY_THRESHOLD = 80


def fuzzy_match_column(df: pd.DataFrame, column: str, value: str, 
                       threshold: int = FUZZY_THRESHOLD) -> pd.DataFrame:
    """
    Efficiently filter dataframe by fuzzy matching on a column.
    Returns rows where the fuzzy match score >= threshold.
    """
    if df.empty or column not in df.columns:
        return pd.DataFrame()
    
    value_clean = clean_text(value)
    if not value_clean:
        return pd.DataFrame()
    
    # Vectorized approach
    mask = df[column].apply(
        lambda x: fuzz.partial_ratio(value_clean, clean_text(str(x))) >= threshold
    )
    
    return df[mask].copy()


def fuzzy_match_list_column(df: pd.DataFrame, column: str, value: str,
                           threshold: int = FUZZY_THRESHOLD) -> pd.DataFrame:
    """
    Filter dataframe where column contains lists, fuzzy matching against list items.
    """
    if df.empty or column not in df.columns:
        return pd.DataFrame()
    
    value_clean = clean_text(value)
    if not value_clean:
        return pd.DataFrame()
    
    def matches_any_item(items):
        if not isinstance(items, list):
            return False
        return any(
            fuzz.partial_ratio(value_clean, clean_text(item)) >= threshold
            for item in items
        )
    
    mask = df[column].apply(matches_any_item)
    return df[mask].copy()


def execute_search(df: pd.DataFrame, args: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Execute search with provided filters using fuzzy matching.
    Returns list of matching doctor records.
    """
    if df.empty:
        return []
    
    filtered_df = df.copy()
    
    # Apply filters sequentially
    if args.get("doctor_name"):
        filtered_df = fuzzy_match_column(
            filtered_df, "Doctor Name", args["doctor_name"]
        )
    
    if args.get("speciality") and not filtered_df.empty:
        filtered_df = fuzzy_match_column(
            filtered_df, "Speciality Description Arabic", args["speciality"]
        )
    
    if args.get("business_unit") and not filtered_df.empty:
        filtered_df = fuzzy_match_list_column(
            filtered_df, "BU Arabic List", args["business_unit"]
        )
    
    return filtered_df.to_dict(orient="records")
