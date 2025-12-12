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


def calculate_similarity_score(row: pd.Series, query: str, args: Dict[str, str]) -> float:
    """
    Calculate a similarity score for a doctor record based on the user query.
    Higher score = more relevant.
    """
    query_clean = clean_text(query).lower()
    total_score = 0.0
    max_possible_score = 0.0
    
    # Weight different fields by importance
    weights = {
        "doctor_name": 0.3,
        "speciality": 0.25,
        "scope_of_service": 0.3,
        "business_unit": 0.15
    }
    
    # Score doctor name
    if args.get("doctor_name"):
        doctor_name = clean_text(str(row.get("Doctor Name", ""))).lower()
        score = fuzz.partial_ratio(query_clean, doctor_name)
        total_score += score * weights["doctor_name"]
        max_possible_score += 100 * weights["doctor_name"]
    else:
        # Even if not explicitly searched, check if query mentions name
        doctor_name = clean_text(str(row.get("Doctor Name", ""))).lower()
        score = fuzz.partial_ratio(query_clean, doctor_name)
        if score > 50:  # Only count if somewhat relevant
            total_score += score * weights["doctor_name"] * 0.5
            max_possible_score += 100 * weights["doctor_name"] * 0.5
    
    # Score speciality
    if args.get("speciality"):
        speciality = clean_text(str(row.get("Speciality Description Arabic", ""))).lower()
        score = fuzz.partial_ratio(query_clean, speciality)
        total_score += score * weights["speciality"]
        max_possible_score += 100 * weights["speciality"]
    else:
        # Check if query mentions speciality
        speciality = clean_text(str(row.get("Speciality Description Arabic", ""))).lower()
        score = fuzz.partial_ratio(query_clean, speciality)
        if score > 50:
            total_score += score * weights["speciality"] * 0.5
            max_possible_score += 100 * weights["speciality"] * 0.5
    
    # Score scope of service (important for recommendations)
    scope = clean_text(str(row.get("Scope of Service Arabic", ""))).lower()
    if args.get("scope_of_service"):
        score = fuzz.partial_ratio(query_clean, scope)
        total_score += score * weights["scope_of_service"]
        max_possible_score += 100 * weights["scope_of_service"]
    else:
        # Always check scope of service for relevance
        score = fuzz.partial_ratio(query_clean, scope)
        if score > 40:  # Lower threshold for scope
            total_score += score * weights["scope_of_service"] * 0.7
            max_possible_score += 100 * weights["scope_of_service"] * 0.7
    
    # Score business unit
    bus_list = row.get("BU Arabic List", [])
    if isinstance(bus_list, list):
        bus_text = " ".join([clean_text(str(bu)).lower() for bu in bus_list])
        if args.get("business_unit"):
            score = fuzz.partial_ratio(query_clean, bus_text)
            total_score += score * weights["business_unit"]
            max_possible_score += 100 * weights["business_unit"]
        else:
            score = fuzz.partial_ratio(query_clean, bus_text)
            if score > 50:
                total_score += score * weights["business_unit"] * 0.3
                max_possible_score += 100 * weights["business_unit"] * 0.3
    
    # Normalize score to 0-100 range
    if max_possible_score > 0:
        normalized_score = (total_score / max_possible_score) * 100
    else:
        normalized_score = 0.0
    
    return normalized_score


def execute_search(df: pd.DataFrame, args: Dict[str, str], user_query: str = "", offset: int = 0, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Execute search with provided filters using fuzzy matching.
    Ranks results by similarity to user query.
    Returns list of matching doctor records sorted by relevance.
    """
    if df.empty:
        return []
    
    filtered_df = df.copy()
    
    # Apply filters sequentially (broad matching first)
    if args.get("doctor_name"):
        filtered_df = fuzzy_match_column(
            filtered_df, "Doctor Name", args["doctor_name"], threshold=70  # Lower threshold for broader results
        )
    
    if args.get("speciality") and not filtered_df.empty:
        filtered_df = fuzzy_match_column(
            filtered_df, "Speciality Description Arabic", args["speciality"], threshold=70
        )
    
    if args.get("business_unit") and not filtered_df.empty:
        filtered_df = fuzzy_match_list_column(
            filtered_df, "BU Arabic List", args["business_unit"], threshold=70
        )
    
    if args.get("scope_of_service") and not filtered_df.empty:
        filtered_df = fuzzy_match_column(
            filtered_df, "Scope of Service Arabic", args["scope_of_service"], threshold=60  # Lower for services
        )
    
    # If no filters matched, use broader search on all fields
    if filtered_df.empty and user_query:
        # Search across all relevant fields
        query_clean = clean_text(user_query).lower()
        mask = pd.Series([False] * len(df))
        
        for idx, row in df.iterrows():
            # Check name
            name_score = fuzz.partial_ratio(query_clean, clean_text(str(row.get("Doctor Name", ""))).lower())
            # Check speciality
            spec_score = fuzz.partial_ratio(query_clean, clean_text(str(row.get("Speciality Description Arabic", ""))).lower())
            # Check scope
            scope_score = fuzz.partial_ratio(query_clean, clean_text(str(row.get("Scope of Service Arabic", ""))).lower())
            
            # If any field has decent match, include it
            if max(name_score, spec_score, scope_score) >= 50:
                mask.iloc[idx] = True
        
        filtered_df = df[mask].copy()
    
    # Calculate similarity scores and rank
    if not filtered_df.empty and user_query:
        filtered_df = filtered_df.copy()
        filtered_df['_similarity_score'] = filtered_df.apply(
            lambda row: calculate_similarity_score(row, user_query, args), axis=1
        )
        # Sort by score descending
        filtered_df = filtered_df.sort_values('_similarity_score', ascending=False)
        # Remove the score column before returning
        filtered_df = filtered_df.drop(columns=['_similarity_score'])
    
    # Apply pagination
    total_results = len(filtered_df)
    start_idx = offset
    end_idx = offset + limit
    paginated_df = filtered_df.iloc[start_idx:end_idx]
    
    results = paginated_df.to_dict(orient="records")
    
    # Add metadata about pagination
    for result in results:
        result['_total_results'] = total_results
        result['_current_offset'] = offset
        result['_has_more'] = end_idx < total_results
    
    return results
