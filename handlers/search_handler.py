# handlers/search_handler.py

import pandas as pd
from typing import Dict, Any, List
from llm.fireworks_client import FireworksClient
from llm.llm_router import generate_final_response
from search.search_functions import execute_search
from handlers.function_parser import normalize_search_args


def handle_search(client: FireworksClient, df: pd.DataFrame, query: str,
                 args: Dict[str, Any], conversation_history: List[Dict[str, str]]) -> tuple:
    """
    Handle search_doctors function call.
    Returns (response_text, updated_history)
    """
    # Normalize search arguments
    normalized_args = normalize_search_args(args)
    
    print(f"🔍 Executing search with args: {normalized_args}")
    
    # Execute the search with user query for similarity ranking
    # Returns top 10+ results based on similarity
    search_results = execute_search(df, normalized_args, user_query=query)
    
    print(f"✅ Found {len(search_results)} results (ranked by relevance)")
    
    # Send results back to LLM for natural formatting
    if search_results:
        final_response = generate_final_response(client, query, search_results)
        
        # Update conversation history
        new_history = conversation_history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": final_response}
        ]
        
        return final_response, new_history
    else:
        error_msg = "⚠️ لم يتم العثور على أطباء مطابقين لبحثك. يرجى المحاولة بكلمات مختلفة."
        new_history = conversation_history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": error_msg}
        ]
        return error_msg, new_history

