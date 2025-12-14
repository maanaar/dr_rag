# handlers/function_parser.py

import json
import re
from typing import Dict, Any, Optional, Tuple


def parse_function_call(message) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Parse function call from LLM message.
    Returns (function_name, args) or (None, None)
    """
    args = None
    function_name = None
    
    # Check for tool_calls first (proper function calling)
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        print(f"🔍 Found {len(tool_calls)} tool call(s)")
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
                print(f"🔍 Parsed {function_name} function arguments: {args}")
                
                # Validate that args are not all empty
                if args and not all(v == "" or v is None for v in args.values()):
                    return function_name, args
                else:
                    print(f"⚠️ All arguments are empty, ignoring function call")
                    return None, None
            except json.JSONDecodeError as e:
                print(f"⚠️ Error parsing function arguments: {e}")
                return None, None
    
    # If no tool_calls, check if function call is in content (fallback)
    if message.content:
        content_str = str(message.content).strip()
        if "search_doctors" in content_str or "book_appointment" in content_str:
            print(f"🔍 Function call found in content, attempting to parse...")
            print(f"   Content: {content_str[:200]}...")  # Limit content display
            
            # Try to parse the entire content as JSON first
            try:
                func_data = json.loads(content_str)
                if isinstance(func_data, dict) and "name" in func_data:
                    function_name = func_data["name"]
                    if "parameters" in func_data:
                        args = func_data["parameters"]
                        
                        # Validate that args are not all empty
                        if args and not all(v == "" or v is None for v in args.values()):
                            print(f"🔍 Extracted {function_name} arguments from JSON content: {args}")
                            return function_name, args
                        else:
                            print(f"⚠️ All arguments are empty, ignoring function call")
                            return None, None
            except json.JSONDecodeError:
                # Try to extract JSON from content using regex
                for func_name in ["search_doctors", "book_appointment"]:
                    json_match = re.search(r'\{[^{}]*"name"\s*:\s*"' + func_name + r'"[^{}]*\}', content_str)
                    if json_match:
                        try:
                            func_data = json.loads(json_match.group())
                            if "parameters" in func_data:
                                args = func_data["parameters"]
                                function_name = func_name
                                
                                # Validate that args are not all empty
                                if args and not all(v == "" or v is None for v in args.values()):
                                    print(f"🔍 Extracted {function_name} arguments from regex match: {args}")
                                    return function_name, args
                                else:
                                    print(f"⚠️ All arguments are empty, ignoring function call")
                                    return None, None
                        except json.JSONDecodeError:
                            pass
    
    return None, None


def normalize_search_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize search arguments (handle parameter name variations).
    """
    normalized_args = {}
    for key, value in args.items():
        # Skip empty strings, None, and whitespace-only strings
        if not value or (isinstance(value, str) and value.strip() == ""):
            continue
            
        if key == "specialty":
            normalized_args["speciality"] = value
        elif key == "speciality":
            normalized_args["speciality"] = value
        elif key == "scope_of_service" or key == "scope_of_services" or key == "service":
            normalized_args["scope_of_service"] = value
        else:
            normalized_args[key] = value
    
    return normalized_args