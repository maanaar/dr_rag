# handlers/booking_request_handler.py

from typing import Dict, Any, List, Tuple
import pandas as pd
from datetime import datetime
import re
from booking.booking_handler import book_appointment as book_appointment_func


def normalize_date_format(date_str: str) -> str:
    """
    Convert various date formats to YYYY-MM-DD format.
    Handles: DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, etc.
    """
    if not date_str:
        return ""
    
    date_str = date_str.strip()
    
    # Try various date formats
    date_formats = [
        "%d/%m/%Y",  # 17/12/2025
        "%d-%m-%Y",  # 17-12-2025
        "%Y-%m-%d",  # 2025-12-17 (already correct)
        "%d/%m/%y",  # 17/12/25
        "%d-%m-%y",  # 17-12-25
    ]
    
    for fmt in date_formats:
        try:
            date_obj = datetime.strptime(date_str, fmt)
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            continue
    
    return date_str  # Return as-is if can't parse


def normalize_time_format(time_str: str) -> str:
    """
    Convert various time formats to HH:MM format.
    Handles: 6pm, 6:00pm, 18:00, 6 PM, etc.
    """
    if not time_str:
        return ""
    
    time_str = time_str.strip().lower()
    
    # Handle am/pm format
    if 'pm' in time_str or 'am' in time_str:
        # Extract hour and optional minutes
        match = re.match(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)', time_str)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2)) if match.group(2) else 0
            period = match.group(3)
            
            # Convert to 24-hour format
            if period == 'pm' and hour != 12:
                hour += 12
            elif period == 'am' and hour == 12:
                hour = 0
            
            return f"{hour:02d}:{minute:02d}"
    
    # Handle 24-hour format (already correct)
    if re.match(r'^\d{1,2}:\d{2}$', time_str):
        parts = time_str.split(':')
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}"
    
    # Handle just hour (assume :00)
    if re.match(r'^\d{1,2}$', time_str):
        return f"{int(time_str):02d}:00"
    
    return time_str  # Return as-is if can't parse


def normalize_booking_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize booking arguments to handle various parameter name variations.
    """
    normalized = {}
    
    # Handle doctor_name
    normalized['doctor_name'] = (
        args.get('doctor_name') or 
        args.get('doctor') or 
        ""
    )
    
    # Handle patient_name
    normalized['patient_name'] = (
        args.get('patient_name') or 
        args.get('patient') or 
        ""
    )
    
    # Handle phone number (various possible keys)
    normalized['patient_phone'] = (
        args.get('patient_phone') or 
        args.get('phone_number') or 
        args.get('phone') or 
        ""
    )
    
    # Handle date (various possible keys)
    date_value = (
        args.get('appointment_date') or 
        args.get('date') or 
        ""
    )
    normalized['appointment_date'] = normalize_date_format(str(date_value)) if date_value else ""
    
    # Handle time (various possible keys)
    time_value = (
        args.get('appointment_time') or 
        args.get('time') or 
        ""
    )
    normalized['appointment_time'] = normalize_time_format(str(time_value)) if time_value else ""
    
    # Optional fields
    normalized['speciality'] = args.get('speciality') or args.get('specialty') or ""
    normalized['business_unit'] = args.get('business_unit') or ""
    normalized['notes'] = args.get('notes') or ""
    
    return normalized


def validate_booking_args(args: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate booking arguments.
    Returns (is_valid, error_message)
    """
    required_fields = {
        'doctor_name': 'اسم الطبيب',
        'patient_name': 'اسم المريض',
        'patient_phone': 'رقم الهاتف',
        'appointment_date': 'التاريخ',
        'appointment_time': 'الوقت'
    }
    
    # Check for missing fields
    missing = []
    for field, name_ar in required_fields.items():
        if not args.get(field) or str(args[field]).strip() == "":
            missing.append(name_ar)
    
    if missing:
        return False, f"⚠️ يرجى تقديم المعلومات التالية: {', '.join(missing)}"
    
    # Validate date format
    try:
        datetime.strptime(args['appointment_date'], "%Y-%m-%d")
    except ValueError:
        return False, f"⚠️ صيغة التاريخ غير صحيحة. تم استلام: {args['appointment_date']}"
    
    # Validate time format
    if not re.match(r'^\d{2}:\d{2}$', args['appointment_time']):
        return False, f"⚠️ صيغة الوقت غير صحيحة. تم استلام: {args['appointment_time']}"
    
    # Validate phone number (basic check)
    phone = args['patient_phone'].replace('-', '').replace(' ', '').replace('+', '')
    if not phone.isdigit() or len(phone) < 10:
        return False, "⚠️ رقم الهاتف غير صحيح. الرجاء إدخال رقم صحيح (10 أرقام على الأقل)"
    
    return True, ""


def handle_booking(df: pd.DataFrame, query: str, args: Dict[str, Any],
                  conversation_history: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
    """
    Handle booking appointment request.
    Returns (response_text, updated_conversation_history)
    """
    print("📅 Processing booking request...")
    print(f"   Original args: {args}")
    
    # Normalize arguments to handle various parameter names
    normalized_args = normalize_booking_args(args)
    print(f"   Normalized args: {normalized_args}")
    
    # Validate arguments
    is_valid, error_msg = validate_booking_args(normalized_args)
    
    if not is_valid:
        new_history = conversation_history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": error_msg}
        ]
        return error_msg, new_history
    
    # Try to find doctor info for additional details
    if not normalized_args['speciality'] or not normalized_args['business_unit']:
        try:
            from rapidfuzz import fuzz
            from data_handling.data_preprocess import clean_text
            
            doctor_name_clean = clean_text(normalized_args['doctor_name']).lower()
            
            best_match = None
            best_score = 0
            
            for idx, row in df.iterrows():
                row_name = clean_text(str(row.get("Doctor Name", ""))).lower()
                score = fuzz.partial_ratio(doctor_name_clean, row_name)
                if score > best_score:
                    best_score = score
                    best_match = row
            
            if best_score >= 70 and best_match is not None:
                if not normalized_args['speciality']:
                    normalized_args['speciality'] = str(best_match.get("Speciality Description Arabic", ""))
                if not normalized_args['business_unit']:
                    bus = best_match.get("BU Arabic List", [])
                    if bus and isinstance(bus, list) and len(bus) > 0:
                        normalized_args['business_unit'] = str(bus[0])
        except Exception as e:
            print(f"⚠️ Could not extract doctor details: {e}")
    
    # Book the appointment
    result = book_appointment_func(
        doctor_name=normalized_args['doctor_name'],
        patient_name=normalized_args['patient_name'],
        patient_phone=normalized_args['patient_phone'],
        appointment_date=normalized_args['appointment_date'],
        appointment_time=normalized_args['appointment_time'],
        df=df,
        speciality=normalized_args['speciality'],
        business_unit=normalized_args['business_unit'],
        notes=normalized_args['notes'] or "حجز من خلال المساعد الآلي"
    )
    
    # Format response
    if result['success']:
        response_text = f"""✅ {result['message']}

📋 تفاصيل الحجز:
- رقم الحجز: {result['appointment_id']}
- الطبيب: {result['doctor_name']}
- المريض: {result['patient_name']}
- التاريخ: {result['appointment_date']}
- الوقت: {result['appointment_time']}"""
        
        if normalized_args['speciality']:
            response_text += f"\n- التخصص: {normalized_args['speciality']}"
        
        if result.get('dr_notes'):
            response_text += f"\n\n📝 ملاحظات الطبيب:\n{result['dr_notes']}"
        
        response_text += "\n\n✓ سيتم التواصل معك قريباً لتأكيد الموعد."
    else:
        # Check if it's a validation error
        if result.get('error') == 'validation_failed':
            response_text = f"⚠️ {result['message']}"
            if result.get('suggestions'):
                response_text += "\n\n💡 يمكنك اختيار أحد الخيارات المتاحة أعلاه."
        else:
            response_text = f"❌ {result['message']}"
    
    new_history = conversation_history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": response_text}
    ]
    
    return response_text, new_history