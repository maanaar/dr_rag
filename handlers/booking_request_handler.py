# handlers/booking_request_handler.py

from typing import Dict, Any, List
import pandas as pd
from booking.booking_handler import book_appointment as book_appointment_func


def handle_booking(df: pd.DataFrame, query: str, args: Dict[str, Any],
                  conversation_history: List[Dict[str, str]]) -> tuple:
    """
    Handle book_appointment function call.
    Returns (response_text, updated_history)
    """
    print(f"📅 Processing booking request...")
    
    # Extract booking parameters
    doctor_name = args.get("doctor_name", "")
    patient_name = args.get("patient_name", "")
    patient_phone = args.get("patient_phone", "")
    appointment_date = args.get("appointment_date", "")
    appointment_time = args.get("appointment_time", "")
    speciality = args.get("speciality", "")
    business_unit = args.get("business_unit", "")
    notes = args.get("notes", "")
    
    # Validate required fields
    if not any([doctor_name, patient_name, patient_phone, appointment_date, appointment_time]):
        error_msg = "⚠️ يرجى تقديم جميع المعلومات المطلوبة للحجز: اسم الطبيب، اسم المريض، رقم الهاتف، التاريخ، والوقت."
        new_history = conversation_history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": error_msg}
        ]
        return error_msg, new_history
    
    # Execute booking
    booking_result = book_appointment_func(
        doctor_name=doctor_name,
        patient_name=patient_name,
        patient_phone=patient_phone,
        appointment_date=appointment_date,
        appointment_time=appointment_time,
        df=df,
        speciality=speciality,
        business_unit=business_unit,
        notes=notes
    )
    
    # Format response
    if booking_result["success"]:
        response_text = f"""✅ {booking_result['message']}

📋 تفاصيل الحجز:
- رقم الحجز: {booking_result['appointment_id']}
- الطبيب: {booking_result['doctor_name']}
- المريض: {booking_result['patient_name']}
- التاريخ: {booking_result['appointment_date']}
- الوقت: {booking_result['appointment_time']}"""
        
        if booking_result.get('dr_notes'):
            response_text += f"\n\n📝 ملاحظات الطبيب:\n{booking_result['dr_notes']}"
    else:
        # Check if it's a validation error
        if booking_result.get('error') == 'validation_failed':
            response_text = f"⚠️ {booking_result['message']}"
            if booking_result.get('suggestions'):
                response_text += "\n\n💡 يمكنك اختيار أحد الخيارات المتاحة أعلاه."
        else:
            response_text = f"❌ {booking_result['message']}"
    
    new_history = conversation_history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": response_text}
    ]
    
    return response_text, new_history

