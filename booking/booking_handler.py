# booking/booking_handler.py

import pandas as pd
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Get the dr_rag directory (parent of booking directory)
BASE_DIR = Path(__file__).parent.parent
APPOINTMENTS_FILE = BASE_DIR / "data" / "appointments.xlsx"


def get_doctor_notes(df: pd.DataFrame, doctor_name: str) -> Optional[str]:
    """
    Get DR Notes for a specific doctor.
    Returns the notes if found, None otherwise.
    """
    if df.empty or not doctor_name:
        return None
    
    # Fuzzy match doctor name
    from rapidfuzz import fuzz
    from data_handling.data_preprocess import clean_text
    
    doctor_name_clean = clean_text(doctor_name).lower()
    
    best_match = None
    best_score = 0
    
    for idx, row in df.iterrows():
        row_name = clean_text(str(row.get("Doctor Name", ""))).lower()
        score = fuzz.partial_ratio(doctor_name_clean, row_name)
        if score > best_score:
            best_score = score
            best_match = row
    
    if best_score >= 70 and best_match is not None:
        notes = best_match.get("DR Notes", "")
        return str(notes) if notes else None
    
    return None


def create_appointments_file_if_not_exists():
    """Create appointments Excel file with headers if it doesn't exist"""
    file_path = APPOINTMENTS_FILE
    
    if not file_path.exists():
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create DataFrame with headers
        df = pd.DataFrame(columns=[
            "Appointment ID",
            "Date Created",
            "Patient Name",
            "Patient Phone",
            "Doctor Name",
            "Speciality",
            "Business Unit",
            "Appointment Date",
            "Appointment Time",
            "Status",
            "Notes",
            "DR Notes Reference"
        ])
        
        df.to_excel(str(file_path), index=False)
        print(f"✅ Created appointments file: {file_path}")


def book_appointment(doctor_name: str, patient_name: str, patient_phone: str,
                    appointment_date: str, appointment_time: str,
                    df: pd.DataFrame, speciality: str = "", 
                    business_unit: str = "", notes: str = "") -> Dict[str, Any]:
    """
    Book an appointment and save to Excel.
    Validates against DR Notes if available.
    
    Args:
        doctor_name: Name of the doctor
        patient_name: Name of the patient
        patient_phone: Phone number of the patient
        appointment_date: Date of appointment (YYYY-MM-DD)
        appointment_time: Time of appointment (HH:MM)
        df: Doctor data DataFrame
        speciality: Doctor's speciality
        business_unit: Business unit
        notes: Additional notes
    
    Returns:
        Dictionary with booking status and details
    """
    try:
        # Get doctor notes
        dr_notes = get_doctor_notes(df, doctor_name)
        
        # Validate booking against DR Notes if they exist
        validation_result = None
        if dr_notes:
            validation_result = validate_booking_against_notes(
                appointment_date, appointment_time, dr_notes
            )
            
            # If validation fails, return error with suggestions
            if not validation_result["valid"]:
                suggestions_text = ""
                if validation_result.get("suggestions"):
                    if ":" in str(validation_result["suggestions"][0]):
                        # Time suggestions
                        suggestions_text = f"\n\nالأوقات المتاحة:\n" + "\n".join([f"- {t}" for t in validation_result["suggestions"]])
                    else:
                        # Date suggestions
                        suggestions_text = f"\n\nالتواريخ المتاحة:\n" + "\n".join([f"- {d}" for d in validation_result["suggestions"]])
                
                return {
                    "success": False,
                    "error": "validation_failed",
                    "message": validation_result["message"] + suggestions_text,
                    "suggestions": validation_result.get("suggestions", []),
                    "dr_notes": dr_notes
                }
        
        # Create appointments file if it doesn't exist
        create_appointments_file_if_not_exists()
        
        # Load existing appointments
        appointments_df = pd.read_excel(str(APPOINTMENTS_FILE))
        
        # Generate appointment ID
        appointment_id = f"APT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(appointments_df) + 1}"
        
        # Create new appointment record
        new_appointment = {
            "Appointment ID": appointment_id,
            "Date Created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Patient Name": patient_name,
            "Patient Phone": patient_phone,
            "Doctor Name": doctor_name,
            "Speciality": speciality,
            "Business Unit": business_unit,
            "Appointment Date": appointment_date,
            "Appointment Time": appointment_time,
            "Status": "Confirmed",
            "Notes": notes,
            "DR Notes Reference": dr_notes if dr_notes else ""
        }
        
        # Add to DataFrame
        new_row = pd.DataFrame([new_appointment])
        appointments_df = pd.concat([appointments_df, new_row], ignore_index=True)
        
        # Save to Excel
        appointments_df.to_excel(str(APPOINTMENTS_FILE), index=False)
        
        validation_msg = ""
        if validation_result and validation_result["valid"]:
            validation_msg = f"\n✓ تم التحقق من التوافق مع ملاحظات الطبيب."
        
        return {
            "success": True,
            "appointment_id": appointment_id,
            "doctor_name": doctor_name,
            "patient_name": patient_name,
            "appointment_date": appointment_date,
            "appointment_time": appointment_time,
            "dr_notes": dr_notes,
            "message": f"تم حجز الموعد بنجاح! رقم الحجز: {appointment_id}{validation_msg}"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"حدث خطأ أثناء حجز الموعد: {str(e)}"
        }


def parse_booking_info_from_notes(dr_notes: str) -> Dict[str, Any]:
    """
    Extract booking information from DR Notes.
    Returns available dates, times, restrictions, etc.
    """
    if not dr_notes:
        return {
            "has_restrictions": False,
            "available_dates": [],
            "available_times": [],
            "restricted_dates": [],
            "restricted_times": [],
            "booking_instructions": "",
            "contact_info": ""
        }
    
    import re
    from datetime import datetime, timedelta
    
    notes = str(dr_notes)
    notes_lower = notes.lower()
    
    info = {
        "has_restrictions": False,
        "available_dates": [],
        "available_times": [],
        "restricted_dates": [],
        "restricted_times": [],
        "booking_instructions": notes,
        "contact_info": ""
    }
    
    # Extract dates (various formats)
    date_patterns = [
        r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',  # DD/MM/YYYY or DD-MM-YYYY
        r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY/MM/DD
        r'يوم\s*(\d{1,2})[/-](\d{1,2})',  # يوم DD/MM
    ]
    
    # Extract times (HH:MM format)
    time_patterns = [
        r'(\d{1,2}):(\d{2})',  # HH:MM
        r'(\d{1,2})\s*ساعة',  # HH ساعة
        r'من\s*(\d{1,2})\s*إلى\s*(\d{1,2})',  # من HH إلى HH
    ]
    
    # Look for available dates
    for pattern in date_patterns:
        matches = re.findall(pattern, notes)
        for match in matches:
            try:
                if len(match) == 3:
                    if len(match[2]) == 2:
                        year = 2000 + int(match[2])
                    else:
                        year = int(match[2])
                    month = int(match[1])
                    day = int(match[0])
                    date_str = f"{year}-{month:02d}-{day:02d}"
                    info["available_dates"].append(date_str)
                    info["has_restrictions"] = True
            except:
                pass
    
    # Look for available times
    for pattern in time_patterns:
        matches = re.findall(pattern, notes)
        for match in matches:
            try:
                if isinstance(match, tuple):
                    if len(match) == 2:
                        # Time range
                        start_hour = int(match[0])
                        end_hour = int(match[1])
                        for hour in range(start_hour, end_hour + 1):
                            info["available_times"].append(f"{hour:02d}:00")
                    else:
                        hour = int(match[0])
                        info["available_times"].append(f"{hour:02d}:00")
                else:
                    hour = int(match)
                    info["available_times"].append(f"{hour:02d}:00")
                info["has_restrictions"] = True
            except:
                pass
    
    # Look for restricted days (like "لا حجز يوم الجمعة")
    restricted_days = {
        "السبت": "Saturday",
        "الأحد": "Sunday", 
        "الاثنين": "Monday",
        "الثلاثاء": "Tuesday",
        "الأربعاء": "Wednesday",
        "الخميس": "Thursday",
        "الجمعة": "Friday"
    }
    
    for day_ar, day_en in restricted_days.items():
        if day_ar in notes or day_en.lower() in notes_lower:
            # Mark this day as restricted
            info["has_restrictions"] = True
    
    # Extract contact info
    phone_pattern = r'(\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\d{10,11})'
    phone_matches = re.findall(phone_pattern, notes)
    if phone_matches:
        info["contact_info"] = phone_matches[0]
    
    return info


def validate_booking_against_notes(requested_date: str, requested_time: str, dr_notes: str) -> Dict[str, Any]:
    """
    Validate if the requested booking matches DR Notes restrictions.
    
    Returns:
        {
            "valid": bool,
            "message": str,
            "suggestions": list
        }
    """
    if not dr_notes:
        return {
            "valid": True,
            "message": "لا توجد قيود في ملاحظات الطبيب. يمكن الحجز.",
            "suggestions": []
        }
    
    booking_info = parse_booking_info_from_notes(dr_notes)
    
    # If no restrictions, allow booking
    if not booking_info["has_restrictions"]:
        return {
            "valid": True,
            "message": "يمكن الحجز في التاريخ والوقت المطلوب.",
            "suggestions": []
        }
    
    # Check if specific dates are required
    if booking_info["available_dates"]:
        if requested_date not in booking_info["available_dates"]:
            return {
                "valid": False,
                "message": f"التاريخ المطلوب ({requested_date}) غير متاح حسب ملاحظات الطبيب.",
                "suggestions": booking_info["available_dates"][:5]  # Suggest first 5 available dates
            }
    
    # Check if specific times are required
    if booking_info["available_times"]:
        requested_hour = requested_time.split(":")[0] if ":" in requested_time else requested_time
        available_hours = [t.split(":")[0] for t in booking_info["available_times"]]
        
        if requested_hour not in available_hours:
            return {
                "valid": False,
                "message": f"الوقت المطلوب ({requested_time}) غير متاح حسب ملاحظات الطبيب.",
                "suggestions": booking_info["available_times"][:5]  # Suggest first 5 available times
            }
    
    # Check restricted days
    from datetime import datetime
    try:
        date_obj = datetime.strptime(requested_date, "%Y-%m-%d")
        day_name_ar = ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"][date_obj.weekday()]
        
        if day_name_ar in booking_info["booking_instructions"] and "لا" in booking_info["booking_instructions"]:
            # Check if this day is restricted
            restricted_days = ["السبت", "الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة"]
            for day in restricted_days:
                if day in booking_info["booking_instructions"] and "لا" in booking_info["booking_instructions"]:
                    if day == day_name_ar:
                        return {
                            "valid": False,
                            "message": f"يوم {day_name_ar} غير متاح للحجز حسب ملاحظات الطبيب.",
                            "suggestions": []
                        }
    except:
        pass
    
    return {
        "valid": True,
        "message": "التاريخ والوقت متاحان حسب ملاحظات الطبيب.",
        "suggestions": []
    }

