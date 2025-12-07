# data_handling/business_unit_map.py

BU_MAPPING = {
    "AKW": "أنسنية لصحة الطفل",
    "ALW": "أنسنية لصحة المرأة",
    "AMH": "مستشفى أنسنية المعادي",
    "ASH": "مستشفى أنسنية الشلالات",
    "FWZ": "عيادات الأمير فواز",
    "HJH": "مستشفى حي الجامعة",
    "KSA-HC": "أنسنية للرعاية المنزلية",
    "LCH": "مركز أنسنية لطب الأسنان",
    "MKR": "مركز حفا لطب الأسنان",
    "SMH": "مستشفى أنسنية سموحة",
    "SNB": "أنسنية سنابل"
}

def map_business_unit(code):
    return BU_MAPPING.get(code.strip(), code.strip())
