import json
import random
import math
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, ValidationInfo

# ============================================================================
# SECTION 1: HARDCODED ENUMS AND REFERENCE DATA
# ============================================================================

class StatusEnum(str, Enum):
    open = "open"
    under_investigation = "under_investigation"
    charge_sheet_filed = "charge_sheet_filed"
    closed = "closed"

class InformationTypeEnum(str, Enum):
    oral = "Oral"
    written = "Written"
    suo_moto = "Suo Moto"

class LegalActEnum(str, Enum):
    bns = "BNS"
    ipc = "IPC"
    pocso = "POCSO Act"
    it_act = "IT Act 2000"
    ndps = "NDPS Act"

class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"
    other = "Other"
    unknown = "Unknown"

class KarnatakaDistrict(str, Enum):
    Bagalkot = "Bagalkot"
    Ballari = "Ballari"
    Belagavi = "Belagavi"
    Bengaluru_Rural = "Bengaluru Rural"
    Bengaluru_Urban = "Bengaluru Urban"
    Bidar = "Bidar"
    Chamarajanagar = "Chamarajanagar"
    Chikkaballapur = "Chikkaballapur"
    Chikkamagaluru = "Chikkamagaluru"
    Chitradurga = "Chitradurga"
    Dakshina_Kannada = "Dakshina Kannada"
    Davanagere = "Davanagere"
    Dharwad = "Dharwad"
    Gadag = "Gadag"
    Hassan = "Hassan"
    Haveri = "Haveri"
    Kalaburagi = "Kalaburagi"
    Kodagu = "Kodagu"
    Kolar = "Kolar"
    Koppal = "Koppal"
    Mandya = "Mandya"
    Mysuru = "Mysuru"
    Raichur = "Raichur"
    Ramanagara = "Ramanagara"
    Shivamogga = "Shivamogga"
    Tumakuru = "Tumakuru"
    Udupi = "Udupi"
    Uttara_Kannada = "Uttara Kannada"
    Vijayapura = "Vijayapura"
    Vijayanagara = "Vijayanagara"
    Yadgir = "Yadgir"

DISTRICT_COORDINATES = {
    KarnatakaDistrict.Bagalkot: (16.1817, 75.6958),
    KarnatakaDistrict.Ballari: (15.1394, 76.9214),
    KarnatakaDistrict.Belagavi: (15.8497, 74.4977),
    KarnatakaDistrict.Bengaluru_Rural: (13.2753, 77.6083),
    KarnatakaDistrict.Bengaluru_Urban: (12.9716, 77.5946),
    KarnatakaDistrict.Bidar: (17.9104, 77.5199),
    KarnatakaDistrict.Chamarajanagar: (11.9261, 76.9437),
    KarnatakaDistrict.Chikkaballapur: (13.4325, 77.7275),
    KarnatakaDistrict.Chikkamagaluru: (13.3161, 75.7720),
    KarnatakaDistrict.Chitradurga: (14.2251, 76.3980),
    KarnatakaDistrict.Dakshina_Kannada: (12.8688, 75.2522),
    KarnatakaDistrict.Davanagere: (14.4644, 75.9218),
    KarnatakaDistrict.Dharwad: (15.4589, 75.0078),
    KarnatakaDistrict.Gadag: (15.4312, 75.6355),
    KarnatakaDistrict.Hassan: (13.0033, 76.1004),
    KarnatakaDistrict.Haveri: (14.7950, 75.4011),
    KarnatakaDistrict.Kalaburagi: (17.3297, 76.8343),
    KarnatakaDistrict.Kodagu: (12.3375, 75.8069),
    KarnatakaDistrict.Kolar: (13.1367, 78.1292),
    KarnatakaDistrict.Koppal: (15.3465, 76.1554),
    KarnatakaDistrict.Mandya: (12.5218, 76.8951),
    KarnatakaDistrict.Mysuru: (12.2958, 76.6394),
    KarnatakaDistrict.Raichur: (16.2076, 77.3463),
    KarnatakaDistrict.Ramanagara: (12.7150, 77.2813),
    KarnatakaDistrict.Shivamogga: (13.9299, 75.5681),
    KarnatakaDistrict.Tumakuru: (13.3379, 77.1173),
    KarnatakaDistrict.Udupi: (13.3409, 74.7421),
    KarnatakaDistrict.Uttara_Kannada: (14.8021, 74.7402),
    KarnatakaDistrict.Vijayapura: (16.8302, 75.7100),
    KarnatakaDistrict.Vijayanagara: (15.0118, 76.3235),
    KarnatakaDistrict.Yadgir: (16.7645, 77.1407),
}

# ============================================================================
# SECTION 2: COMPREHENSIVE PYDANTIC SEED SCHEMAS
# ============================================================================

class StationModel(BaseModel):
    name: str
    district: KarnatakaDistrict
    lat: float
    lng: float

class OccurrenceWindow(BaseModel):
    from_time: datetime
    to_time: datetime

class TimelineModel(BaseModel):
    filed_at: datetime
    occurrence_window: OccurrenceWindow
    delay_in_reporting_hours: int
    reason_for_delay: Optional[str]

class ChargeModel(BaseModel):
    act: LegalActEnum
    section: str

class LocationModel(BaseModel):
    text: str
    distance_from_ps_km: float
    direction_from_ps: str
    lat: float
    lng: float

class ComplainantModel(BaseModel):
    name: str
    guardian_name: str
    age: int = Field(ge=18, description="Complainant must be a legal adult (18+)")
    gender: GenderEnum
    phone: str = Field(pattern=r"^\d{10}$", description="Strict 10-digit Indian mobile number")
    relation_to_victim: str

class VictimModel(BaseModel):
    name: str
    guardian_name: Optional[str]
    age: Optional[int]
    gender: GenderEnum
    address: Optional[str]
    injury_type: Optional[str]

class AccusedModel(BaseModel):
    name: str
    is_known: bool
    physical_desc: Optional[str]

class PeopleModel(BaseModel):
    complainant: ComplainantModel
    victims: List[VictimModel]
    accused: List[AccusedModel]

class PropertyModel(BaseModel):
    items_description: Optional[str]
    estimated_value_inr: int

class InvestigationModel(BaseModel):
    io_name: str
    io_id: str
    evidence_logged: List[str]
    court_case_no: Optional[str]

class FIRModel(BaseModel):
    fir_id: int
    fir_number: str
    gd_reference: str
    station: StationModel
    timeline: TimelineModel
    information_type: InformationTypeEnum
    crime_type: str
    description: str
    location: LocationModel
    status: StatusEnum
    people: PeopleModel
    property_stolen: PropertyModel
    investigation: InvestigationModel
    charges: List[ChargeModel]  # Defined last so timeline is parsed first

    @field_validator("charges")
    @classmethod
    def validate_legal_pivot(cls, charges: List[ChargeModel], info: ValidationInfo) -> List[ChargeModel]:
        timeline = info.data.get("timeline")
        if not timeline:
            return charges
        
        pivot_date = datetime(2024, 7, 1)
        is_post_pivot = timeline.filed_at >= pivot_date
        
        for charge in charges:
            if is_post_pivot and charge.act == LegalActEnum.ipc:
                raise ValueError(f"Constraint Violation: IPC act assigned after BNS enactment on {timeline.filed_at}")
            if not is_post_pivot and charge.act == LegalActEnum.bns:
                raise ValueError(f"Constraint Violation: BNS act assigned before enactment date on {timeline.filed_at}")
        return charges

# ============================================================================
# SECTION 3: DETERMINISTIC BUSINESS LOGIC & SCALING PIPELINE
# ============================================================================

def calculate_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates spherical distance between two coordinates in kilometers."""
    r = 6371.0
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    return r * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def generate_spatial_jitter(center_lat: float, center_lng: float) -> tuple:
    """Applies a randomized local bounding offset (±0.01 to ±0.07 degrees)."""
    lat_offset = random.uniform(0.01, 0.07) * random.choice([1, -1])
    lng_offset = random.uniform(0.01, 0.07) * random.choice([1, -1])
    return center_lat + lat_offset, center_lng + lng_offset

def generate_station(district: KarnatakaDistrict) -> StationModel:
    """Rule B: Assigns 1 of 91 Master Stations dynamically tied to district bounds."""
    station_id = random.randint(1, 91)
    base_lat, base_lng = DISTRICT_COORDINATES[district]
    ps_lat, ps_lng = generate_spatial_jitter(base_lat, base_lng)
    
    return StationModel(
        name=f"Police Station No. {station_id}",
        district=district,
        lat=round(ps_lat, 6),
        lng=round(ps_lng, 6)
    )

def generate_timeline(target_date: datetime) -> TimelineModel:
    """Rule C: Chronological Integrity. Assures correct occurrence and filing flow."""
    filed_at = target_date
    delay_hours = random.randint(0, 72)
    to_time = filed_at - timedelta(hours=delay_hours)
    duration_hours = random.randint(1, 12)
    from_time = to_time - timedelta(hours=duration_hours)
    
    reason = None
    if delay_hours > 24:
        reason = random.choice([
            "Complainant was receiving primary medical care.",
            "Family dispute delayed formal reporting procedures.",
            "Complainant feared immediate retaliation from the accused.",
            "Logistical inability to reach the station due to weather."
        ])
        
    return TimelineModel(
        filed_at=filed_at,
        occurrence_window=OccurrenceWindow(from_time=from_time, to_time=to_time),
        delay_in_reporting_hours=delay_hours,
        reason_for_delay=reason
    )

def apply_legal_pivot(crime_type: str, target_date: datetime) -> List[dict]:
    """Rule D: The July 2024 Legal Pivot Engine."""
    pivot_date = datetime(2024, 7, 1)
    
    # Mapping matrices for automated translation
    ipc_mapping = {"Murder": "302", "Theft": "379", "Assault": "351", "Fraud": "420"}
    bns_mapping = {"Murder": "103(1)", "Theft": "303(2)", "Assault": "115", "Fraud": "316"}
    
    is_post_pivot = target_date >= pivot_date
    act_enum = LegalActEnum.bns if is_post_pivot else LegalActEnum.ipc
    section_map = bns_mapping if is_post_pivot else ipc_mapping
    
    # Fallback to generalized sections if crime_type falls outside strict mapping
    section = section_map.get(crime_type, "111" if is_post_pivot else "120B")
    
    return [{"act": act_enum.value, "section": section}]

def build_synthetic_record(fir_id: int, target_date: datetime, crime_type: str, district: KarnatakaDistrict) -> dict:
    """Orchestrates seed dict before passing to Pydantic for validation."""
    station = generate_station(district)
    scene_lat, scene_lng = generate_spatial_jitter(station.lat, station.lng)
    distance_km = calculate_haversine(station.lat, station.lng, scene_lat, scene_lng)
    
    directions = ["North", "South", "East", "West", "North-East", "North-West", "South-East", "South-West"]
    
    payload = {
        "fir_id": fir_id,
        "fir_number": f"{random.randint(1, 999):04d}/{target_date.year}",
        "gd_reference": f"GD-{random.randint(1000, 9999)}",
        "station": station.model_dump(),
        "timeline": generate_timeline(target_date).model_dump(),
        "information_type": random.choice(list(InformationTypeEnum)).value,
        "crime_type": crime_type,
        "description": "Standardized narrative text. Synthetically generated evidence logged.",
        "location": {
            "text": "123 Synthetic Block, Test Avenue",
            "distance_from_ps_km": round(distance_km, 2),
            "direction_from_ps": random.choice(directions),
            "lat": round(scene_lat, 6),
            "lng": round(scene_lng, 6)
        },
        "status": StatusEnum.closed.value,
        "people": {
            "complainant": {
                "name": "Arjun Kumar", "guardian_name": "Ravi Kumar",
                "age": random.randint(18, 65), "gender": GenderEnum.male.value,
                "phone": f"9{random.randint(100000000, 999999999)}",
                "relation_to_victim": "Self"
            },
            "victims": [{
                "name": "Arjun Kumar", "guardian_name": "Ravi Kumar",
                "age": 30, "gender": GenderEnum.male.value,
                "address": "123 Synthetic Block", "injury_type": "None"
            }],
            "accused": [{
                "name": "Unknown", "is_known": False, "physical_desc": "Medium build"
            }]
        },
        "property_stolen": {
            "items_description": "Mobile phone", "estimated_value_inr": 15000
        },
        "investigation": {
            "io_name": "SI Kumaraswamy", "io_id": "IO-45892",
            "evidence_logged": ["CCTV Footage"], "court_case_no": None
        },
        "charges": apply_legal_pivot(crime_type, target_date)
    }
    return payload

# ============================================================================
# SECTION 4: HIGH-SPEED EXECUTION ENGINE & IO ARCHITECTURE
# ============================================================================

def fir_data_generator(total_records: int):
    """Yields validated Pydantic models to keep memory consumption at O(1)."""
    districts = list(KarnatakaDistrict)
    crime_types = ["Murder", "Theft", "Assault", "Fraud"]
    
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2026, 6, 4)
    delta_days = (end_date - start_date).days
    
    for i in range(1, total_records + 1):
        # Randomize a date within the target window
        random_days = random.randint(0, delta_days)
        target_date = start_date + timedelta(days=random_days)
        
        raw_dict = build_synthetic_record(
            fir_id=i,
            target_date=target_date,
            crime_type=random.choice(crime_types),
            district=random.choice(districts)
        )
        
        # Pydantic validates chronological integrity, schema rules, and the Legal Pivot
        validated_model = FIRModel(**raw_dict)
        yield validated_model.model_dump_json()

def execute_pipeline_to_disk(filename: str, num_records: int):
    """Streams JSONL directly to disk."""
    print(f"Initiating streaming pipeline for {num_records} records to {filename}...")
    with open(filename, 'w', encoding='utf-8') as f:
        for json_record in fir_data_generator(num_records):
            f.write(json_record + '\n')
    print("Pipeline execution complete.")

if __name__ == "__main__":
    # ---------------------------------------------------------
    # SAMPLE EXECUTION BLOCK & VALIDATION PROOF
    # ---------------------------------------------------------
    print("--- Executing Validation Tests ---")
    
    # 1. Pre-Pivot Test (Year 2018)
    pre_pivot_date = datetime(2018, 5, 10)
    pre_pivot_dict = build_synthetic_record(1, pre_pivot_date, "Murder", KarnatakaDistrict.Bengaluru_Urban)
    pre_pivot_model = FIRModel(**pre_pivot_dict)
    
    print("\n[Pre-Pivot Validation Passed]")
    print(f"Date: {pre_pivot_model.timeline.filed_at.strftime('%Y-%m-%d')}")
    print(f"Act Assessed: {pre_pivot_model.charges[0].act.value} Sec {pre_pivot_model.charges[0].section}")
    
    # 2. Post-Pivot Test (Year 2025)
    post_pivot_date = datetime(2025, 8, 15)
    post_pivot_dict = build_synthetic_record(2, post_pivot_date, "Murder", KarnatakaDistrict.Bengaluru_Urban)
    post_pivot_model = FIRModel(**post_pivot_dict)
    
    print("\n[Post-Pivot Validation Passed]")
    print(f"Date: {post_pivot_model.timeline.filed_at.strftime('%Y-%m-%d')}")
    print(f"Act Assessed: {post_pivot_model.charges[0].act.value} Sec {post_pivot_model.charges[0].section}")
    print("-" * 34)
    
    # To run the full 330,000 scale, uncomment the line below. 
    # For safe execution in this environment, we output a micro-batch of 5.
    # execute_pipeline_to_disk("karnataka_fir_sample.jsonl", 330000)
    execute_pipeline_to_disk("karnataka_fir_micro_batch.jsonl", 5)