"""
Karnataka FIR Hybrid Synthesis Generator
=========================================
Principal Data Architect Blueprint — Production Grade
Spans: 2010-01-01 to 2026-06-04 | Target: ~330,000 records (10% sample)

Two-Phase Architecture:
  Phase 1 (Seed)     — Local Ollama LLM generates culturally rich Karnataka FIR narrative
                       seeds in JSON batches of 5–10 per invoke, loop repeated 500 times.
  Phase 2 (Augment)  — Pure-Python augmentation engine multiplies each seed across
                       variant districts / dates / identities to reach the target count.

Modes (--mode flag):
  demo     — Run pre/post pivot validation proof only (no LLM needed)
  seed     — Run Ollama LLM loop → writes llm_seeds.jsonl
  augment  — Read llm_seeds.jsonl → augment to target count → writes final JSONL
  full     — seed + augment in sequence
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import re
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from langchain_ollama import ChatOllama
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("KA-FIR")


# =============================================================================
# SECTION 1: HARDCODED ENUMS AND REFERENCE DATA
# =============================================================================

class StatusEnum(str, Enum):
    open = "open"
    under_investigation = "under_investigation"
    charge_sheet_filed = "charge_sheet_filed"
    closed = "closed"


class InformationTypeEnum(str, Enum):
    Oral = "Oral"
    Written = "Written"
    Suo_Moto = "Suo Moto"


class LegalActEnum(str, Enum):
    BNS = "BNS"
    IPC = "IPC"
    POCSO_Act = "POCSO Act"
    IT_Act_2000 = "IT Act 2000"
    NDPS_Act = "NDPS Act"


class GenderEnum(str, Enum):
    Male = "Male"
    Female = "Female"
    Other = "Other"
    Unknown = "Unknown"


class KarnatakaDistrict(str, Enum):
    BAGALKOT = "Bagalkot"
    BALLARI = "Ballari"
    BELAGAVI = "Belagavi"
    BENGALURU_RURAL = "Bengaluru Rural"
    BENGALURU_URBAN = "Bengaluru Urban"
    BIDAR = "Bidar"
    CHAMARAJANAGAR = "Chamarajanagar"
    CHIKKABALLAPUR = "Chikkaballapur"
    CHIKKAMAGALURU = "Chikkamagaluru"
    CHITRADURGA = "Chitradurga"
    DAKSHINA_KANNADA = "Dakshina Kannada"
    DAVANAGERE = "Davanagere"
    DHARWAD = "Dharwad"
    GADAG = "Gadag"
    HASSAN = "Hassan"
    HAVERI = "Haveri"
    KALABURAGI = "Kalaburagi"
    KODAGU = "Kodagu"
    KOLAR = "Kolar"
    KOPPAL = "Koppal"
    MANDYA = "Mandya"
    MYSURU = "Mysuru"
    RAICHUR = "Raichur"
    RAMANAGARA = "Ramanagara"
    SHIVAMOGGA = "Shivamogga"
    TUMAKURU = "Tumakuru"
    UDUPI = "Udupi"
    UTTARA_KANNADA = "Uttara Kannada"
    VIJAYAPURA = "Vijayapura"
    VIJAYANAGARA = "Vijayanagara"
    YADGIR = "Yadgir"


# Real geographical centroids for all 31 Karnataka districts
DISTRICT_COORDINATES: dict[KarnatakaDistrict, tuple[float, float]] = {
    KarnatakaDistrict.BAGALKOT:          (16.1817, 75.6958),
    KarnatakaDistrict.BALLARI:           (15.1394, 76.9214),
    KarnatakaDistrict.BELAGAVI:          (15.8497, 74.4977),
    KarnatakaDistrict.BENGALURU_RURAL:   (13.2753, 77.6083),
    KarnatakaDistrict.BENGALURU_URBAN:   (12.9716, 77.5946),
    KarnatakaDistrict.BIDAR:             (17.9104, 77.5199),
    KarnatakaDistrict.CHAMARAJANAGAR:    (11.9261, 76.9437),
    KarnatakaDistrict.CHIKKABALLAPUR:    (13.4325, 77.7275),
    KarnatakaDistrict.CHIKKAMAGALURU:    (13.3161, 75.7720),
    KarnatakaDistrict.CHITRADURGA:       (14.2251, 76.3980),
    KarnatakaDistrict.DAKSHINA_KANNADA:  (12.8688, 75.2522),
    KarnatakaDistrict.DAVANAGERE:        (14.4644, 75.9218),
    KarnatakaDistrict.DHARWAD:           (15.4589, 75.0078),
    KarnatakaDistrict.GADAG:             (15.4312, 75.6355),
    KarnatakaDistrict.HASSAN:            (13.0033, 76.1004),
    KarnatakaDistrict.HAVERI:            (14.7950, 75.4011),
    KarnatakaDistrict.KALABURAGI:        (17.3297, 76.8343),
    KarnatakaDistrict.KODAGU:            (12.3375, 75.8069),
    KarnatakaDistrict.KOLAR:             (13.1367, 78.1292),
    KarnatakaDistrict.KOPPAL:            (15.3465, 76.1554),
    KarnatakaDistrict.MANDYA:            (12.5218, 76.8951),
    KarnatakaDistrict.MYSURU:            (12.2958, 76.6394),
    KarnatakaDistrict.RAICHUR:           (16.2076, 77.3463),
    KarnatakaDistrict.RAMANAGARA:        (12.7150, 77.2813),
    KarnatakaDistrict.SHIVAMOGGA:        (13.9299, 75.5681),
    KarnatakaDistrict.TUMAKURU:          (13.3379, 77.1173),
    KarnatakaDistrict.UDUPI:             (13.3409, 74.7421),
    KarnatakaDistrict.UTTARA_KANNADA:    (14.8021, 74.7402),
    KarnatakaDistrict.VIJAYAPURA:        (16.8302, 75.7100),
    KarnatakaDistrict.VIJAYANAGARA:      (15.0118, 76.3235),
    KarnatakaDistrict.YADGIR:            (16.7645, 77.1407),
}

# 91 master station slots — indices 1..91 cycled by (fir_id - 1) % 91
MASTER_POLICE_STATION_INDEX: list[int] = list(range(1, 92))

# The legal pivot date: IPC before this, BNS on/after this
PIVOT_DATE = datetime(2024, 7, 1)
DATASET_START = datetime(2010, 1, 1)
DATASET_END = datetime(2026, 6, 4, 23, 59, 59)

# ---------------------------------------------------------------------------
# Karnataka-specific name pools (Kannada/Karnataka culture)
# ---------------------------------------------------------------------------

MALE_FIRST_NAMES: list[str] = [
    "Manjunatha", "Venkataramaiah", "Siddaramaiah", "Basavaraj", "Shivarudraiah",
    "Kumaraswamy", "Nagarajappa", "Shivakumaraiah", "Ramakrishnaiah", "Devaraj",
    "Govindaraju", "Chandrashekhar", "Lingappa", "Hanumanthappa", "Thippeswamy",
    "Veeranna", "Siddappa", "Munirajappa", "Karibasappa", "Puttaswamy",
    "Anand", "Suresh", "Mahesh", "Prakash", "Ganesh", "Naresh", "Dinesh",
    "Ravi", "Kiran", "Arun", "Vijay", "Rajesh", "Santosh", "Harish",
    "Nataraj", "Lokesh", "Yogesh", "Rakesh", "Girish", "Umesh",
    "Pradeep", "Sunil", "Sathish", "Naveen", "Deepak", "Manoj",
    "Ashok", "Ramesh", "Shiva", "Vinay", "Roshan", "Preetham",
    "Bharath", "Mohan", "Chandan", "Raju", "Sathyanarayana", "Krishnamurthy",
]

FEMALE_FIRST_NAMES: list[str] = [
    "Sowmya", "Chandrakala", "Renukamba", "Kavitha", "Shanthamma",
    "Meenakshi", "Savithri", "Girija", "Anasuya", "Kamakshi",
    "Sumathi", "Lakshmidevamma", "Saroja", "Parvathamma", "Thayamma",
    "Nirmala", "Usha", "Vijayalakshmi", "Saritha", "Geetha",
    "Ananya", "Pooja", "Priya", "Divya", "Rekha", "Deepa",
    "Shilpa", "Nandini", "Pallavi", "Sneha", "Sahana", "Varsha",
    "Bhavana", "Radha", "Madhuri", "Asha", "Yamini", "Veena",
    "Sushma", "Poornima", "Shwetha", "Mamatha", "Pushpa", "Kamala",
    "Hema", "Padma", "Jayalakshmi", "Sharadha", "Nagaveni", "Ambika",
]

SURNAMES: list[str] = [
    "Gowda", "Naik", "Shetty", "Hegde", "Rao", "Nayak", "Patil",
    "Desai", "Kulkarni", "Joshi", "Bhat", "Pai", "Kamat", "Prabhu",
    "Shenoy", "Alva", "D'Souza", "Fernandes", "Bangera", "Salian",
    "Reddy", "Naidu", "Iyer", "Iyengar", "Pillai", "Menon", "Nair",
    "Venkataswamy", "Krishnappa", "Thimmarayappa", "Lingayat", "Vokkaliga",
    "Mudgal", "Hosamani", "Kambali", "Lamani", "Banjara", "Kuruba",
    "Lingappa", "Basappa", "Swamy", "Murthy", "Sharma", "Verma",
    "Singh", "Khan", "Shaikh", "Patel", "Kannada", "Kodava",
    "Wodeya", "Talawar", "Arakalagudu", "Holalkere", "Chikkodi",
]

# Kannada honorific guardian prefixes used in police records
GUARDIAN_PREFIXES: list[str] = ["S/O", "D/O", "W/O", "H/O", "S/O Late"]

# Police officer rank pool (Karnataka Police ranks)
OFFICER_RANKS: list[str] = ["PSI", "SI", "PI", "HC", "ASI", "ACP", "DSP"]

OFFICER_NAMES: list[str] = [
    "Nagarajappa", "Shivakumaraiah", "Kumaraswamy", "Hanumanthappa", "Basavaraj",
    "Venkataramaiah", "Thippeswamy", "Ramakrishnaiah", "Govindaraju", "Chandrashekhar",
    "Manjunatha", "Siddaramaiah", "Lingappa", "Siddappa", "Puttaswamy",
    "Srinivas", "Mahesh", "Pradeep", "Ravi", "Suresh",
    "Kiran", "Girish", "Nataraj", "Lokesh", "Dinesh",
    "Anand", "Sathyanarayana", "Krishnamurthy", "Karibasappa", "Munirajappa",
]

# ---------------------------------------------------------------------------
# Karnataka locality / hobli / ward names by district (15 per district)
# ---------------------------------------------------------------------------

DISTRICT_LOCALITY_MAP: dict[str, list[str]] = {
    "Bagalkot": [
        "Jamkhandi", "Mudhol", "Badami", "Bilgi", "Hungund",
        "Ilkal", "Kerur", "Guledagudda", "Lokapur", "Rabkavi Banhatti",
        "Mahalingpur", "Teradal", "Ainapur", "Navalagi", "Amingad",
    ],
    "Ballari": [
        "Hospet", "Sandur", "Siruguppa", "Kudligi", "Hagaribommanahalli",
        "Hadagali", "Kampli", "Tekkalakota", "Harpanahalli", "Kottur",
        "Bellary Camp", "Hirekerur", "Kurugodu", "Ramghatta Nagar", "Toranagallu",
    ],
    "Belagavi": [
        "Hubli Road", "Gokak", "Bailhongal", "Chikkodi", "Raibag",
        "Mudalgi", "Nippani", "Athani", "Kagawad", "Savadatti",
        "Parasgad", "Ramadurga", "Sadalga", "Shirhatti", "Examba",
    ],
    "Bengaluru Rural": [
        "Nelamangala", "Doddaballapur", "Devanahalli", "Hoskote",
        "Vijayapura", "Magadi", "Ramanagar", "Channapatna",
        "Kanakapura", "Anekal", "Bidadi", "Harohalli", "Avathi", "Kasaba", "Thubagere",
    ],
    "Bengaluru Urban": [
        "Rajajinagar", "Malleshwaram", "Jayanagar", "Basavanagudi",
        "Koramangala", "HSR Layout", "Electronic City", "Whitefield",
        "Yelahanka", "Hebbal", "RT Nagar", "KR Puram", "Bannerghatta Road",
        "BTM Layout", "Shivajinagar",
    ],
    "Bidar": [
        "Basavakalyan", "Humnabad", "Aurad", "Chincholi", "Bhalki",
        "Kamalnagar", "Udgir Road", "Hudgi", "Ladha", "Rajeshwar",
        "Janwada", "Chitguppa", "Ranjol", "Hallikhed", "Bidar Cantonment",
    ],
    "Chamarajanagar": [
        "Kollegal", "Gundlupet", "Yelandur", "Hanur", "Ponnachi",
        "Sathyamangalam Road", "T Narasipur", "Talabetta", "Ramapura", "Begur",
        "Vedanthangal", "Honganur", "Kowdalli", "Galipura", "Dodda Gajanur",
    ],
    "Chikkaballapur": [
        "Gudibande", "Gauribidanur", "Bagepalli", "Sidlaghatta", "Chintamani",
        "Mandikal", "Nandi Hills Area", "Peresandra", "Chelur", "Kasaba",
        "Kaiwara", "Harappanahalli", "Tirumani", "Kundana", "Lakkur",
    ],
    "Chikkamagaluru": [
        "Kadur", "Tarikere", "Mudigere", "Sringeri", "Koppa",
        "Narasimharajapura", "Thirthahalli", "Kalasa", "Aldur", "Kottigehara",
        "Belur", "Birur", "Javagal", "Mallandur", "Ajjampura",
    ],
    "Chitradurga": [
        "Hiriyur", "Holalkere", "Hosadurga", "Molakalmuru", "Challakere",
        "Turuvanur", "Kashipura", "Dodderi", "Ramabhimana", "Kasaba",
        "Madhugiri", "Parashurampura", "Uchangidurga", "Vani Vilas Sagara Area", "Sirigere",
    ],
    "Dakshina Kannada": [
        "Mangaluru", "Puttur", "Sullia", "Bantwal", "Belthangady",
        "Moodbidri", "Ullal", "Mulki", "Karkala", "Buntwal",
        "Vitla", "Kadaba", "Uppinangady", "Mudipu", "Dharmasthala Area",
    ],
    "Davanagere": [
        "Harihar", "Channagiri", "Jagalur", "Nyamati", "Honnali",
        "Santhebennur", "Anagodu", "Harihara Industrial Area", "Kasaba", "Kondajji",
        "Malladihalli", "Guttur", "Shivani", "Alur", "Nagenahalli",
    ],
    "Dharwad": [
        "Hubli", "Gadag Road", "Kundgol", "Kalghatgi", "Navalgund",
        "Annigeri", "Dharwad City", "Hirekerur", "Kalgatagi", "Shirol",
        "Amminabhavi", "Tarihal", "Vidyanagar", "Deshpande Nagar", "Gokul Road",
    ],
    "Gadag": [
        "Mundargi", "Shirahatti", "Nargund", "Ron", "Betgeri",
        "Lakshmeshwar", "Dambal", "Hole Alur", "Gadag City", "Haveri Road Area",
        "Kadadi", "Soratur", "Hotgi", "Narendra", "Adavi",
    ],
    "Hassan": [
        "Arsikere", "Channarayapatna", "Belur", "Sakleshpur", "Alur",
        "Hole Narsipur", "Arkalgud", "Manjarabad", "Dudda", "Gorur",
        "Arabithittu", "Kattaya", "Nuggehalli", "Shravanabelagola", "Doodda Gaddavalli",
    ],
    "Haveri": [
        "Shiggaon", "Byadagi", "Savanur", "Ranebennur", "Hanagal",
        "Guttal", "Hirekerur", "Motebennur", "Bankapur", "Kuknur",
        "Tirlapura", "Masur", "Devihosur", "Rattihalli", "Karajagi",
    ],
    "Kalaburagi": [
        "Afzalpur", "Aland", "Chittapur", "Jevargi", "Sedam",
        "Yadgir Road", "Chitguppa", "Wadi", "Murki", "Kembhavi",
        "Shahapur", "Sulepet", "Sannati", "Gobur", "Chincholi",
    ],
    "Kodagu": [
        "Madikeri", "Virajpet", "Somwarpet", "Kushalnagar", "Ponnampet",
        "Ammathi", "Gonikoppal", "Siddapur", "Shanivarasanthe", "Napoklu",
        "Balele", "Bhagamandala", "Thithimathi", "Aiyangeri", "Murnad",
    ],
    "Kolar": [
        "Kolar Gold Fields", "Chintamani", "Srinivaspur", "Mulbagal", "Bangarpet",
        "Malur", "Tayalur", "Bethamangala", "Nangali", "Kasaba",
        "Oorgaum", "Robertsonpet", "Champion Reef", "Dodderi", "Nandi Cross",
    ],
    "Koppal": [
        "Gangavathi", "Yelbarga", "Kushtagi", "Kanakagiri", "Munirabad",
        "Itagi", "Kuknur", "Karatagi", "Irkal Road", "Hosabande",
        "Ginigera", "Talakal", "Hirehalli", "Kampli Road", "Bevur",
    ],
    "Mandya": [
        "Srirangapatna", "Maddur", "Malavalli", "Nagamangala", "Krishnarajapete",
        "Pandavapura", "Melukote", "Shivapura", "Bellur", "Keragodu",
        "Hosaholalu", "Kasaba", "Seehalli", "Gavadagere", "Doddamavalli",
    ],
    "Mysuru": [
        "Mysuru City", "Nanjangud", "T Narasipur", "Heggadadevankote", "K R Nagar",
        "Periyapatna", "Hunsur", "Piriyapatna", "Jayapura", "Bilikere",
        "Bannur", "Saligrama", "Saragur", "Krishnarajanagara", "Rammanahalli",
    ],
    "Raichur": [
        "Lingasugur", "Devadurga", "Manvi", "Mudgal", "Sindhanur",
        "Maski", "Yelburga", "Lingsugur", "Kavital", "Raichur City",
        "Saidapur", "Kallur", "Jalahalli", "Tenginhal", "Turvihal",
    ],
    "Ramanagara": [
        "Channapatna", "Magadi", "Kanakapura", "Ramanagara City", "Bidadi",
        "Harohalli", "Sathanur", "Muthsandra", "Shivanahalli", "Kamatha",
        "Doddaballapur Road", "Sathanur Colony", "Jadigenahalli", "Mayaganahalli", "Anekal Road",
    ],
    "Shivamogga": [
        "Sagar", "Sorab", "Shikaripura", "Thirthahalli", "Hosanagara",
        "Bhadravathi", "Shimoga City", "Jog Falls Area", "Keladi", "Ayanur",
        "Hiriyadka", "Anandapura", "Gundagallu", "Mandagadde", "Mattur",
    ],
    "Tumakuru": [
        "Tiptur", "Gubbi", "Madhugiri", "Sira", "Pavagada",
        "Koratagere", "Kunigal", "Tumakuru City", "Chikkanayakanahalli", "Kyatsandra",
        "Turuvekere", "Yeldur", "Banavara", "Kora", "Kasaba",
    ],
    "Udupi": [
        "Manipal", "Udupi City", "Karkala", "Kundapur", "Brahmavar",
        "Byndoor", "Hebri", "Kaup", "Padubidri", "Shirva",
        "Malpe", "Katapadi", "Innanje", "Perdur", "Ambalpady",
    ],
    "Uttara Kannada": [
        "Karwar", "Sirsi", "Kumta", "Honavar", "Ankola",
        "Yellapur", "Mundgod", "Haliyal", "Joida", "Siddapur",
        "Dandeli", "Suppa", "Gokarna", "Bhatkal", "Ulavi",
    ],
    "Vijayapura": [
        "Bijapur City", "Sindagi", "Indi", "Muddebihal", "Basavana Bagewadi",
        "Talikota", "Tikota", "Nidagundi", "Managundi", "Kolhar",
        "Devara Hipparagi", "Kanakagiri Road", "Atharga", "Hanchinal", "Ingleshwar",
    ],
    "Vijayanagara": [
        "Hospet", "Sandur", "Harapanahalli", "Hagari Bommanahalli", "Kottur",
        "Kampli", "Yelburga", "Siriguppa Road", "Tekkalakote", "Huligi",
        "Hampi Area", "Kamalapura", "Daroji", "Kotturu", "Ananthasagara",
    ],
    "Yadgir": [
        "Shorapur", "Gurmatkal", "Shahapur", "Wadagera", "Hunsagi",
        "Gogi", "Saidapur", "Kembhavi", "Deodurga Road", "Malkheda",
        "Hatti", "Kotgal", "Wandal", "Nandyalamakki", "Nalwar",
    ],
}

# Karnataka road/street naming pool per district type
URBAN_STREETS: list[str] = [
    "M G Road", "Residency Road", "Brigade Road", "Rajajinagar Main Road",
    "Chamrajpet 2nd Cross", "Basaveshwaranagar 3rd Stage", "J P Nagar 6th Phase",
    "Shivajinagar Bus Stand Road", "K G Road", "Museum Road",
    "Cunningham Road", "Richmond Road", "Palace Road", "Lavelle Road",
    "Commercial Street", "Frazer Town Main Road", "Victoria Road",
    "Hosur Road", "Bannerghatta Road", "Kanakapura Road",
]

RURAL_STREETS: list[str] = [
    "NH-48 Service Road", "Taluk Office Road", "Gramapanchayat Main Road",
    "Hobli Road Near Water Tank", "Market Road", "Temple Street",
    "Old Bus Stand Road", "APMC Yard Road", "Revenue Colony Road",
    "Agricultural Cooperative Road", "Zilla Panchayat Road",
    "Forest Department Road", "Irrigation Bund Road", "Tanda Colony Road",
    "Govt Hospital Approach Road",
]

# Karnataka-specific Karnataka landmark references
# Generic landmarks found across all Karnataka districts
KARNATAKA_LANDMARKS_GENERIC: list[str] = [
    "KSRTC Bus Terminus",
    "Railway Station approach road",
    "Taluk office compound",
    "Zilla Panchayat office",
    "Government District Hospital",
    "Revenue Inspector's office",
    "APMC Market yard",
    "Krishi Seva Kendra",
    "Hopcoms vegetable market",
    "Gram Panchayat bhavan",
    "Anjaneya Swami Temple",
    "Veerabhadreshwara Temple",
    "Basaveshwara Circle",
    "Dr Ambedkar Circle",
    "SC/ST Colony entrance",
    "Milk Cooperative Society",
    "Shri Chamundeshwari Temple road",
    "Hanuman Gudi junction",
    "Sub-registrar office",
    "Primary Health Centre",
    "Taluk Agricultural Office",
    "Samudaya Bhavana",
]

# Bengaluru-specific landmarks — ONLY used when district is Bengaluru Urban/Rural
KARNATAKA_LANDMARKS_BENGALURU: list[str] = [
    "Vidhana Soudha junction",
    "Lalbagh Botanical Garden gate",
    "Cubbon Park entrance",
    "BMTC stop",
    "Kempegowda Memorial",
    "Nada Prabhu Kempegowda Park",
    "Majestic Bus Station",
    "MG Road Metro Station",
]

# Combined list for backward compatibility (used in LLM prompt examples)
KARNATAKA_LANDMARKS: list[str] = KARNATAKA_LANDMARKS_GENERIC + KARNATAKA_LANDMARKS_BENGALURU

# Karnataka culturally grounded delay reasons in FIR reporting
DELAY_REASONS_KARNATAKA: list[str] = [
    "The complainant was attending the Dasara Mahotsava celebration in Mysuru and returned to the district only after the festivities concluded.",
    "Due to the Cauvery water agitation bandh, transportation was disrupted and the complainant could not reach the police station until normalcy was restored.",
    "The family first attempted mediation through the Gram Nyayalaya under the guidance of the village panchayat elder before deciding to approach the police.",
    "Heavy monsoon flooding in the taluk blocked all roads, preventing the complainant from travelling to the police station until waters receded.",
    "The complainant was undergoing treatment at the Government District Hospital and was discharged only after a prolonged stay, after which the FIR was immediately filed.",
    "The family consulted the local Sri Muttha swamiji for guidance and social resolution before resorting to formal law enforcement.",
    "The incident occurred during the Ugadi festival period; the complainant was at a relative's home in another taluk and returned after three days.",
    "The complainant feared retaliatory action from the accused's dominant community group and sought police escort before filing the report.",
    "Rajyotsava holiday and subsequent weekend resulted in station staff being on ceremonial duty; the complainant was advised to return the following working day.",
    "The complainant, a daily wage agricultural labourer, could not afford to lose workdays during the harvest season and filed the report only after the paddy threshing was completed.",
    "The accused initially agreed to compensate through a local community arbitration arranged by the Ward Member; the agreement collapsed, prompting the formal complaint.",
    "Power outages and network disruptions in the interior hobli area prevented the complainant from contacting the PCR van or helpline for an extended period.",
    "The victim first approached the Revenue Inspector and was redirected to the Judicial Magistrate's court before finally being guided to file the FIR at the police station.",
    "The complainant was summoned as a witness in an ongoing case at the JMFC court and was unable to leave proceedings to file the FIR until the adjournment was granted.",
    "Due to the Hampi Utsava and resulting heavy tourist traffic, the complainant's vehicle was stranded on NH-67; the report was filed immediately upon reaching the city.",
]

# ---------------------------------------------------------------------------
# Crime Library — Karnataka-culturally grounded incident data
# ---------------------------------------------------------------------------

CRIME_LIBRARY: dict[str, dict[str, Any]] = {
    "Murder": {
        "incident_summary": (
            "A fatal altercation arising from a long-standing property boundary dispute near an agricultural land parcel "
            "in the taluk culminated in a grievous assault resulting in the death of the victim."
        ),
        "modus_operandi": (
            "The accused, known to the victim from the same village, allegedly attacked the victim with an iron rod and "
            "a sickle during a heated confrontation near the irrigation canal bund and fled toward the forest fringe."
        ),
        "contextual_notes": "Fellow farmers working an adjacent field raised an alarm and alerted the Gram Panchayat head.",
        "evidence": {
            "physical_items": [
                "blood-stained iron rod seized under mahazar",
                "victim's dhoti with forensic samples",
                "scene-of-offence photographs",
                "post-mortem report from Government District Hospital",
            ],
            "digital_traces": [
                "mobile tower CDR extract",
                "CCTV footage from KSRTC bus shelter 200 metres away",
            ],
            "witness_clues": [
                "statement of adjacent land farmer",
                "statement of APMC broker who witnessed the flight",
                "Gram Panchayat member's deposition",
            ],
        },
    },
    "Theft": {
        "incident_summary": (
            "Valuable property including gold ornaments of traditional Karnataka design and household cash were "
            "reported stolen from a residential house during the night hours while the family attended a neighbourhood function."
        ),
        "modus_operandi": (
            "The suspect allegedly entered through the rear window of the tiled-roof house, disabled the main door latch "
            "from inside, removed gold ornaments kept in the puja room steel almirah, and exited without disturbing the exterior."
        ),
        "contextual_notes": "The theft was discovered when the complainant returned after the Satyanarayana Pooja at the neighbour's house.",
        "evidence": {
            "physical_items": [
                "footwear impression cast near rear window",
                "disturbed puja room cupboard photographs",
                "gold jewellery description list (Kannada-format)",
                "broken latch seized as material object",
            ],
            "digital_traces": [
                "lane CCTV footage from local cable operator's camera",
                "ATM camera footage from nearest Karnataka Bank ATM",
            ],
            "witness_clues": [
                "statement of night watchman from Raithara Seva Cooperative",
                "neighbour's statement about unknown person seen loitering",
            ],
        },
    },
    "Assault": {
        "incident_summary": (
            "A physical altercation between members of two families over access to the common drinking water borewell "
            "outside the SC/ST colony resulted in injuries to two persons."
        ),
        "modus_operandi": (
            "The accused persons allegedly used wooden sticks and agricultural implements during the confrontation "
            "which took place in front of the Anjaneya Swami Temple on the taluk road at approximately 9:30 PM."
        ),
        "contextual_notes": "The Ward Member intervened and separated the parties; injured persons were taken to the Primary Health Centre.",
        "evidence": {
            "physical_items": [
                "wooden stick seized as weapon",
                "medical injury certificate from PHC",
                "torn clothing of the victim",
                "scene-of-offence sketch",
            ],
            "digital_traces": [
                "phone call records between parties",
                "PCR van dispatch log",
            ],
            "witness_clues": [
                "Ward Member's statement",
                "PHC doctor's statement",
                "auto driver who was passing by",
            ],
        },
    },
    "Cheating": {
        "incident_summary": (
            "The complainant was induced into investing in a fraudulent chit fund scheme operated by the accused, "
            "who had posed as an agent of a reputed financial cooperative society in the district."
        ),
        "modus_operandi": (
            "The accused allegedly established credibility by presenting forged documents bearing the letterhead of "
            "a Karnataka State Finance Corporation branch and collected instalments over several months before absconding."
        ),
        "contextual_notes": "Multiple complainants from the same ward have reported similar cheating patterns against the same accused.",
        "evidence": {
            "physical_items": [
                "forged chit fund receipt booklets",
                "fraudulent KSFC-branded letterhead",
                "hand-written instalment ledger",
            ],
            "digital_traces": [
                "WhatsApp message chain showing payment confirmations",
                "UPI payment receipts (Google Pay / PhonePe)",
                "email correspondence with fake corporate domain",
            ],
            "witness_clues": [
                "co-investor's statement",
                "local KSFC branch manager's statement refuting the documents",
            ],
        },
    },
    "Kidnapping": {
        "incident_summary": (
            "A minor was reported missing from the school premises during the lunch break; "
            "the complainant alleged the child was lured away by an unknown person in an autorickshaw."
        ),
        "modus_operandi": (
            "The accused allegedly waited near the government primary school gate and enticed the child with "
            "sweets before forcing the minor into a waiting yellow-and-black autorickshaw which drove toward NH-48."
        ),
        "contextual_notes": "Child helpline 1098 was notified; the DCRB and local SHO coordinated an immediate search.",
        "evidence": {
            "physical_items": [
                "school attendance register photograph",
                "autorickshaw sketch based on witness description",
            ],
            "digital_traces": [
                "toll plaza ANPR camera data",
                "mobile tower pinging on the route",
                "Railway Station CCTV extract",
            ],
            "witness_clues": [
                "school teacher's statement",
                "adjacent shop owner near school gate",
                "fruit vendor at the junction",
            ],
        },
    },
    "Rape": {
        "incident_summary": (
            "A grave sexual offence was reported by the complainant on behalf of the victim; "
            "immediate medical examination at the government hospital was undertaken upon registration."
        ),
        "modus_operandi": (
            "The accused, known to the victim through prior acquaintance at a work site, "
            "allegedly isolated the victim at a secluded location near the stone quarry "
            "and committed the offence under threat of harm."
        ),
        "contextual_notes": (
            "The matter was reported to the Sakhi One-Stop Centre at the district hospital before the FIR was registered "
            "at the request of the medical social worker."
        ),
        "evidence": {
            "physical_items": [
                "medical examination report sealed by CMOH",
                "forensic swab samples sent to FSL Bengaluru",
                "victim's clothing seized under mahazar",
            ],
            "digital_traces": [
                "call detail records of accused",
                "location data from victim's mobile",
            ],
            "witness_clues": [
                "Sakhi counsellor's statement",
                "first responder police constable's statement",
                "co-worker at the quarry site",
            ],
        },
    },
    "Cyber Fraud": {
        "incident_summary": (
            "The complainant received a call from an impersonator posing as an officer of the Karnataka Electricity "
            "Supply Company (BESCOM/HESCOM) demanding payment to avoid power disconnection, "
            "which led to financial loss through UPI transfer."
        ),
        "modus_operandi": (
            "The fraudster, using a spoofed caller-ID resembling official BESCOM helpline numbers, "
            "directed the victim to a phishing UPI link and extracted ₹{amount} through repeated small transfers "
            "to avoid threshold alerts."
        ),
        "contextual_notes": "A similar MO has been reported in 11 other cases from the district in the same month.",
        "evidence": {
            "physical_items": [
                "printed screenshot of fraudulent UPI request",
                "bank passbook extract",
            ],
            "digital_traces": [
                "UPI transaction reference ID",
                "call detail records from spoofed number",
                "IP address logs from payment gateway",
                "IMEI of device used by fraudster (traced via CDR)",
            ],
            "witness_clues": [
                "BESCOM helpdesk officer's statement confirming no such call was made",
                "bank relationship manager's statement",
            ],
        },
    },
    "Burglary": {
        "incident_summary": (
            "Unauthorized entry into a locked commercial establishment was reported; goods and cash "
            "were found missing when the owner arrived to open the shop the following morning."
        ),
        "modus_operandi": (
            "The perpetrator allegedly broke the shutter lock of the Kirana store using a heavy iron implement, "
            "removed the cash drawer and a gunny bag of goods, and fled before the dawn patrol reached the area."
        ),
        "contextual_notes": "Adjacent shop owners heard metallic sounds around 2:30 AM but assumed it was a stray dog.",
        "evidence": {
            "physical_items": [
                "broken shutter lock",
                "tool-mark photographs",
                "cash tray with fingerprint lift",
                "CCTV still frame printout",
            ],
            "digital_traces": [
                "local cable network CCTV footage",
                "mobile data of accused (cell tower fix)",
            ],
            "witness_clues": [
                "adjacent shop owner's statement",
                "night beat constable's statement",
            ],
        },
    },
    "Property Damage": {
        "incident_summary": (
            "Agricultural machinery and crops standing in the field were allegedly destroyed by the accused "
            "during a pre-dawn confrontation over a land boundary dispute."
        ),
        "modus_operandi": (
            "The accused allegedly uprooted standing sugarcane crop over a 2-acre stretch and damaged an irrigation "
            "motor set with a crowbar, causing substantial loss to the complainant."
        ),
        "contextual_notes": "The Taluk Agricultural Officer was called to assess the crop damage for official valuation.",
        "evidence": {
            "physical_items": [
                "crop damage assessment report by Taluk Agricultural Officer",
                "damaged motor set photographs",
                "mahazar of uprooted crop",
            ],
            "digital_traces": [
                "neighbour farmer's mobile video recording",
            ],
            "witness_clues": [
                "adjacent land farmer's statement",
                "Taluk Agricultural Officer's statement",
            ],
        },
    },
    "Eve Teasing": {
        "incident_summary": (
            "The complainant reported that she was subjected to verbal harassment and obscene gestures "
            "by the accused while travelling on the KSRTC bus between two taluk towns."
        ),
        "modus_operandi": (
            "The accused, a repeat offender known to the local women's protection cell, allegedly made "
            "offensive remarks and followed the victim from the bus stop before she sought help from a nearby PCR van."
        ),
        "contextual_notes": "The incident was witnessed by multiple co-passengers; the women's helpline 181 was contacted.",
        "evidence": {
            "physical_items": [
                "written complaint submitted to bus conductor",
                "victim's statement to KSRTC supervisor",
            ],
            "digital_traces": [
                "KSRTC bus onboard CCTV recording",
                "bus stop CCTV footage",
            ],
            "witness_clues": [
                "co-passenger women's statements",
                "KSRTC conductor's statement",
                "PCR van constable's statement",
            ],
        },
    },
    "Riot": {
        "incident_summary": (
            "An unlawful assembly of more than ten persons from rival community groups engaged in stone-pelting "
            "and arson near the market area during a local festival procession, causing injury to civilians and property damage."
        ),
        "modus_operandi": (
            "Members of the assembly allegedly brought sharpened rods and petrol-filled bottles concealed "
            "in vehicles; the confrontation began when the procession passed a sensitive intersection "
            "and resulted in a bonfire of two shops."
        ),
        "contextual_notes": "Section 144 CrPC / 163 BNSS was immediately invoked by the District Magistrate; RAF deployed.",
        "evidence": {
            "physical_items": [
                "seized sharp weapons under Section 102 CrPC",
                "photographs of burnt shop fronts",
                "stone fragments collected as material objects",
            ],
            "digital_traces": [
                "drone footage of the unlawful assembly",
                "mobile video recordings by bystanders",
                "social media posts inciting the assembly (archived)",
            ],
            "witness_clues": [
                "revenue inspector's panchanama",
                "fire brigade officer's statement",
                "local journalist's deposition",
            ],
        },
    },
}

# Section mapping: crime_type → {act_value → section_string}
CRIME_SECTION_MAP: dict[str, dict[str, str]] = {
    "Murder":          {LegalActEnum.IPC.value: "302",       LegalActEnum.BNS.value: "103(1)"},
    "Theft":           {LegalActEnum.IPC.value: "379",       LegalActEnum.BNS.value: "303(2)"},
    "Assault":         {LegalActEnum.IPC.value: "323",       LegalActEnum.BNS.value: "115(2)"},
    "Cheating":        {LegalActEnum.IPC.value: "420",       LegalActEnum.BNS.value: "318(4)"},
    "Kidnapping":      {LegalActEnum.IPC.value: "363",       LegalActEnum.BNS.value: "137(2)"},
    "Rape":            {LegalActEnum.IPC.value: "376",       LegalActEnum.BNS.value: "63"},
    "Cyber Fraud":     {LegalActEnum.IPC.value: "66C",       LegalActEnum.BNS.value: "66C"},   # IT Act 2000
    "Burglary":        {LegalActEnum.IPC.value: "457",       LegalActEnum.BNS.value: "305"},
    "Property Damage": {LegalActEnum.IPC.value: "427",       LegalActEnum.BNS.value: "324(4)"},
    "Eve Teasing":     {LegalActEnum.IPC.value: "354A",      LegalActEnum.BNS.value: "75"},
    "Riot":            {LegalActEnum.IPC.value: "147",       LegalActEnum.BNS.value: "189(2)"},
}

# Special act overrides for certain crime types
SPECIAL_ACT_OVERRIDES: dict[str, LegalActEnum] = {
    "Cyber Fraud": LegalActEnum.IT_Act_2000,
}

# Weighted crime type distribution reflecting real Karnataka crime statistics
CRIME_TYPE_WEIGHTS: list[tuple[str, float]] = [
    ("Theft",           0.22),
    ("Assault",         0.16),
    ("Cheating",        0.14),
    ("Cyber Fraud",     0.13),
    ("Burglary",        0.08),
    ("Property Damage", 0.08),
    ("Kidnapping",      0.05),
    ("Eve Teasing",     0.05),
    ("Rape",            0.04),
    ("Riot",            0.03),
    ("Murder",          0.02),
]


# =============================================================================
# SECTION 2: COMPREHENSIVE PYDANTIC SEED SCHEMAS
# =============================================================================

class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class EvidenceSeedModel(StrictBaseModel):
    physical_items: list[str] = Field(
        default_factory=list,
        description="Observed physical evidence items that can be woven into the FIR narrative.",
    )
    digital_traces: list[str] = Field(
        default_factory=list,
        description="Digital or electronic traces: CCTV, CDR, UPI logs, tower records.",
    )
    witness_clues: list[str] = Field(
        default_factory=list,
        description="Witness statements or observational clues available at the seed stage.",
    )


class NarrativeSeedModel(StrictBaseModel):
    crime_type: str = Field(
        ...,
        description="High-level offense category driving description and legal pivot mapping.",
    )
    incident_summary: str = Field(
        ...,
        description="Concise factual summary of the incident for the seed layer.",
    )
    modus_operandi: str = Field(
        ...,
        description="Detailed description of how the offense was allegedly committed.",
    )
    evidence: EvidenceSeedModel = Field(
        default_factory=EvidenceSeedModel,
        description="Nested evidence seed with physical, digital, and witness components.",
    )
    contextual_notes: str = Field(
        default="",
        description="Karnataka-specific cultural or administrative context notes.",
    )
    location_hint: str = Field(
        default="",
        description="LLM-supplied location hint (locality/landmark) for description construction.",
    )


# ---------------------------------------------------------------------------
# LLM Structured Output Models (used with with_structured_output)
# These are lenient (extra="ignore") so partial LLM responses don't crash.
# ---------------------------------------------------------------------------

class LLMNarrativeOutput(BaseModel):
    """Lenient narrative schema enforced on LLM output."""
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)
    crime_type: str = "Theft"
    incident_summary: str = ""
    modus_operandi: str = ""
    contextual_notes: str = ""
    location_hint: str = ""
    evidence: EvidenceSeedModel = Field(default_factory=EvidenceSeedModel)


class LLMSeedRecord(BaseModel):
    """Full seed record schema enforced on LLM output via with_structured_output."""
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)
    crime_type: str = "Theft"
    complainant_name: str = ""
    guardian_name: str = ""
    complainant_age: int = Field(default=30, ge=18, le=80)
    complainant_gender: str = "Male"
    relation_to_victim: str = "self"
    victim_name: str = ""
    victim_guardian_name: str = ""
    victim_gender: str = "Female"
    victim_age: int = Field(default=30, ge=0, le=90)
    accused_known: bool = False
    accused_name: str = "Unknown"
    accused_physical_desc: Optional[str] = None
    injury_type: Optional[str] = None
    estimated_value_inr: int = Field(default=0, ge=0)
    district: str = ""
    narrative: LLMNarrativeOutput = Field(default_factory=LLMNarrativeOutput)


class LLMSeedBatch(BaseModel):
    """Batch wrapper — LLM returns N seeds at once via with_structured_output."""
    model_config = ConfigDict(extra="ignore")
    seeds: list[LLMSeedRecord] = Field(default_factory=list)


class StationModel(StrictBaseModel):
    name: str = Field(
        ...,
        description='Police station name in the format "Police Station No. X".',
    )
    district: KarnatakaDistrict = Field(
        ...,
        description="Karnataka district the station belongs to.",
    )
    lat: float = Field(..., description="Station latitude in decimal degrees.")
    lng: float = Field(..., description="Station longitude in decimal degrees.")


class OccurrenceWindow(StrictBaseModel):
    from_time: datetime = Field(
        ..., description="Start time of the incident occurrence window."
    )
    to_time: datetime = Field(
        ...,
        description="End time of the incident occurrence window; must be on or before filed_at.",
    )


class TimelineModel(StrictBaseModel):
    filed_at: datetime = Field(..., description="FIR filing timestamp.")
    occurrence_window: OccurrenceWindow = Field(
        ..., description="Nested occurrence window describing when the offense took place."
    )
    delay_in_reporting_hours: int = Field(
        ..., ge=0, description="Computed gap between occurrence end and FIR filing in whole hours."
    )
    reason_for_delay: Optional[str] = Field(
        default=None,
        description="Populated when reporting delay exceeds 24 hours; Karnataka-culturally grounded.",
    )


class ChargeModel(StrictBaseModel):
    act: LegalActEnum = Field(
        ..., description="Applicable legal act for the charge, e.g. IPC or BNS."
    )
    section: str = Field(
        ..., min_length=1, description="Section number or subsection string under the chosen act."
    )


class LocationModel(StrictBaseModel):
    text: str = Field(
        ...,
        description="Detailed Karnataka crime scene address with locality, ward, taluk, and district.",
    )
    distance_from_ps_km: float = Field(
        ..., ge=0, description="Approximate distance from police station in kilometres."
    )
    direction_from_ps: str = Field(
        ..., description='Compass direction from the police station, e.g. "North-East".'
    )
    lat: float = Field(..., description="Crime scene latitude in decimal degrees.")
    lng: float = Field(..., description="Crime scene longitude in decimal degrees.")


class ComplainantModel(StrictBaseModel):
    name: str = Field(..., description="Complainant full name in Karnataka naming convention.")
    guardian_name: str = Field(
        ...,
        description="Guardian/parent name with S/O, D/O, or W/O prefix as per police records.",
    )
    age: int = Field(..., ge=18, description="Complainant age in years; must be an adult (18+).")
    gender: GenderEnum = Field(..., description="Complainant gender.")
    phone: str = Field(
        ...,
        pattern=r"^[6-9]\d{9}$",
        description="10-digit Indian mobile number starting with 6, 7, 8, or 9.",
    )
    relation_to_victim: str = Field(
        ..., description="Relationship of the complainant to the victim."
    )


class VictimModel(StrictBaseModel):
    name: str = Field(..., description='Victim full name or "State" for suo-moto cases.')
    guardian_name: Optional[str] = Field(
        default=None, description="Victim guardian/parent name if applicable."
    )
    age: Optional[int] = Field(default=None, ge=0, description="Victim age in years if known.")
    gender: GenderEnum = Field(..., description="Victim gender.")
    address: Optional[str] = Field(default=None, description="Victim address if available.")
    injury_type: Optional[str] = Field(
        default=None, description="Nature of injury or harm type if applicable."
    )


class AccusedModel(StrictBaseModel):
    name: str = Field(..., description='Accused full name or "Unknown".')
    is_known: bool = Field(..., description="Whether the accused identity is known.")
    physical_desc: Optional[str] = Field(
        default=None, description="Free-text physical description of the accused."
    )


class PeopleModel(StrictBaseModel):
    complainant: ComplainantModel = Field(..., description="Nested complainant details.")
    victims: list[VictimModel] = Field(
        default_factory=list, description="List of victims associated with the FIR."
    )
    accused: list[AccusedModel] = Field(
        default_factory=list, description="List of accused persons associated with the FIR."
    )


class PropertyModel(StrictBaseModel):
    items_description: Optional[str] = Field(
        default=None, description="Description of stolen or damaged property, if any."
    )
    estimated_value_inr: int = Field(
        ..., ge=0, description="Estimated property value in Indian Rupees."
    )


class InvestigationModel(StrictBaseModel):
    io_name: str = Field(
        ...,
        description='Investigating Officer name and rank, e.g. "SI Nagarajappa".',
    )
    io_id: str = Field(..., description="Internal IO identifier combining district, station, ID.")
    evidence_logged: list[str] = Field(
        default_factory=list,
        description="Evidence strings logged during investigation.",
    )
    court_case_no: Optional[str] = Field(
        default=None,
        description="Associated court case number if the case has progressed to charge sheet.",
    )


class FIRModel(StrictBaseModel):
    fir_id: int = Field(..., ge=1, description="Unique synthetic FIR identifier.")
    fir_number: str = Field(
        ...,
        pattern=r"^\d{4}/\d{4}$",
        description="Formatted as XXXX/YYYY where XXXX is zero-padded serial and YYYY is year.",
    )
    gd_reference: str = Field(..., description="General Diary reference string.")
    station: StationModel = Field(..., description="Nested police station record.")
    timeline: TimelineModel = Field(..., description="Nested timeline record.")
    information_type: InformationTypeEnum = Field(
        ..., description="How the information was received by police."
    )
    crime_type: str = Field(
        ..., description="Synthetic crime category used to derive the legal charge mapping."
    )
    charges: list[ChargeModel] = Field(
        default_factory=list, description="List of act/section pairs for the FIR."
    )
    description: str = Field(
        ..., description="Detailed narrative paragraph describing the incident."
    )
    location: LocationModel = Field(..., description="Crime scene location record.")
    status: StatusEnum = Field(..., description="Current case status.")
    people: PeopleModel = Field(..., description="Nested people record.")
    property_stolen: PropertyModel = Field(..., description="Property loss or theft details.")
    investigation: InvestigationModel = Field(..., description="Nested investigation record.")

    @field_validator("charges")
    @classmethod
    def validate_charges_against_filing_date(
        cls,
        charges: list[ChargeModel],
        info: ValidationInfo,
    ) -> list[ChargeModel]:
        """
        Enforce the July 1, 2024 legal pivot:
        - Before 2024-07-01  : only IPC-based charges are permitted (BNS raises error).
        - On/after 2024-07-01: only BNS-based charges are permitted (IPC raises error).
        Special acts (POCSO, IT Act 2000, NDPS) are exempt from this pivot rule.
        """
        timeline = info.data.get("timeline")
        if timeline is None:
            return charges

        filed_at = (
            timeline.filed_at
            if isinstance(timeline, TimelineModel)
            else timeline.get("filed_at")
        )
        pivot = PIVOT_DATE.replace(tzinfo=filed_at.tzinfo) if filed_at.tzinfo else PIVOT_DATE

        # Acts that are exempt from the IPC/BNS pivot restriction
        pivot_exempt = {LegalActEnum.POCSO_Act, LegalActEnum.IT_Act_2000, LegalActEnum.NDPS_Act}

        for charge in charges:
            act = (
                charge.act
                if isinstance(charge, ChargeModel)
                else ChargeModel.model_validate(charge).act
            )
            if act in pivot_exempt:
                continue  # Special legislation — not subject to pivot
            if filed_at < pivot and act == LegalActEnum.BNS:
                raise ValueError(
                    f"BNS charges are not permitted for FIRs filed before 2024-07-01 (filed: {filed_at.date()})."
                )
            if filed_at >= pivot and act == LegalActEnum.IPC:
                raise ValueError(
                    f"IPC charges are not permitted for FIRs filed on/after 2024-07-01 (filed: {filed_at.date()})."
                )
        return charges


# Resolve forward references for all Pydantic models
for _model_cls in [
    EvidenceSeedModel, NarrativeSeedModel,
    LLMNarrativeOutput, LLMSeedRecord, LLMSeedBatch,
    StationModel, OccurrenceWindow,
    TimelineModel, ChargeModel, LocationModel, ComplainantModel, VictimModel,
    AccusedModel, PeopleModel, PropertyModel, InvestigationModel, FIRModel,
]:
    _model_cls.model_rebuild()


# =============================================================================
# SECTION 3: DETERMINISTIC BUSINESS LOGIC & SCALING PIPELINE
# =============================================================================

# ---------------------------------------------------------------------------
# 3A — Geospatial helpers
# ---------------------------------------------------------------------------

def generate_spatial_jitter(center_lat: float, center_lng: float) -> tuple[float, float]:
    """
    Rule A — Geospatial Jittering.
    Apply a randomized offset in the ±0.01 to ±0.07 degree range so that every
    police station and crime scene has a unique coordinate realistically clustered
    around its parent district centroid.
    """
    lat_offset = random.choice([-1.0, 1.0]) * random.uniform(0.01, 0.07)
    lng_offset = random.choice([-1.0, 1.0]) * random.uniform(0.01, 0.07)
    return round(center_lat + lat_offset, 6), round(center_lng + lng_offset, 6)


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculate the great-circle distance between two coordinate pairs using the
    Haversine formula, returning the result in kilometres.
    """
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return round(R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)), 2)


def bearing_to_compass(lat1: float, lng1: float, lat2: float, lng2: float) -> str:
    """Convert bearing from station to crime scene into a human-readable compass label."""
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlng = math.radians(lng2 - lng1)
    y = math.sin(dlng) * math.cos(lat2r)
    x = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlng)
    bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    labels = ["North", "North-East", "East", "South-East", "South", "South-West", "West", "North-West"]
    return labels[int((bearing + 22.5) // 45.0) % 8]


# ---------------------------------------------------------------------------
# 3B — Identity generators (Karnataka-specific names)
# ---------------------------------------------------------------------------

def random_indian_phone() -> str:
    """Generate a random 10-digit Indian mobile number starting with 6–9."""
    return str(random.randint(6_000_000_000, 9_999_999_999))


def generate_person_name(gender: GenderEnum) -> str:
    """Generate a Karnataka-culturally appropriate full name for the given gender."""
    if gender == GenderEnum.Male:
        first = random.choice(MALE_FIRST_NAMES)
    elif gender == GenderEnum.Female:
        first = random.choice(FEMALE_FIRST_NAMES)
    else:
        first = random.choice(MALE_FIRST_NAMES + FEMALE_FIRST_NAMES)
    return f"{first} {random.choice(SURNAMES)}"


def generate_guardian_label(name: str, gender: Optional[GenderEnum] = None) -> str:
    """Generate a guardian name string using Karnataka police record format.
    If gender is provided, the prefix is locked to that gender:
      Male   → S/O or S/O Late
      Female → D/O or W/O
    """
    if gender == GenderEnum.Male:
        prefix = random.choice(["S/O", "S/O Late"])
    elif gender == GenderEnum.Female:
        prefix = random.choice(["D/O", "W/O"])
    else:
        prefix = random.choice(GUARDIAN_PREFIXES)
    guardian = generate_person_name(random.choice([GenderEnum.Male, GenderEnum.Female]))
    return f"{prefix} {guardian}"


def generate_officer_name() -> str:
    """Generate an Investigating Officer name with Karnataka Police rank."""
    rank = random.choice(OFFICER_RANKS)
    officer = random.choice(OFFICER_NAMES)
    return f"{rank} {officer}"


# ---------------------------------------------------------------------------
# 3C — Temporal helpers
# ---------------------------------------------------------------------------

def random_datetime_between(start_dt: datetime, end_dt: datetime) -> datetime:
    """Return a uniformly random datetime between start_dt and end_dt inclusive."""
    if end_dt < start_dt:
        raise ValueError("end_dt must be >= start_dt.")
    total_secs = int((end_dt - start_dt).total_seconds())
    return start_dt + timedelta(seconds=random.randint(0, total_secs))


def generate_occurrence_window(filed_at: datetime) -> OccurrenceWindow:
    """
    Rule C — Chronological Integrity.
    Build an incident occurrence window whose to_time is always ≤ filed_at.
    The delay is randomised between 0 and 96 hours; duration 20–360 minutes.
    """
    delay_hrs = random.randint(0, 96)
    delay_min = random.randint(0, 59)
    to_time = filed_at - timedelta(hours=delay_hrs, minutes=delay_min)
    if to_time > filed_at:
        to_time = filed_at
    from_time = to_time - timedelta(minutes=random.randint(20, 360))
    return OccurrenceWindow(from_time=from_time, to_time=to_time)


def generate_delay_reason(delay_hours: float) -> Optional[str]:
    """
    Return None if delay ≤ 24 hours.
    Otherwise return a Karnataka-culturally grounded textual reason.
    """
    if delay_hours <= 24:
        return None
    return random.choice(DELAY_REASONS_KARNATAKA)


def generate_timeline(filed_at: datetime) -> TimelineModel:
    occ = generate_occurrence_window(filed_at)
    exact_delay = (filed_at - occ.to_time).total_seconds() / 3600.0
    return TimelineModel(
        filed_at=filed_at,
        occurrence_window=occ,
        delay_in_reporting_hours=int(exact_delay),
        reason_for_delay=generate_delay_reason(exact_delay),
    )


def select_status(filed_at: datetime) -> StatusEnum:
    """
    Assign a realistic case status based on how old the FIR is relative to dataset end.
    Older cases are more likely to be closed; recent ones more likely open/under investigation.
    """
    age_days = max(0, (DATASET_END.date() - filed_at.date()).days)
    if age_days < 30:
        weights = [0.40, 0.45, 0.10, 0.05]
    elif age_days < 365:
        weights = [0.10, 0.40, 0.30, 0.20]
    else:
        weights = [0.05, 0.10, 0.40, 0.45]
    return random.choices(list(StatusEnum), weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# 3D — Legal Pivot Engine
# ---------------------------------------------------------------------------

def apply_legal_pivot(
    charges: list[Any],
    target_date: datetime,
    crime_type: str,
) -> list[ChargeModel]:
    """
    Rule D — The July 2024 Legal Pivot Engine.

    Determines whether the applicable act is IPC or BNS based on target_date:
      - Before 2024-07-01  → force IPC, map to correct IPC section for crime_type
      - On/after 2024-07-01 → force BNS, map to correct BNS section for crime_type

    Special legislation (IT Act 2000 for Cyber Fraud) overrides the pivot entirely.
    Preserves the existing charge count (or creates 1 charge if the list is empty).
    """
    pivot = PIVOT_DATE.replace(tzinfo=target_date.tzinfo) if target_date.tzinfo else PIVOT_DATE

    # Determine act regime
    if crime_type in SPECIAL_ACT_OVERRIDES:
        act = SPECIAL_ACT_OVERRIDES[crime_type]  # e.g. IT Act 2000 for Cyber Fraud
    else:
        act = LegalActEnum.IPC if target_date < pivot else LegalActEnum.BNS

    # Retrieve section from the map; fall back to a safe generic section
    section_map = CRIME_SECTION_MAP.get(
        crime_type,
        {LegalActEnum.IPC.value: "420", LegalActEnum.BNS.value: "318(4)"},
    )
    section = section_map.get(act.value, "420" if act == LegalActEnum.IPC else "318(4)")

    # Generate at least 1 charge; preserve count if seeds contained charges
    count = max(1, len(charges))
    return [ChargeModel(act=act, section=section) for _ in range(count)]


# ---------------------------------------------------------------------------
# 3E — Station + Location builders
# ---------------------------------------------------------------------------

def format_station_name(station_index: int) -> str:
    return f"Police Station No. {station_index}"


def select_district_for_record(fir_id: int) -> KarnatakaDistrict:
    """Distribute records evenly across all 31 districts in a round-robin fashion."""
    return list(KarnatakaDistrict)[(fir_id - 1) % len(KarnatakaDistrict)]


def build_station_model(fir_id: int, district: KarnatakaDistrict) -> StationModel:
    """
    Rule B — Station Bounding.
    Select from the 91-station master index using round-robin based on fir_id.
    Apply geospatial jitter around the district centroid for unique coordinates.
    """
    station_index = MASTER_POLICE_STATION_INDEX[(fir_id - 1) % len(MASTER_POLICE_STATION_INDEX)]
    center_lat, center_lng = DISTRICT_COORDINATES[district]
    lat, lng = generate_spatial_jitter(center_lat, center_lng)
    return StationModel(
        name=format_station_name(station_index),
        district=district,
        lat=lat,
        lng=lng,
    )


def build_location_model(
    district: KarnatakaDistrict,
    station: StationModel,
    location_hint: str = "",
) -> LocationModel:
    """
    Build a Karnataka-specific crime scene location record.
    Uses district locality map for real hobli/ward names; falls back to generic street pools.
    Applies geospatial jitter around district centroid for unique scene coordinates.
    """
    center_lat, center_lng = DISTRICT_COORDINATES[district]
    scene_lat, scene_lng = generate_spatial_jitter(center_lat, center_lng)

    distance = haversine_km(station.lat, station.lng, scene_lat, scene_lng)
    direction = bearing_to_compass(station.lat, station.lng, scene_lat, scene_lng)

    # Select a real locality name for this district
    localities = DISTRICT_LOCALITY_MAP.get(district.value, ["Town Area"])
    locality = random.choice(localities)

    # Select street type based on whether it is an urban district
    urban_districts = {
        KarnatakaDistrict.BENGALURU_URBAN,
        KarnatakaDistrict.BENGALURU_RURAL,
        KarnatakaDistrict.MYSURU,
        KarnatakaDistrict.DHARWAD,
        KarnatakaDistrict.DAKSHINA_KANNADA,
        KarnatakaDistrict.SHIVAMOGGA,
    }
    street_pool = URBAN_STREETS if district in urban_districts else RURAL_STREETS
    street = random.choice(street_pool)

    # If LLM supplied a location_hint, weave it into the text
    hint_clause = f", near {location_hint}," if location_hint.strip() else ""
    # Use Bengaluru landmarks only in Bengaluru districts; generic elsewhere
    _blr = {KarnatakaDistrict.BENGALURU_URBAN, KarnatakaDistrict.BENGALURU_RURAL}
    landmark_pool = KARNATAKA_LANDMARKS if district in _blr else KARNATAKA_LANDMARKS_GENERIC
    landmark = random.choice(landmark_pool)

    house_no = random.randint(1, 999)
    text = (
        f"House/Site No. {house_no}, {street}{hint_clause} near {landmark}, "
        f"{locality}, {district.value} District, Karnataka"
    )
    return LocationModel(
        text=text,
        distance_from_ps_km=distance,
        direction_from_ps=direction,
        lat=scene_lat,
        lng=scene_lng,
    )


# ---------------------------------------------------------------------------
# 3F — People + Property + Investigation builders
# ---------------------------------------------------------------------------

def build_people_model(payload: dict[str, Any]) -> PeopleModel:
    crime_type = payload.get("crime_type", "Theft")
    c_gender = GenderEnum(
        payload.get("complainant_gender", random.choice([GenderEnum.Male.value, GenderEnum.Female.value]))
    )
    complainant = ComplainantModel(
        name=payload.get("complainant_name") or generate_person_name(c_gender),
        guardian_name=payload.get("guardian_name") or generate_guardian_label(
            payload.get("complainant_name", "")
        ),
        age=int(payload.get("complainant_age") or random.randint(21, 65)),
        gender=c_gender,
        phone=payload.get("phone") or random_indian_phone(),
        relation_to_victim=payload.get("relation_to_victim") or random.choice(
            ["self", "spouse", "parent", "sibling", "neighbour", "shop owner", "colleague", "friend"]
        ),
    )

    victim_count = max(1, int(payload.get("victim_count") or 1))
    victims: list[VictimModel] = []
    for idx in range(victim_count):
        if idx == 0 and payload.get("victim_name"):
            v_name = payload["victim_name"]
            v_gender = GenderEnum(payload.get("victim_gender", GenderEnum.Unknown.value))
            v_age = payload.get("victim_age")
        else:
            v_gender = random.choice(list(GenderEnum))
            v_name = (
                generate_person_name(v_gender)
                if v_gender != GenderEnum.Unknown
                else "State"
            )
            v_age = random.randint(14, 75)
        victims.append(
            VictimModel(
                name=v_name,
                guardian_name=payload.get("victim_guardian_name")
                or generate_guardian_label(v_name),
                age=v_age,
                gender=v_gender,
                address=payload.get("victim_address"),
                injury_type=payload.get("injury_type"),
            )
        )

    accused_known = bool(payload.get("accused_known", False))
    accused_name_raw = payload.get("accused_name") or "Unknown"
    accused = [
        AccusedModel(
            name=accused_name_raw if accused_known else "Unknown",
            is_known=accused_known,
            physical_desc=(
                payload.get("accused_physical_desc")
                or (
                    f"Approximately 5'8\" tall, medium build, wearing dark clothing. "
                    f"Speaks {random.choice(['Kannada', 'Telugu', 'Urdu', 'Hindi'])}."
                    if accused_known
                    else None
                )
            ),
        )
    ]

    return PeopleModel(complainant=complainant, victims=victims, accused=accused)


def build_property_model(payload: dict[str, Any]) -> PropertyModel:
    crime_type = payload.get("crime_type", "Theft")
    items_description = payload.get("items_description")
    estimated_value = int(payload.get("estimated_value_inr") or 0)

    # Populate Karnataka-specific property descriptions for property crimes
    if crime_type in {"Theft", "Burglary"} and not items_description:
        items_description = random.choice([
            "gold ornaments including vaddanam, bangles (kankana), and a gold chain of traditional Karnataka design",
            "locked steel almirah contents: cash, passbook, and household gold of approx. 20 grams",
            "Kirana store inventory including edible oil tins, packed goods, and the day's cash float",
            "two-wheeler Honda Activa (AP/KA registration) with tools and documents inside the dicky",
            "mobile handset (Samsung Galaxy), Aadhaar card, and KSRTC travel pass",
        ])
    if crime_type == "Property Damage" and not items_description:
        items_description = random.choice([
            "standing sugarcane crop over 2 acres (Taluk Agricultural Officer assessed loss)",
            "irrigation motor set and associated pipeline",
            "tractor attachment (rotavator) and farm shed roof",
        ])
    if crime_type in {"Cyber Fraud", "Cheating"} and estimated_value <= 0:
        estimated_value = random.randint(10_000, 5_00_000)
    if crime_type in {"Theft", "Burglary"} and estimated_value <= 0:
        estimated_value = random.randint(5_000, 2_00_000)

    return PropertyModel(
        items_description=items_description,
        estimated_value_inr=max(0, estimated_value),
    )


def build_investigation_model(
    fir_id: int,
    district: KarnatakaDistrict,
    station: StationModel,
    seed: NarrativeSeedModel,
    status: StatusEnum,
) -> InvestigationModel:
    io_name = generate_officer_name()
    # Unique IO identifier: district abbreviation + station number + fir_id
    io_id = f"IO/{district.value[:3].upper()}/PS{station.name.split()[-1]}/{fir_id:07d}"

    # Deduplicated evidence log from seed
    evidence_logged = list(
        dict.fromkeys(
            seed.evidence.physical_items
            + seed.evidence.digital_traces
            + seed.evidence.witness_clues
        )
    )

    court_case_no = None
    if status in {StatusEnum.charge_sheet_filed, StatusEnum.closed}:
        court_case_no = f"CC/{district.value[:3].upper()}/{fir_id:07d}/S{random.randint(2010, 2026)}"

    return InvestigationModel(
        io_name=io_name,
        io_id=io_id,
        evidence_logged=evidence_logged,
        court_case_no=court_case_no,
    )


# ---------------------------------------------------------------------------
# 3G — Description generator (narrates the FIR in Karnataka style)
# ---------------------------------------------------------------------------

def generate_description(
    seed: NarrativeSeedModel,
    crime_type: str,
    district: KarnatakaDistrict,
    location: LocationModel,
    complainant: ComplainantModel,
    victims: list[VictimModel],
    accused: list[AccusedModel],
    timeline: TimelineModel,
) -> str:
    victim_names = (
        ", ".join(v.name for v in victims) if victims else "unknown persons"
    )
    accused_text = (
        "unknown accused persons"
        if not accused or not any(a.is_known for a in accused)
        else ", ".join(a.name for a in accused)
    )
    evidence_bits = (
        seed.evidence.physical_items
        + seed.evidence.digital_traces
        + seed.evidence.witness_clues
    )
    evidence_summary = (
        "; ".join(evidence_bits) if evidence_bits else "standard documentary and oral evidence"
    )

    # Build modus operandi with {amount} placeholder filled if applicable
    mo_text = seed.modus_operandi
    if "{amount}" in mo_text:
        mo_text = mo_text.replace(
            "{amount}", f"₹{random.randint(10_000, 5_00_000):,}"
        )

    return (
        f"On {timeline.occurrence_window.to_time:%d %B %Y at %H:%M hrs}, an incident of {crime_type} "
        f"was reported at {location.text}, which falls under the jurisdiction of {complainant.name}'s "
        f"registered police station in {district.value} District. The complainant, {complainant.name} "
        f"({complainant.guardian_name}), age {complainant.age} years, {complainant.gender.value}, "
        f"filed this FIR on {timeline.filed_at:%d-%m-%Y at %H:%M hrs}. "
        f"Summary of incident: {seed.incident_summary} "
        f"Alleged modus operandi: {mo_text} "
        f"Victim(s): {victim_names}. Accused: {accused_text}. "
        f"Evidence on record: {evidence_summary}. "
        f"Additional context: {seed.contextual_notes or 'No additional notes provided.'} "
        f"{'Delay reason: ' + timeline.reason_for_delay + '.' if timeline.reason_for_delay else ''}"
    ).strip()


# ---------------------------------------------------------------------------
# 3H — Master seed payload builder (pure Python, no LLM)
# ---------------------------------------------------------------------------

def generate_seed_payload(crime_type: str, district: KarnatakaDistrict) -> dict[str, Any]:
    """
    Build a deterministic Karnataka-culturally grounded seed payload using the
    hardcoded CRIME_LIBRARY. This is the fallback when Ollama is unavailable.
    """
    lib = CRIME_LIBRARY.get(crime_type, CRIME_LIBRARY["Theft"])
    complainant_gender = random.choice([GenderEnum.Male, GenderEnum.Female])
    victim_gender = random.choice([GenderEnum.Male, GenderEnum.Female, GenderEnum.Other])
    accused_known = random.random() < 0.45  # 45% of cases have a known accused

    return {
        "crime_type": crime_type,
        "district": district.value,
        "complainant_name": generate_person_name(complainant_gender),
        "guardian_name": generate_guardian_label(""),
        "complainant_age": random.randint(21, 65),
        "complainant_gender": complainant_gender.value,
        "relation_to_victim": random.choice(
            ["self", "spouse", "parent", "sibling", "neighbour", "shop owner", "colleague"]
        ),
        "victim_name": generate_person_name(victim_gender),
        "victim_guardian_name": generate_guardian_label(""),
        "victim_gender": victim_gender.value,
        "victim_age": random.randint(14, 75),
        "accused_known": accused_known,
        "accused_name": generate_person_name(GenderEnum.Male) if accused_known else "Unknown",
        "injury_type": random.choice(
            [None, "simple injury", "grievous injury", "fracture", "contusion", "fatal injury"]
        ),
        "estimated_value_inr": (
            random.randint(5_000, 5_00_000)
            if crime_type in {"Theft", "Cheating", "Cyber Fraud", "Burglary"}
            else 0
        ),
        "narrative": {
            "crime_type": crime_type,
            "incident_summary": lib["incident_summary"],
            "modus_operandi": lib["modus_operandi"],
            "contextual_notes": lib["contextual_notes"],
            "evidence": lib["evidence"],
            "location_hint": random.choice(
                DISTRICT_LOCALITY_MAP.get(district.value, ["taluk area"])
            ),
        },
    }


def parse_narrative_seed(payload: dict[str, Any]) -> NarrativeSeedModel:
    return NarrativeSeedModel.model_validate(payload.get("narrative", {}))


# ---------------------------------------------------------------------------
# 3I — Full FIR builder from seed payload
# ---------------------------------------------------------------------------

def build_fir_from_seed_payload(
    seed_payload: dict[str, Any],
    fir_id: int,
    filed_at: datetime,
) -> FIRModel:
    """
    Orchestrate all sub-builders to produce a validated FIRModel from a seed payload dict.
    This is the central assembly function that wires together all pipeline components.
    """
    district = KarnatakaDistrict(
        seed_payload.get("district", select_district_for_record(fir_id).value)
    )
    station = build_station_model(fir_id, district)
    seed = parse_narrative_seed(seed_payload)
    timeline = generate_timeline(filed_at)

    crime_type = seed_payload.get("crime_type", seed.crime_type)

    # Rule D: apply legal pivot with an empty base list (generates 1 fresh charge)
    charges = apply_legal_pivot([], filed_at, crime_type)

    people = build_people_model(seed_payload)
    location = build_location_model(district, station, seed.location_hint)
    property_stolen = build_property_model(seed_payload)
    status = select_status(filed_at)
    investigation = build_investigation_model(fir_id, district, station, seed, status)

    fir_number = f"{fir_id % 10_000:04d}/{filed_at.year}"
    gd_reference = f"GD/{district.value[:3].upper()}/PS{station.name.split()[-1]}/{fir_id:07d}"

    description = generate_description(
        seed=seed,
        crime_type=crime_type,
        district=district,
        location=location,
        complainant=people.complainant,
        victims=people.victims,
        accused=people.accused,
        timeline=timeline,
    )

    return FIRModel(
        fir_id=fir_id,
        fir_number=fir_number,
        gd_reference=gd_reference,
        station=station,
        timeline=timeline,
        information_type=random.choice(list(InformationTypeEnum)),
        crime_type=crime_type,
        charges=charges,
        description=description,
        location=location,
        status=status,
        people=people,
        property_stolen=property_stolen,
        investigation=investigation,
    )


# =============================================================================
# SECTION 4: OLLAMA LLM INTEGRATION — SEED PHASE (via LangChain ChatOllama)
# =============================================================================

# ---------------------------------------------------------------------------
# 4A — Year-weighted temporal distribution
#
# Karnataka crime registration grew roughly 8–12% per year from 2010 to 2026.
# We model this with a compound growth weight so that recent years receive
# proportionally more records, matching real-world crime reporting trends.
# ---------------------------------------------------------------------------

# Annual growth rate: each year gets ~10% more cases than the previous one
YEAR_GROWTH_RATE: float = 1.10


def build_year_weights(start_year: int, end_year: int) -> dict[int, float]:
    """
    Build a {year: weight} mapping with exponentially increasing weights.
    The weight for each year y is YEAR_GROWTH_RATE^(y - start_year), so:
      2010 → 1.00,  2015 → 1.61,  2020 → 2.59,  2025 → 4.18
    This ensures the generated dataset has fewer records in early years and
    more records in later years, reflecting real Karnataka crime growth.
    """
    return {
        y: YEAR_GROWTH_RATE ** (y - start_year)
        for y in range(start_year, end_year + 1)
    }


def year_weighted_datetime(
    start_date: date,
    end_date: date,
    year_weights: dict[int, float],
) -> datetime:
    """
    Pick a random datetime with a bias toward more recent years.

    Algorithm:
      1. Sample a year from year_weights using weighted random choice.
      2. Within that year, sample a uniform random datetime.
      3. Clamp the result so it never exceeds end_date or precedes start_date.

    This produces a dataset whose per-year case count increases at ~10% per year
    from start_date to end_date, matching Karnataka crime growth trends.
    """
    years = list(year_weights.keys())
    weights = list(year_weights.values())

    # Weighted year selection
    chosen_year: int = random.choices(years, weights=weights, k=1)[0]

    # Clamp year boundaries against the overall date range
    year_start = datetime(
        chosen_year, 1, 1, 0, 0, 0
    )
    year_end = datetime(
        chosen_year, 12, 31, 23, 59, 59
    )
    effective_start = max(year_start, datetime.combine(start_date, datetime.min.time()))
    effective_end   = min(year_end,   datetime.combine(end_date,   datetime(1, 1, 1, 23, 59, 59).time()))

    # If the clamped range is degenerate (can happen at boundary years), fall back
    if effective_end <= effective_start:
        return effective_start

    return random_datetime_between(effective_start, effective_end)



# ---------------------------------------------------------------------------
# 4B — LangChain ChatOllama wrapper
# ---------------------------------------------------------------------------

def _invoke_ollama(
    model: str,
    prompt: str,
    max_retries: int = 3,
) -> LLMSeedBatch:
    """
    Call the local Ollama model via LangChain ChatOllama with structured output.

    Uses llm.with_structured_output(LLMSeedBatch) so the LLM is forced to
    return data that matches the schema directly — no manual JSON parsing needed.
    ChatOllama connects to http://localhost:11434 automatically.

    Returns a validated LLMSeedBatch object.
    Retries up to max_retries times on LangChain/Ollama errors.
    """
    llm = ChatOllama(
        model=model,
        temperature=0.7,        # Lower = more consistent structure; raise to 0.85 for variety
        top_p=0.92,
        num_predict=2048,
    )
    # with_structured_output enforces the LLMSeedBatch Pydantic schema at the model level
    structured_llm = llm.with_structured_output(LLMSeedBatch)

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            result = structured_llm.invoke(prompt)
            return result  # type: ignore[return-value]
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log.warning(
                "ChatOllama structured attempt %d/%d failed: %s",
                attempt, max_retries, exc,
            )

    raise RuntimeError(
        f"ChatOllama unreachable after {max_retries} attempts: {last_exc}. "
        "Ensure Ollama is running (`ollama serve`) and the model is pulled."
    )


def build_llm_prompt(
    district: KarnatakaDistrict,
    crime_types: list[str],
    batch_size: int,
) -> str:
    """
    Build a structured prompt instructing the LLM to generate a JSON array of
    Karnataka FIR narrative seeds. The prompt is tightly constrained so the
    output can be parsed deterministically.
    """
    crime_list_str = json.dumps(crime_types)
    locality_examples = ", ".join(
        DISTRICT_LOCALITY_MAP.get(district.value, ["taluk area"])[:5]
    )
    landmark_examples = ", ".join(random.sample(KARNATAKA_LANDMARKS, 4))
    male_names = ", ".join(random.sample(MALE_FIRST_NAMES, 5))
    female_names = ", ".join(random.sample(FEMALE_FIRST_NAMES, 5))
    surnames = ", ".join(random.sample(SURNAMES, 5))

    return f"""You are a synthetic data generator for the Karnataka State Police (India).
Generate exactly {batch_size} FIR (First Information Report) narrative seeds as a JSON object.

### Context
- District: {district.value}, Karnataka, India
- Real localities in this district: {locality_examples}
- Karnataka landmarks: {landmark_examples}
- Male Kannada names to use: {male_names} (combine with surnames: {surnames})
- Female Kannada names to use: {female_names} (combine with surnames: {surnames})
- Crime types to distribute across the {batch_size} records: {crime_list_str}

### STRICT OUTPUT FORMAT
Return ONLY this JSON object, no text before or after:
{{
  "seeds": [
    {{
      "crime_type": "<one of the listed crime types>",
      "complainant_name": "<Kannada full name>",
      "guardian_name": "<S/O or D/O or W/O Kannada full name>",
      "complainant_age": <integer 21-65>,
      "complainant_gender": "<Male|Female>",
      "relation_to_victim": "<self|spouse|parent|sibling|neighbour|colleague>",
      "victim_name": "<Kannada full name>",
      "victim_guardian_name": "<S/O or D/O Kannada name>",
      "victim_gender": "<Male|Female|Other>",
      "victim_age": <integer 14-75>,
      "accused_known": <true|false>,
      "accused_name": "<Kannada full name or 'Unknown'>",
      "accused_physical_desc": "<description or null>",
      "injury_type": "<null|simple injury|grievous injury|fracture|fatal injury>",
      "estimated_value_inr": <integer, 0 if no property crime>,
      "district": "{district.value}",
      "narrative": {{
        "crime_type": "<same as above>",
        "incident_summary": "<2-3 sentence Karnataka-specific summary mentioning local place, culture, or context>",
        "modus_operandi": "<2-3 sentences describing how the crime was committed, referencing Karnataka local details>",
        "contextual_notes": "<1 sentence with Karnataka administrative/cultural context>",
        "location_hint": "<specific locality or landmark in {district.value} district>",
        "evidence": {{
          "physical_items": ["<item 1>", "<item 2>"],
          "digital_traces": ["<trace 1>"],
          "witness_clues": ["<clue 1>", "<clue 2>"]
        }}
      }}
    }}
  ]
}}

### Rules
1. All names MUST be Karnataka Kannada names — no generic Hindi/English names.
2. incident_summary and modus_operandi MUST reference real places in {district.value}.
3. The location_hint must be a real locality in {district.value} district.
4. evidence items must be realistic Karnataka police evidence terminology.
5. Do NOT include any text outside the JSON object.
"""


def parse_llm_response(text: str) -> list[dict[str, Any]]:
    """
    Robustly extract the 'seeds' array from the LLM's JSON response.
    Handles common LLM output quirks: extra markdown fences, trailing commas,
    partial JSON, and nested extraction.
    """
    if not text or not text.strip():
        return []

    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text.strip(), flags=re.MULTILINE)
    text = text.strip()

    # Attempt 1: parse as-is
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "seeds" in obj:
            return obj["seeds"]
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass

    # Attempt 2: find outermost JSON object via brace matching
    brace_start = text.find("{")
    if brace_start != -1:
        depth, i = 0, brace_start
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[brace_start: i + 1])
                        if isinstance(obj, dict) and "seeds" in obj:
                            return obj["seeds"]
                    except json.JSONDecodeError:
                        break
            i += 1

    # Attempt 3: find a JSON array directly
    bracket_start = text.find("[")
    if bracket_start != -1:
        depth, i = 0, bracket_start
        while i < len(text):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[bracket_start: i + 1])
                    except json.JSONDecodeError:
                        break
            i += 1

    log.warning("Could not parse LLM response as valid JSON. Snippet: %s", text[:200])
    return []


def merge_with_library_fallback(
    llm_seed: dict[str, Any],
    crime_type: str,
    district: KarnatakaDistrict,
) -> dict[str, Any]:
    """
    Merge an LLM-generated seed with the hardcoded CRIME_LIBRARY to fill any
    missing or malformed fields. The LLM content takes priority; library fills gaps.
    """
    lib = CRIME_LIBRARY.get(crime_type, CRIME_LIBRARY["Theft"])
    narrative = llm_seed.get("narrative", {})

    # Ensure the narrative sub-dict has all required fields
    # Use explicit checks instead of setdefault to also catch empty strings
    narrative.setdefault("crime_type", crime_type)
    if not narrative.get("incident_summary"):
        narrative["incident_summary"] = lib["incident_summary"]
    if not narrative.get("modus_operandi"):
        narrative["modus_operandi"] = lib["modus_operandi"]
    if not narrative.get("contextual_notes"):
        narrative["contextual_notes"] = lib["contextual_notes"]
    if not narrative.get("location_hint"):
        narrative["location_hint"] = ""
    ev = narrative.setdefault("evidence", {})
    if not ev.get("physical_items"):
        ev["physical_items"] = lib["evidence"]["physical_items"]
    if not ev.get("digital_traces"):
        ev["digital_traces"] = lib["evidence"]["digital_traces"]
    if not ev.get("witness_clues"):
        ev["witness_clues"] = lib["evidence"]["witness_clues"]

    # Ensure top-level seed fields have values
    if not llm_seed.get("complainant_name"):
        llm_seed["complainant_name"] = generate_person_name(GenderEnum.Male)
    if not llm_seed.get("victim_name"):
        llm_seed["victim_name"] = generate_person_name(GenderEnum.Female)
    if not llm_seed.get("guardian_name"):
        llm_seed["guardian_name"] = generate_guardian_label("")
    if not llm_seed.get("complainant_age"):
        llm_seed["complainant_age"] = random.randint(21, 65)
    # Sanitize gender fields — LLM often produces "", "null", or other invalid values
    _valid_genders = {e.value for e in GenderEnum}
    cg = llm_seed.get("complainant_gender", "")
    if not cg or cg not in _valid_genders:
        llm_seed["complainant_gender"] = GenderEnum.Male.value
    vg = llm_seed.get("victim_gender", "")
    if not vg or vg not in _valid_genders:
        llm_seed["victim_gender"] = GenderEnum.Female.value
    llm_seed.setdefault("victim_age", random.randint(20, 60))
    llm_seed.setdefault("accused_known", random.random() < 0.45)
    llm_seed.setdefault("accused_name", "Unknown")
    # Sanitize injury_type — LLM sometimes outputs the string "null" instead of JSON null
    if llm_seed.get("injury_type") in (None, "null", "None", ""):
        llm_seed["injury_type"] = None
    llm_seed.setdefault("estimated_value_inr", 0)
    llm_seed["district"] = district.value
    llm_seed["crime_type"] = crime_type
    llm_seed["narrative"] = narrative

    return llm_seed


def run_llm_seed_loop(
    output_path: str | Path,
    loops: int = 500,
    batch_size: int = 7,
    model: str = "llama3.1:8b",
    fallback_on_error: bool = True,
    rng_seed: int = 42,
) -> int:
    """
    Phase 1 — LLM Seed Loop (via LangChain ChatOllama).

    Invokes the local Ollama model `loops` times using LangChain's ChatOllama
    interface — no endpoint URL configuration is required. Each loop requests
    `batch_size` (5–10) FIR narrative seeds for a randomly selected Karnataka
    district and crime type mix. Seeds are validated and streamed directly to
    `output_path` as JSON Lines (O(1) memory footprint).

    On LLM failure (model not available, parse error, etc.), the loop falls back
    to the hardcoded CRIME_LIBRARY pure-Python seed generator so the pipeline
    never stalls entirely.

    Returns the total number of valid seed records written.
    """
    random.seed(rng_seed)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    crime_pool = [ct for ct, _ in CRIME_TYPE_WEIGHTS]
    districts = list(KarnatakaDistrict)

    # ── Auto-resume: detect already-completed loops ────────────────────────
    existing_lines = 0
    if path.exists():
        with path.open("r", encoding="utf-8") as _rf:
            for _l in _rf:
                if _l.strip():
                    existing_lines += 1
    completed_loops = existing_lines // batch_size  # best estimate
    if completed_loops > 0:
        log.info(
            "Resume detected: %d existing seeds (~%d loops done). Skipping ahead.",
            existing_lines, completed_loops,
        )
        # Fast-forward random state to match where we left off (no I/O)
        for _skip in range(completed_loops):
            random.choice(districts)
            random.choices(
                crime_pool,
                weights=[w for _, w in CRIME_TYPE_WEIGHTS],
                k=batch_size,
            )
        total_written = existing_lines

    log.info(
        "Starting LLM seed loop: %d iterations × %d records/loop = ~%d seeds (model: %s)",
        loops, batch_size, loops * batch_size, model,
    )
    log.info("Using LangChain ChatOllama — no endpoint config required.")

    # Append mode — existing seeds are preserved; new seeds are added below them
    with path.open("a", encoding="utf-8") as fh:
        for loop_idx in range(1, loops + 1):
            # Skip already-completed loops
            if loop_idx <= completed_loops:
                continue
            # ── Select district + crime mix for this batch ─────────────────
            district = random.choice(districts)
            batch_crimes = random.choices(
                crime_pool,
                weights=[w for _, w in CRIME_TYPE_WEIGHTS],
                k=batch_size,
            )
            # Deduplicate while preserving order (for the prompt's crime list)
            unique_crimes = list(dict.fromkeys(batch_crimes))

            log.info(
                "[Loop %3d/%d] District=%-18s Crimes=%s",
                loop_idx, loops, district.value, unique_crimes,
            )

            prompt = build_llm_prompt(district, unique_crimes, batch_size)
            raw_seeds: list[dict[str, Any]] = []

            # ── Call ChatOllama with structured output (schema-enforced) ───
            try:
                seed_batch: LLMSeedBatch = _invoke_ollama(model=model, prompt=prompt)
                raw_seeds = [s.model_dump() for s in seed_batch.seeds]
                if raw_seeds:
                    log.info("  └─ LLM returned %d seed(s) [structured output]", len(raw_seeds))
            except Exception as exc:  # Catch ALL errors (ValidationError, RuntimeError, etc.)
                log.error("LLM call failed at loop %d: %s", loop_idx, exc)
                if not fallback_on_error:
                    raise
                log.warning("  └─ Falling back to pure-Python seed generation for this loop.")

            # ── Fallback: pure-Python seeds if LLM produced nothing ────────
            if not raw_seeds:
                raw_seeds = [generate_seed_payload(ct, district) for ct in batch_crimes]

            # ── Write valid seeds to JSONL ─────────────────────────────────
            for seed_dict in raw_seeds[:batch_size]:
                crime_type = seed_dict.get("crime_type", random.choice(crime_pool))
                if crime_type not in CRIME_LIBRARY:
                    crime_type = random.choice(crime_pool)

                merged = merge_with_library_fallback(seed_dict, crime_type, district)
                fh.write(json.dumps(merged, ensure_ascii=False, default=str) + "\n")
                total_written += 1

            # Force-flush to disk after every loop so no data is lost on crash/stop
            fh.flush()
            os.fsync(fh.fileno())
            if loop_idx % 50 == 0:
                log.info("Progress: %d seeds written after %d loops.", total_written, loop_idx)

    log.info("Seed phase complete. Total seeds written: %d → %s", total_written, path)
    return total_written


# =============================================================================
# SECTION 5: AUGMENTATION ENGINE — SCALE PHASE
# =============================================================================

# Fields that are systematically mutated to create variants
_AUGMENTATION_CRIME_OFFSETS: dict[str, str] = {
    "Theft": "Burglary",
    "Burglary": "Theft",
    "Cheating": "Cyber Fraud",
    "Cyber Fraud": "Cheating",
    "Assault": "Riot",
    "Riot": "Assault",
    "Murder": "Murder",  # kept the same, date/district varied
    "Kidnapping": "Kidnapping",
    "Rape": "Rape",
    "Property Damage": "Property Damage",
    "Eve Teasing": "Eve Teasing",
}


def augment_seed_record(
    seed_dict: dict[str, Any],
    variant_index: int,
    target_date: datetime,
    district_override: Optional[KarnatakaDistrict] = None,
) -> dict[str, Any]:
    """
    Create a variant of an existing seed by mutating non-narrative fields.
    The variant_index determines which fields change, ensuring systematic coverage.

    Mutation strategy by variant_index modulo:
      mod 0  : change district only
      mod 1  : change complainant/victim names + phone
      mod 2  : change accused known/unknown flag + physical desc
      mod 3  : change district + names + estimated value
      mod 4  : swap crime_type to a related category
      mod 5  : change injury type + relation to victim
      mod 6  : change complainant age + gender
      mod 7  : change all identities (full person refresh)
    """
    variant = copy.deepcopy(seed_dict)
    mod = variant_index % 8

    # ── ALWAYS refresh identities for every variant ──────────────────────
    # This prevents hundreds of records sharing "Santosh Kambali / Girish Singh"
    c_gender = random.choice([GenderEnum.Male, GenderEnum.Female])
    variant["complainant_name"] = generate_person_name(c_gender)
    variant["complainant_gender"] = c_gender.value
    variant["guardian_name"] = generate_guardian_label("", gender=c_gender)
    variant["complainant_age"] = random.randint(21, 65)
    variant["phone"] = random_indian_phone()

    v_gender = random.choice([GenderEnum.Male, GenderEnum.Female])
    variant["victim_name"] = generate_person_name(v_gender)
    variant["victim_gender"] = v_gender.value
    variant["victim_guardian_name"] = generate_guardian_label("", gender=v_gender)
    variant["victim_age"] = max(1, random.randint(14, 75))

    # ── ALWAYS refresh evidence from CRIME_LIBRARY to match crime type ───
    ct = variant.get("crime_type", "Theft")
    lib = CRIME_LIBRARY.get(ct, CRIME_LIBRARY["Theft"])
    if "narrative" in variant:
        variant["narrative"]["evidence"] = copy.deepcopy(lib["evidence"])

    # Always randomise the district if an override is provided
    if district_override:
        variant["district"] = district_override.value
        # Update narrative location_hint to reflect new district
        if "narrative" in variant:
            new_locs = DISTRICT_LOCALITY_MAP.get(district_override.value, ["taluk area"])
            variant["narrative"]["location_hint"] = random.choice(new_locs)

    if mod in {0, 3}:
        # Rotate district (if no override, pick a new random one)
        if not district_override:
            new_district = random.choice(list(KarnatakaDistrict))
            variant["district"] = new_district.value
            if "narrative" in variant:
                locs = DISTRICT_LOCALITY_MAP.get(new_district.value, ["taluk area"])
                variant["narrative"]["location_hint"] = random.choice(locs)

    if mod in {2, 7}:
        # Flip accused known status
        accused_known = not bool(variant.get("accused_known", False))
        variant["accused_known"] = accused_known
        if accused_known:
            variant["accused_name"] = generate_person_name(GenderEnum.Male)
            variant["accused_physical_desc"] = (
                f"Approximately {random.choice(['5 feet 6 inches', '5 feet 9 inches', '6 feet'])} tall, "
                f"{random.choice(['slim', 'medium', 'heavy'])} build, "
                f"speaks {random.choice(['Kannada', 'Telugu', 'Urdu'])}."
            )
        else:
            variant["accused_name"] = "Unknown"
            variant["accused_physical_desc"] = None

    if mod in {3, 5}:
        # Adjust estimated value and injury type
        if ct in {"Theft", "Cheating", "Cyber Fraud", "Burglary"}:
            variant["estimated_value_inr"] = random.randint(5_000, 5_00_000)
        variant["injury_type"] = random.choice(
            [None, "simple injury", "grievous injury", "fracture", "contusion"]
        )

    if mod == 4:
        # Swap to a related crime type
        old_ct = variant.get("crime_type", "Theft")
        new_ct = _AUGMENTATION_CRIME_OFFSETS.get(old_ct, old_ct)
        variant["crime_type"] = new_ct
        if "narrative" in variant:
            new_lib = CRIME_LIBRARY.get(new_ct, CRIME_LIBRARY["Theft"])
            variant["narrative"]["crime_type"] = new_ct
            variant["narrative"]["incident_summary"] = new_lib["incident_summary"]
            variant["narrative"]["modus_operandi"] = new_lib["modus_operandi"]
            variant["narrative"]["evidence"] = copy.deepcopy(new_lib["evidence"])

    if mod in {5, 6}:
        # Change relation and age
        variant["relation_to_victim"] = random.choice(
            ["self", "spouse", "parent", "sibling", "neighbour", "colleague"]
        )
        variant["complainant_age"] = random.randint(21, 65)

    if mod in {6, 7}:
        # Flip complainant gender
        new_gender = (
            GenderEnum.Female
            if variant.get("complainant_gender") == GenderEnum.Male.value
            else GenderEnum.Male
        )
        variant["complainant_gender"] = new_gender.value
        variant["complainant_name"] = generate_person_name(new_gender)
        variant["guardian_name"] = generate_guardian_label("", gender=new_gender)

    return variant


# ---------------------------------------------------------------------------
# 5B — Post-augmentation seed sanitizer (enforces all guardrail rules)
# ---------------------------------------------------------------------------

# Bengaluru-specific landmarks that must NEVER appear in non-Bengaluru districts
_BENGALURU_LANDMARKS = {
    "BMTC", "Cubbon Park", "Lalbagh", "Lalbagh Botanical Garden",
    "Vidhana Soudha", "MG Road", "Brigade Road", "Koramangala",
    "Electronic City", "Whitefield", "Majestic", "Jayanagar",
    "Basavanagudi", "HSR Layout", "BTM Layout", "Indiranagar",
}

# Coastal districts — only these may reference beaches, ports, harbours
_COASTAL_DISTRICTS = {"Dakshina Kannada", "Udupi", "Uttara Kannada"}

# Bengaluru districts
_BENGALURU_DISTRICTS = {"Bengaluru Urban", "Bengaluru Rural"}

# District-specific contextual notes for realistic colour
_DISTRICT_CONTEXT_MAP: dict[str, list[str]] = {
    "Bengaluru Urban": [
        "The area is a densely populated IT corridor with high foot traffic.",
        "This locality falls within the jurisdiction of the CBD police sub-division.",
        "The incident occurred near an active metro construction zone.",
    ],
    "Bengaluru Rural": [
        "This area is a rapidly urbanizing peri-urban zone near the airport corridor.",
        "The incident occurred in a mixed agricultural and industrial zone.",
        "The locality is a growing residential satellite town.",
    ],
    "Mandya": [
        "The region is known for its sugarcane cultivation and jaggery trade.",
        "This is a major sugar-producing belt in the Cauvery river basin.",
        "The area falls in the heart of the Mandya irrigation command area.",
    ],
    "Ballari": [
        "The district is known for iron ore mining and related industrial activity.",
        "The area is adjacent to the Sandur mining belt.",
        "Ballari is a major commercial hub in the Hyderabad-Karnataka region.",
    ],
    "Mysuru": [
        "The area is a popular heritage and tourism destination.",
        "The incident occurred in the vicinity of the university campus.",
        "Mysuru's silk weaving industry attracts migrant workers from neighbouring states.",
    ],
    "Dakshina Kannada": [
        "The coastal belt sees high tourist and fishing activity year-round.",
        "The port area has significant commercial shipping and trade traffic.",
        "This is part of the Mangaluru urban agglomeration with dense residential areas.",
    ],
    "Udupi": [
        "The district is known for its temple tourism and educational institutions.",
        "Manipal's student population contributes to a unique urban-rural dynamic.",
        "The coastal fishing communities are economically active in this taluk.",
    ],
    "Uttara Kannada": [
        "The area is a mix of coastal fishing villages and dense Western Ghats forest.",
        "The Dandeli wildlife sanctuary draws eco-tourism in this region.",
        "Karwar port and naval base drive economic activity in this district.",
    ],
    "Bidar": [
        "The district has a strong presence of the tur dal and jowar agricultural economy.",
        "Bidar is historically significant for its Bahmani-era fort and cultural heritage.",
        "The area borders Telangana and has significant cross-state commuter traffic.",
    ],
}
# Fallback for districts not in the map
_DEFAULT_CONTEXT = [
    "The incident occurred in a mixed agricultural and residential area.",
    "The locality is served by the nearest taluk police station.",
    "This area is part of the district's main commercial and residential zone.",
]

# Crime types that require injury_type to NOT be null
_VIOLENT_CRIMES = {"Murder", "Assault", "Riot"}

# Crime types that require estimated_value > 0
_PROPERTY_CRIMES = {"Theft", "Burglary", "Property Damage", "Cheating", "Cyber Fraud"}

# Crime types where injury_type MUST be null
_NO_INJURY_CRIMES = {"Cyber Fraud"}

# Valid injury types
_VALID_INJURIES = {"simple injury", "grievous injury", "fracture", "contusion", "fatal injury"}


def sanitize_augmented_seed(variant: dict[str, Any]) -> dict[str, Any]:
    """
    Enforce all guardrail rules on a seed dict AFTER augmentation
    and BEFORE it is passed to build_fir_from_seed_payload.

    Rules enforced:
      1. Gender-prefix lock on guardian names
      2. Age/relational logic (self, spouse, parent)
      3. Identity collision prevention (accused ≠ complainant/victim)
      4. Geographic alignment (no Bengaluru landmarks in rural districts)
      5. Crime-type / evidence alignment
      6. Strict null enforcement (no "None", "N/A", "" strings)
    """
    crime_type = variant.get("crime_type", "Theft")
    district = variant.get("district", "")

    # ── Rule 6: Strict Null Enforcement (run first to normalize data) ─────
    for key in ("accused_physical_desc", "injury_type", "victim_address"):
        val = variant.get(key)
        if val in ("None", "N/A", "null", ""):
            variant[key] = None

    # Normalize victim_guardian_name
    vgn = variant.get("victim_guardian_name")
    if vgn in ("N/A", "None", "null", ""):
        variant["victim_guardian_name"] = generate_guardian_label("")

    # ── Rule 6c: Gender Enum Sanitization ─────────────────────────────────
    _valid_genders = {e.value for e in GenderEnum}
    cg = variant.get("complainant_gender", "")
    if not cg or cg not in _valid_genders:
        variant["complainant_gender"] = GenderEnum.Male.value
    vg = variant.get("victim_gender", "")
    if not vg or vg not in _valid_genders:
        variant["victim_gender"] = GenderEnum.Female.value

    # ── Rule 1: Gender-Prefix Lock ────────────────────────────────────────
    c_gender_str = variant.get("complainant_gender", "Male")
    c_gender = GenderEnum(c_gender_str) if c_gender_str in _valid_genders else GenderEnum.Male
    guardian_name = variant.get("guardian_name", "")
    if guardian_name:
        # Extract prefix and guardian person name
        if c_gender == GenderEnum.Male:
            # Males MUST use S/O or S/O Late
            if not guardian_name.startswith(("S/O", "H/O")):
                # Replace prefix with S/O, keep the guardian person name
                parts = guardian_name.split(" ", 1)
                # Find where the actual name starts (after any prefix like D/O, W/O)
                for i, p in enumerate(parts):
                    if "/" not in p and p != "Late":
                        person_name = " ".join(parts[i:])
                        break
                else:
                    person_name = generate_person_name(random.choice([GenderEnum.Male, GenderEnum.Female]))
                variant["guardian_name"] = f"S/O {person_name}"
        elif c_gender == GenderEnum.Female:
            # Females MUST use D/O or W/O
            if not guardian_name.startswith(("D/O", "W/O", "S/O Late")):
                parts = guardian_name.split(" ", 1)
                for i, p in enumerate(parts):
                    if "/" not in p and p != "Late":
                        person_name = " ".join(parts[i:])
                        break
                else:
                    person_name = generate_person_name(random.choice([GenderEnum.Male, GenderEnum.Female]))
                prefix = random.choice(["D/O", "W/O"])
                variant["guardian_name"] = f"{prefix} {person_name}"

    # Same for victim_guardian_name
    v_gender_str = variant.get("victim_gender", "Male")
    v_gender = GenderEnum(v_gender_str) if v_gender_str in _valid_genders else GenderEnum.Male
    v_guardian = variant.get("victim_guardian_name", "")
    if v_guardian:
        if v_gender == GenderEnum.Male:
            if not v_guardian.startswith(("S/O", "H/O")):
                parts = v_guardian.split(" ", 1)
                for i, p in enumerate(parts):
                    if "/" not in p and p != "Late":
                        person_name = " ".join(parts[i:])
                        break
                else:
                    person_name = generate_person_name(random.choice([GenderEnum.Male, GenderEnum.Female]))
                variant["victim_guardian_name"] = f"S/O {person_name}"
        elif v_gender == GenderEnum.Female:
            if not v_guardian.startswith(("D/O", "W/O")):
                parts = v_guardian.split(" ", 1)
                for i, p in enumerate(parts):
                    if "/" not in p and p != "Late":
                        person_name = " ".join(parts[i:])
                        break
                else:
                    person_name = generate_person_name(random.choice([GenderEnum.Male, GenderEnum.Female]))
                prefix = random.choice(["D/O", "W/O"])
                variant["victim_guardian_name"] = f"{prefix} {person_name}"

    # ── Rule 1b: Marital Logic — W/O requires age >= 18 ───────────────────
    if variant.get("guardian_name", "").startswith("W/O"):
        if (variant.get("complainant_age") or 0) < 18:
            variant["complainant_age"] = random.randint(18, 65)
    if variant.get("victim_guardian_name", "").startswith("W/O"):
        if (variant.get("victim_age") or 0) < 18:
            variant["victim_age"] = random.randint(18, 60)

    # ── Rule 2a: "Self" Paradox — complainant == victim ───────────────────
    relation = variant.get("relation_to_victim", "")
    if relation == "self":
        variant["victim_name"] = variant.get("complainant_name", variant.get("victim_name", ""))
        variant["victim_age"] = variant.get("complainant_age", variant.get("victim_age", 30))
        variant["victim_gender"] = variant.get("complainant_gender", variant.get("victim_gender", "Male"))
        variant["victim_guardian_name"] = variant.get("guardian_name", variant.get("victim_guardian_name", ""))

    # ── Rule 2b: "Spouse" Rule — distinct names, opposite genders ─────────
    if relation == "spouse":
        c_name = variant.get("complainant_name", "")
        v_name = variant.get("victim_name", "")
        # Ensure distinct names
        if c_name == v_name or not v_name:
            opp_gender = GenderEnum.Female if c_gender == GenderEnum.Male else GenderEnum.Male
            variant["victim_name"] = generate_person_name(opp_gender)
            variant["victim_gender"] = opp_gender.value
        else:
            # Ensure opposite genders
            if variant.get("victim_gender") == variant.get("complainant_gender"):
                opp_gender = GenderEnum.Female if c_gender == GenderEnum.Male else GenderEnum.Male
                variant["victim_gender"] = opp_gender.value
                variant["victim_name"] = generate_person_name(opp_gender)

    # ── Rule 2c: "Parent" Rule — complainant age > victim age + 18 ────────
    if relation == "parent":
        c_age = variant.get("complainant_age", 40)
        v_age = variant.get("victim_age", 20)
        if c_age < v_age + 18:
            # Adjust: make complainant older or victim younger
            variant["complainant_age"] = max(c_age, v_age + random.randint(18, 30))
            if variant["complainant_age"] > 75:
                variant["victim_age"] = max(1, variant["complainant_age"] - random.randint(18, 30))

    # ── Rule 2d: Identity Collision — accused ≠ complainant/victim ────────
    accused_name = variant.get("accused_name", "Unknown")
    c_name = variant.get("complainant_name", "")
    v_name = variant.get("victim_name", "")
    if accused_name != "Unknown" and (accused_name == c_name or accused_name == v_name):
        variant["accused_name"] = generate_person_name(GenderEnum.Male)

    # ── Rule 4a: Unknown Accused — name MUST be exactly "Unknown" ─────────
    if not variant.get("accused_known", False):
        variant["accused_name"] = "Unknown"
        variant["accused_physical_desc"] = None

    # ── Rule 3: Geographic Alignment ──────────────────────────────────────
    narrative = variant.get("narrative", {})
    if narrative and district:
        summary = narrative.get("incident_summary", "")
        mo = narrative.get("modus_operandi", "")
        ctx = narrative.get("contextual_notes", "")

        # 3a: Eradicate Bengaluru landmarks from non-Bengaluru districts
        if district not in _BENGALURU_DISTRICTS:
            for landmark in _BENGALURU_LANDMARKS:
                if landmark in summary or landmark in mo or landmark in ctx:
                    # Replace entire narrative from CRIME_LIBRARY to be safe
                    lib = CRIME_LIBRARY.get(crime_type, CRIME_LIBRARY["Theft"])
                    narrative["incident_summary"] = lib["incident_summary"]
                    narrative["modus_operandi"] = lib["modus_operandi"]
                    break

        # 3b: Remove coastal references from landlocked districts
        if district not in _COASTAL_DISTRICTS:
            coastal_terms = ["beach", "port", "harbour", "harbor", "fishing boat", "trawler", "coastal"]
            combined = (summary + " " + mo + " " + ctx).lower()
            if any(term in combined for term in coastal_terms):
                lib = CRIME_LIBRARY.get(crime_type, CRIME_LIBRARY["Theft"])
                narrative["incident_summary"] = lib["incident_summary"]
                narrative["modus_operandi"] = lib["modus_operandi"]

        # 3c: Replace generic boilerplate contextual_notes
        generic_phrases = [
            "peak tourist season for Srirangapatna",
            "Srirangapatna's famous Ranganathaswamy Temple",
            "peak tourist season",
        ]
        if ctx and any(phrase in ctx for phrase in generic_phrases):
            ctx_pool = _DISTRICT_CONTEXT_MAP.get(district, _DEFAULT_CONTEXT)
            narrative["contextual_notes"] = random.choice(ctx_pool)

        variant["narrative"] = narrative

    # ── Rule 5a: Crime-Evidence Alignment — Property Crimes ───────────────
    if crime_type in _PROPERTY_CRIMES:
        if (variant.get("estimated_value_inr") or 0) <= 0:
            variant["estimated_value_inr"] = random.randint(5_000, 5_00_000)
        # injury_type should be null for pure property crimes (unless struggle mentioned)
        if crime_type in {"Cyber Fraud"}:
            variant["injury_type"] = None

    # ── Rule 5b: Crime-Evidence Alignment — Violent Crimes ────────────────
    if crime_type in _VIOLENT_CRIMES:
        inj = variant.get("injury_type")
        if not inj or inj not in _VALID_INJURIES:
            if crime_type == "Murder":
                variant["injury_type"] = "fatal injury"
            else:
                variant["injury_type"] = random.choice(["simple injury", "grievous injury", "fracture"])

    # ── Rule 5c: Crime-Evidence Alignment — Cyber Crimes ──────────────────
    if crime_type == "Cyber Fraud":
        variant["injury_type"] = None
        # Ensure evidence is digital
        ev = narrative.get("evidence", {}) if narrative else {}
        digital = ev.get("digital_traces", [])
        if not digital:
            ev["digital_traces"] = [
                "UPI transaction reference ID",
                "call detail records from spoofed number",
                "IP address logs from payment gateway",
            ]
            if narrative:
                narrative["evidence"] = ev
                variant["narrative"] = narrative

    # ── Rule 6b: Null enforcement on evidence arrays ──────────────────────
    if narrative:
        ev = narrative.get("evidence", {})
        for key in ("physical_items", "digital_traces", "witness_clues"):
            val = ev.get(key)
            if val is None or val == "null":
                # Use crime library fallback
                lib = CRIME_LIBRARY.get(crime_type, CRIME_LIBRARY["Theft"])
                ev[key] = lib["evidence"].get(key, [])
        narrative["evidence"] = ev
        variant["narrative"] = narrative

    return variant


def run_augmentation_pipeline(
    seeds_path: str | Path,
    output_path: str | Path,
    target_count: int = 330_000,
    start_date: date = date(2010, 1, 1),
    end_date: date = date(2026, 6, 4),
    rng_seed: int = 42,
) -> int:
    """
    Phase 2 — Augmentation Engine with year-weighted temporal distribution.

    Reads all seed records from `seeds_path` (JSON Lines), then generates
    `target_count` fully validated FIRModel records by applying the
    `augment_seed_record` mutation engine and streaming output to `output_path`.

    Temporal distribution:
    ─────────────────────
    FIR dates are sampled using `year_weighted_datetime`, which biases the
    filing date toward more recent years using a ~10% per-year growth rate.
    This produces a realistic dataset where 2010 has fewer cases than 2026,
    matching actual Karnataka crime reporting trends.

    Example approximate record distribution (target=330,000):
      2010: ~8,300    2015: ~13,400   2020: ~21,500   2025: ~34,600
      (actual counts vary due to randomness within each year)

    Returns the total number of records written.
    """
    random.seed(rng_seed)
    seeds_path = Path(seeds_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load seeds (small file: 2,500–5,000 records) ─────────────────────
    seeds: list[dict[str, Any]] = []
    with seeds_path.open("r", encoding="utf-8") as sf:
        for line in sf:
            line = line.strip()
            if line:
                try:
                    seeds.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not seeds:
        raise ValueError(f"No valid seeds found in {seeds_path}")

    seed_count = len(seeds)
    variants_per_seed = max(1, math.ceil(target_count / seed_count))
    log.info(
        "Augmentation: %d seeds × %d variants = %d records → target %d",
        seed_count, variants_per_seed, seed_count * variants_per_seed, target_count,
    )

    # ── Build year-weighted distribution for filing-date sampling ──────────
    # Growth: ~10% more cases per year from start_date.year to end_date.year.
    yw = build_year_weights(start_date.year, end_date.year)
    log.info(
        "Year-weight distribution (first=%.2f, last=%.2f, ratio=%.1fx)",
        yw[start_date.year], yw[end_date.year],
        yw[end_date.year] / yw[start_date.year],
    )

    districts = list(KarnatakaDistrict)

    # ── Count existing records so fir_id continues from the last written ID ──
    existing_count = 0
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as _cf:
            for _line in _cf:
                if _line.strip():
                    existing_count += 1
    if existing_count:
        log.info("Append mode: %d records already exist → fir_id will start at %d", existing_count, existing_count + 1)

    fir_id = existing_count
    total_written = 0

    # Append mode — existing FIR records are preserved; new records added below
    with output_path.open("a", encoding="utf-8") as out_fh:
        for seed_dict in seeds:
            if total_written >= target_count:
                break

            for variant_idx in range(variants_per_seed):
                if total_written >= target_count:
                    break

                fir_id += 1

                # Sample a year-biased filing date (more recent years more likely)
                filed_at = year_weighted_datetime(start_date, end_date, yw)

                # Rotate district round-robin across all 31 districts
                district_override = districts[(fir_id - 1) % len(districts)]

                # Apply field-level mutations to produce a unique variant
                variant_seed = augment_seed_record(
                    seed_dict,
                    variant_index=variant_idx,
                    target_date=filed_at,
                    district_override=district_override,
                )

                # Enforce all guardrail rules (gender-prefix, geo, crime-evidence, nulls)
                variant_seed = sanitize_augmented_seed(variant_seed)

                try:
                    fir = build_fir_from_seed_payload(
                        variant_seed, fir_id=fir_id, filed_at=filed_at
                    )
                except (ValidationError, ValueError, KeyError) as exc:
                    log.warning(
                        "Skipping variant %d (FIR %d): %s", variant_idx, fir_id, exc
                    )
                    continue

                # Stream directly to disk — O(1) memory footprint
                out_fh.write(
                    json.dumps(
                        fir.model_dump(mode="json"),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                total_written += 1

                if total_written % 10_000 == 0:
                    out_fh.flush()
                    log.info(
                        "Augmentation progress: %d / %d records written.",
                        total_written, target_count,
                    )

    log.info("Augmentation complete. Total records: %d → %s", total_written, output_path)
    return total_written


# =============================================================================
# SECTION 6: DEMO VALIDATION BLOCK
# =============================================================================

def demo_validation_and_pivot() -> None:
    """
    Run an explicit demonstration of the pre-pivot (IPC) and post-pivot (BNS)
    legal regime validation. Asserts prove the pivot engine is working correctly.
    Also demonstrates the Karnataka-specific name and location generation.
    """
    print("\n" + "=" * 70)
    print("DEMO: Karnataka FIR Hybrid Generator — Validation & Pivot Proof")
    print("=" * 70)

    # ── Pre-pivot record: 2018, Murder, Mysuru ─────────────────────────────
    pre_seed = generate_seed_payload("Murder", KarnatakaDistrict.MYSURU)
    pre_seed["complainant_name"] = "Siddaramaiah Gowda"
    pre_seed["guardian_name"] = "S/O Late Munirathnappa Gowda"
    pre_seed["victim_name"] = "Basavaraj Gowda"
    pre_seed["victim_age"] = 34
    pre_seed["accused_known"] = True
    pre_seed["accused_name"] = "Veeranna Nayak"

    pre_fir = build_fir_from_seed_payload(pre_seed, fir_id=1, filed_at=datetime(2018, 5, 14, 10, 30))

    print("\n[PRE-PIVOT] FIR dated 2018-05-14 (Murder / Mysuru)")
    print(f"  FIR Number      : {pre_fir.fir_number}")
    print(f"  Station         : {pre_fir.station.name}, {pre_fir.station.district.value}")
    print(f"  Act / Section   : {pre_fir.charges[0].act.value} / Section {pre_fir.charges[0].section}")
    print(f"  Complainant     : {pre_fir.people.complainant.name} ({pre_fir.people.complainant.guardian_name})")
    print(f"  Victim          : {pre_fir.people.victims[0].name}, age {pre_fir.people.victims[0].age}")
    print(f"  Accused         : {pre_fir.people.accused[0].name} (Known: {pre_fir.people.accused[0].is_known})")
    print(f"  Location        : {pre_fir.location.text}")
    print(f"  IO              : {pre_fir.investigation.io_name}")
    print(f"  Status          : {pre_fir.status.value}")
    print(f"  Delay           : {pre_fir.timeline.delay_in_reporting_hours} hrs")

    # Assertion: pre-pivot must use IPC
    assert pre_fir.charges[0].act == LegalActEnum.IPC, "FAIL: Pre-pivot charge must be IPC"
    assert pre_fir.charges[0].section == "302", f"FAIL: Murder section must be 302, got {pre_fir.charges[0].section}"
    print("  [PASS] ASSERT: IPC / 302 confirmed for pre-pivot Murder FIR")

    # Round-trip validation
    pre_rt = FIRModel.model_validate(pre_fir.model_dump(mode="json"))
    assert pre_rt.charges[0].act == LegalActEnum.IPC
    print("  [PASS] ASSERT: Round-trip Pydantic validation passed")

    # ── Post-pivot record: 2025, Theft, Bengaluru Urban ────────────────────
    post_seed = generate_seed_payload("Theft", KarnatakaDistrict.BENGALURU_URBAN)
    post_seed["complainant_name"] = "Ananya Hegde"
    post_seed["guardian_name"] = "D/O Srinivasa Hegde"
    post_seed["complainant_gender"] = "Female"
    post_seed["victim_name"] = "Ananya Hegde"
    post_seed["accused_known"] = True
    post_seed["accused_name"] = "Rakesh Naik"
    post_seed["estimated_value_inr"] = 125_000

    post_fir = build_fir_from_seed_payload(post_seed, fir_id=2, filed_at=datetime(2025, 3, 21, 15, 45))

    print("\n[POST-PIVOT] FIR dated 2025-03-21 (Theft / Bengaluru Urban)")
    print(f"  FIR Number      : {post_fir.fir_number}")
    print(f"  Station         : {post_fir.station.name}, {post_fir.station.district.value}")
    print(f"  Act / Section   : {post_fir.charges[0].act.value} / Section {post_fir.charges[0].section}")
    print(f"  Complainant     : {post_fir.people.complainant.name} ({post_fir.people.complainant.guardian_name})")
    print(f"  Victim          : {post_fir.people.victims[0].name}, age {post_fir.people.victims[0].age}")
    print(f"  Accused         : {post_fir.people.accused[0].name} (Known: {post_fir.people.accused[0].is_known})")
    print(f"  Property Value  : INR {post_fir.property_stolen.estimated_value_inr:,}")
    print(f"  Location        : {post_fir.location.text}")
    print(f"  IO              : {post_fir.investigation.io_name}")
    print(f"  Status          : {post_fir.status.value}")

    # Assertion: post-pivot must use BNS
    assert post_fir.charges[0].act == LegalActEnum.BNS, "FAIL: Post-pivot charge must be BNS"
    assert post_fir.charges[0].section == "303(2)", f"FAIL: Theft section must be 303(2), got {post_fir.charges[0].section}"
    print("  [PASS] ASSERT: BNS / 303(2) confirmed for post-pivot Theft FIR")

    post_rt = FIRModel.model_validate(post_fir.model_dump(mode="json"))
    assert post_rt.charges[0].act == LegalActEnum.BNS
    print("  [PASS] ASSERT: Round-trip Pydantic validation passed")

    # ── IT Act override: Cyber Fraud (any date → IT Act 2000) ─────────────
    cyber_seed = generate_seed_payload("Cyber Fraud", KarnatakaDistrict.DAKSHINA_KANNADA)
    cyber_fir = build_fir_from_seed_payload(cyber_seed, fir_id=3, filed_at=datetime(2023, 8, 10, 9, 0))

    print("\n[IT ACT OVERRIDE] FIR dated 2023-08-10 (Cyber Fraud / Dakshina Kannada)")
    print(f"  Act / Section   : {cyber_fir.charges[0].act.value} / Section {cyber_fir.charges[0].section}")
    assert cyber_fir.charges[0].act == LegalActEnum.IT_Act_2000, "FAIL: Cyber Fraud must use IT Act 2000"
    print("  [PASS] ASSERT: IT Act 2000 override confirmed for Cyber Fraud (pre-pivot date)")

    print("\n" + "=" * 70)
    print("[ALL PASS] All assertions passed. Generator is production-ready.")
    print("=" * 70 + "\n")


# =============================================================================
# SECTION 7: CLI ENTRYPOINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Karnataka FIR Hybrid Generator — Two-phase synthetic data pipeline.\n\n"
            "Modes:\n"
            "  demo    — Run validation proof (no LLM needed)\n"
            "  seed    — Phase 1: Ollama LLM generates narrative seeds\n"
            "  augment — Phase 2: Augment seeds to target record count\n"
            "  full    — Run seed + augment in sequence\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "seed", "augment", "full"],
        default="demo",
        help="Execution mode (default: demo)",
    )
    parser.add_argument(
        "--seeds-file",
        type=str,
        default="llm_seeds.jsonl",
        help="Path to the JSONL file where LLM seeds are stored/read.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="karnataka_fir_330k.jsonl",
        help="Path to the final JSONL output file.",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=500,
        help="Number of Ollama LLM invocations for seed phase (default: 500).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=7,
        help="Records requested per Ollama LLM invoke (default: 7, range: 5-10).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1:8b",
        help=(
            "Ollama model name to use via LangChain ChatOllama (default: llama3.1:8b). "
            "Examples: llama3.1:8b, mistral, phi3, gemma2. "
            "Run `ollama pull <model>` first if not already downloaded."
        ),
    )
    parser.add_argument(
        "--target",
        type=int,
        default=330_000,
        help="Target total record count for augmentation (default: 330000).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2010-01-01",
        help="Inclusive start date for generated FIR dates (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2026-06-04",
        help="Inclusive end date for generated FIR dates (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip individual record errors instead of aborting (seed phase LLM fallback).",
    )

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    if args.mode == "demo":
        # ── Demo mode: validation proof only, no file I/O needed ────────────
        demo_validation_and_pivot()

    elif args.mode in {"seed", "full"}:
        # ── Seed phase: LLM generates narrative seeds via ChatOllama ─────────
        demo_validation_and_pivot()
        batch_size = max(5, min(10, args.batch_size))  # Clamp to valid range 5–10
        total_seeds = run_llm_seed_loop(
            output_path=args.seeds_file,
            loops=args.loops,
            batch_size=batch_size,
            model=args.model,
            fallback_on_error=args.skip_errors,
            rng_seed=args.seed,
        )
        print(f"\n[OK] Seed phase complete. {total_seeds} seeds → {args.seeds_file}")
        if args.mode == "seed":
            return
        # Fall through to augment phase for "full" mode

    if args.mode in {"augment", "full"}:
        # ── Augment phase: scale seeds to target count ───────────────────────
        total_records = run_augmentation_pipeline(
            seeds_path=args.seeds_file,
            output_path=args.output,
            target_count=args.target,
            start_date=start_date,
            end_date=end_date,
            rng_seed=args.seed,
        )
        print(f"\n[OK] Augmentation complete. {total_records} records → {args.output}")


if __name__ == "__main__":
    main()
