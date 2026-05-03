import os
import random
from datetime import datetime, timedelta
import pandas as pd

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

random.seed(42)

provinces = [
    ("Phnom Penh", "ភ្នំពេញ", 11.5564, 104.9282, 1.35),
    ("Kandal", "កណ្ដាល", 11.4833, 104.9500, 1.15),
    ("Kampong Speu", "កំពង់ស្ពឺ", 11.4533, 104.5200, 1.10),
    ("Takeo", "តាកែវ", 10.9900, 104.7850, 1.00),
    ("Kampot", "កំពត", 10.6100, 104.1800, 0.95),
    ("Kep", "កែប", 10.4829, 104.3167, 0.75),
    ("Preah Sihanouk", "ព្រះសីហនុ", 10.6253, 103.5234, 1.25),
    ("Koh Kong", "កោះកុង", 11.6175, 102.9849, 0.85),
    ("Battambang", "បាត់ដំបង", 13.0957, 103.2022, 1.10),
    ("Banteay Meanchey", "បន្ទាយមានជ័យ", 13.6673, 102.8975, 1.05),
    ("Siem Reap", "សៀមរាប", 13.3671, 103.8448, 1.20),
    ("Oddar Meanchey", "ឧត្តរមានជ័យ", 14.1600, 103.5000, 0.80),
    ("Preah Vihear", "ព្រះវិហារ", 13.8070, 104.9800, 0.80),
    ("Kampong Thom", "កំពង់ធំ", 12.7111, 104.8887, 0.95),
    ("Kampong Cham", "កំពង់ចាម", 12.0000, 105.4500, 1.10),
    ("Tboung Khmum", "ត្បូងឃ្មុំ", 11.8900, 105.6500, 0.90),
    ("Kratie", "ក្រចេះ", 12.4881, 106.0188, 0.85),
    ("Stung Treng", "ស្ទឹងត្រែង", 13.5259, 105.9683, 0.75),
    ("Ratanakiri", "រតនគិរី", 13.7394, 106.9873, 0.70),
    ("Mondulkiri", "មណ្ឌលគិរី", 12.4558, 107.1881, 0.70),
    ("Pursat", "ពោធិ៍សាត់", 12.5388, 103.9192, 0.90),
    ("Kampong Chhnang", "កំពង់ឆ្នាំង", 12.2500, 104.6667, 0.90),
    ("Prey Veng", "ព្រៃវែង", 11.4868, 105.3253, 1.00),
    ("Svay Rieng", "ស្វាយរៀង", 11.0879, 105.7994, 0.95),
    ("Pailin", "ប៉ៃលិន", 12.8489, 102.6093, 0.75)
]

districts = {
    "Phnom Penh": ["Chamkarmon", "Daun Penh", "Toul Kork", "Sen Sok", "Meanchey", "Chbar Ampov"],
    "Kandal": ["Takhmao", "Ang Snuol", "Kien Svay", "Khsach Kandal"],
    "Kampong Speu": ["Chbar Mon", "Samraong Tong", "Phnom Sruoch"],
    "Takeo": ["Doun Kaev", "Bati", "Tram Kak"],
    "Kampot": ["Kampot", "Chhuk", "Angkor Chey"],
    "Kep": ["Kep", "Damnak Chang'aeur"],
    "Preah Sihanouk": ["Sihanoukville", "Prey Nob", "Stung Hav"],
    "Koh Kong": ["Khemarak Phoumin", "Sre Ambel", "Mondol Seima"],
    "Battambang": ["Battambang", "Bavel", "Sangkae", "Thma Koul"],
    "Banteay Meanchey": ["Sisophon", "Poipet", "Mongkol Borei"],
    "Siem Reap": ["Siem Reap", "Prasat Bakong", "Puok"],
    "Oddar Meanchey": ["Samraong", "Anlong Veng"],
    "Preah Vihear": ["Preah Vihear", "Tbaeng Meanchey"],
    "Kampong Thom": ["Stung Sen", "Baray", "Kampong Svay"],
    "Kampong Cham": ["Kampong Cham", "Cheung Prey", "Batheay"],
    "Tboung Khmum": ["Suong", "Tboung Khmum", "Memot"],
    "Kratie": ["Kratie", "Chhloung", "Snuol"],
    "Stung Treng": ["Stung Treng", "Siem Pang"],
    "Ratanakiri": ["Banlung", "O'Yadav"],
    "Mondulkiri": ["Sen Monorom", "Keo Seima"],
    "Pursat": ["Pursat", "Krakor", "Veal Veng"],
    "Kampong Chhnang": ["Kampong Chhnang", "Rolea B'ier"],
    "Prey Veng": ["Prey Veng", "Neak Loeung", "Peam Ro"],
    "Svay Rieng": ["Svay Rieng", "Bavet", "Romeas Haek"],
    "Pailin": ["Pailin", "Sala Krau"]
}

vehicle_types = [
    ("Motorbike", "ម៉ូតូ", 1.35),
    ("Car", "រថយន្ត", 0.95),
    ("Truck", "រថយន្តដឹកទំនិញ", 1.25),
    ("Bus", "រថយន្តក្រុង", 1.10),
    ("Tuk Tuk", "តុកតុក", 0.85),
    ("Bicycle", "កង់", 0.75),
    ("Van", "រថយន្តវ៉ែន", 1.00)
]

weather_types = [
    ("Clear", "អាកាសធាតុល្អ", 0.85),
    ("Rainy", "ភ្លៀង", 1.35),
    ("Foggy", "អ័ព្ទ", 1.25),
    ("Storm", "ព្យុះ/ភ្លៀងខ្លាំង", 1.55),
    ("Cloudy", "មេឃពពក", 1.00)
]

road_types = [
    ("Urban Road", "ផ្លូវក្នុងក្រុង", 1.05),
    ("National Road", "ផ្លូវជាតិ", 1.30),
    ("Provincial Road", "ផ្លូវខេត្ត", 1.10),
    ("Rural Road", "ផ្លូវជនបទ", 0.95),
    ("Highway", "ផ្លូវល្បឿនលឿន", 1.25),
    ("Market Area Road", "ផ្លូវតំបន់ផ្សារ", 1.15),
    ("School Zone Road", "ផ្លូវតំបន់សាលារៀន", 1.00)
]

road_conditions = [
    ("Dry", "ស្ងួត", 0.90),
    ("Wet", "សើម", 1.25),
    ("Damaged", "ខូចខាត", 1.35),
    ("Under Construction", "កំពុងសាងសង់", 1.30),
    ("Poor Lighting", "ពន្លឺមិនគ្រប់គ្រាន់", 1.40)
]

lighting = [
    ("Daylight", "ពេលថ្ងៃ", 0.85),
    ("Night with lights", "យប់មានភ្លើង", 1.10),
    ("Night poor lights", "យប់ភ្លើងមិនគ្រប់គ្រាន់", 1.45)
]

driver_age_groups = [
    ("Under 18", "ក្រោម ១៨", 1.25),
    ("18-25", "១៨-២៥", 1.30),
    ("26-40", "២៦-៤០", 1.00),
    ("41-60", "៤១-៦០", 0.95),
    ("Above 60", "លើស ៦០", 1.15)
]

helmet = [
    ("Yes", "ពាក់", 0.85),
    ("No", "មិនពាក់", 1.35),
    ("Unknown", "មិនដឹង", 1.00)
]

seatbelt = [
    ("Yes", "ពាក់", 0.85),
    ("No", "មិនពាក់", 1.25),
    ("Unknown", "មិនដឹង", 1.00)
]

speeding = [
    ("Yes", "បើកបរលើសល្បឿន", 1.60),
    ("No", "មិនលើសល្បឿន", 0.85)
]

alcohol = [
    ("Yes", "មានជាតិអាល់កុល", 1.65),
    ("No", "មិនមាន", 0.90)
]

holiday = [
    ("Yes", "ថ្ងៃឈប់សម្រាក", 1.25),
    ("No", "ថ្ងៃធម្មតា", 0.95)
]


def risk_label(score):
    if score < 7:
        return "Low", "ទាប"
    elif score < 15:
        return "Medium", "មធ្យម"
    else:
        return "High", "ខ្ពស់"


def severity_label(fatalities, injuries):
    if fatalities >= 2 or injuries >= 8:
        return "Severe", "ធ្ងន់ធ្ងរ"
    elif fatalities == 1 or injuries >= 3:
        return "Moderate", "មធ្យម"
    else:
        return "Minor", "ស្រាល"


rows = []
n = 25000
start_date = datetime(2021, 1, 1)

for i in range(1, n + 1):
    prov_en, prov_km, lat0, lon0, prov_factor = random.choice(provinces)
    dist = random.choice(districts.get(prov_en, [prov_en]))

    vehicle_en, vehicle_km, veh_factor = random.choice(vehicle_types)
    weather_en, weather_km, weather_factor = random.choice(weather_types)
    road_en, road_km, road_factor = random.choice(road_types)
    cond_en, cond_km, cond_factor = random.choice(road_conditions)
    light_en, light_km, light_factor = random.choice(lighting)
    age_en, age_km, age_factor = random.choice(driver_age_groups)
    helmet_en, helmet_km, helmet_factor = random.choice(helmet)
    seatbelt_en, seatbelt_km, seatbelt_factor = random.choice(seatbelt)
    speed_en, speed_km, speed_factor = random.choice(speeding)
    alcohol_en, alcohol_km, alcohol_factor = random.choice(alcohol)
    hol_en, hol_km, hol_factor = random.choice(holiday)

    dt = start_date + timedelta(
        days=random.randint(0, 4 * 365),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )

    hour = dt.hour

    if 18 <= hour <= 23 or 0 <= hour <= 5:
        time_factor = 1.35
        time_period_en, time_period_km = "Night", "ពេលយប់"
    elif 6 <= hour <= 9:
        time_factor = 1.15
        time_period_en, time_period_km = "Morning Rush", "ម៉ោងមមាញឹកព្រឹក"
    elif 16 <= hour <= 18:
        time_factor = 1.20
        time_period_en, time_period_km = "Evening Rush", "ម៉ោងមមាញឹកល្ងាច"
    else:
        time_factor = 0.90
        time_period_en, time_period_km = "Day", "ពេលថ្ងៃ"

    weekend_factor = 1.15 if dt.weekday() >= 5 else 0.95

    lat = lat0 + random.uniform(-0.22, 0.22)
    lon = lon0 + random.uniform(-0.22, 0.22)

    base_risk = (
        prov_factor * veh_factor * weather_factor * road_factor *
        cond_factor * light_factor * age_factor * helmet_factor *
        seatbelt_factor * speed_factor * alcohol_factor * hol_factor *
        time_factor * weekend_factor
    )

    accident_count = max(1, int(random.gauss(base_risk * 2.2, 1.2)))
    injuries = max(0, int(random.gauss(base_risk * 1.6, 1.5)))

    fatality_prob = min(0.55, max(0.01, (base_risk - 0.8) / 8))
    fatalities = 0

    if random.random() < fatality_prob:
        fatalities = 1

    if random.random() < fatality_prob * 0.18:
        fatalities += 1

    risk_score = round(
        accident_count * 1 +
        injuries * 2 +
        fatalities * 5 +
        base_risk * 2,
        2
    )

    r_en, r_km = risk_label(risk_score)
    sev_en, sev_km = severity_label(fatalities, injuries)

    rows.append({
        "record_id": f"CAM-RC-{i:05d}",
        "date": dt.strftime("%Y-%m-%d"),
        "year": dt.year,
        "month": dt.month,
        "day_of_week": dt.strftime("%A"),
        "hour": hour,
        "time_period_en": time_period_en,
        "time_period_km": time_period_km,
        "province_en": prov_en,
        "province_km": prov_km,
        "district_en": dist,
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "vehicle_type_en": vehicle_en,
        "vehicle_type_km": vehicle_km,
        "weather_en": weather_en,
        "weather_km": weather_km,
        "road_type_en": road_en,
        "road_type_km": road_km,
        "road_condition_en": cond_en,
        "road_condition_km": cond_km,
        "lighting_en": light_en,
        "lighting_km": light_km,
        "driver_age_group_en": age_en,
        "driver_age_group_km": age_km,
        "helmet_used_en": helmet_en,
        "helmet_used_km": helmet_km,
        "seatbelt_used_en": seatbelt_en,
        "seatbelt_used_km": seatbelt_km,
        "speeding_en": speed_en,
        "speeding_km": speed_km,
        "alcohol_related_en": alcohol_en,
        "alcohol_related_km": alcohol_km,
        "holiday_en": hol_en,
        "holiday_km": hol_km,
        "accident_count": accident_count,
        "injuries": injuries,
        "fatalities": fatalities,
        "risk_score": risk_score,
        "risk_level_en": r_en,
        "risk_level_km": r_km,
        "severity_en": sev_en,
        "severity_km": sev_km
    })

df = pd.DataFrame(rows)

df.to_csv(
    "data/raw/cambodia_road_accident_large_dataset.csv",
    index=False,
    encoding="utf-8-sig"
)

summary = df.groupby(["province_en", "province_km"], as_index=False).agg(
    total_records=("record_id", "count"),
    total_accidents=("accident_count", "sum"),
    total_injuries=("injuries", "sum"),
    total_fatalities=("fatalities", "sum"),
    avg_risk_score=("risk_score", "mean")
)

summary["avg_risk_score"] = summary["avg_risk_score"].round(2)

summary.to_csv(
    "data/processed/province_risk_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Dataset generated successfully.")
print("Rows:", len(df))
print("Saved to data/raw/cambodia_road_accident_large_dataset.csv")
print(df.head())