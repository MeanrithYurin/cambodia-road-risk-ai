import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap

st.set_page_config(
    page_title="Cambodia Road Accident Risk AI",
    layout="wide"
)

DATA_PATH = "data/raw/cambodia_road_accident_large_dataset.csv"
MODEL_PATH = "models/best_risk_model.pkl"

st.sidebar.title("Menu")
language = st.sidebar.selectbox("Language / ភាសា", ["English", "ខ្មែរ"])

TEXT = {
    "English": {
        "title": "Cambodia Road Accident Risk Prediction System",
        "subtitle": "Machine Learning + Cambodia Map Dashboard + Khmer Safety Warning",
        "total_records": "Total Records",
        "total_accidents": "Total Accidents",
        "total_injuries": "Total Injuries",
        "total_fatalities": "Total Fatalities",
        "prediction": "Risk Prediction",
        "dashboard": "Dashboard",
        "map": "Accident Risk Map",
        "province": "Province",
        "vehicle": "Vehicle Type",
        "weather": "Weather",
        "road_type": "Road Type",
        "road_condition": "Road Condition",
        "lighting": "Lighting",
        "age": "Driver Age Group",
        "helmet": "Helmet Used",
        "seatbelt": "Seatbelt Used",
        "speeding": "Speeding",
        "alcohol": "Alcohol Related",
        "holiday": "Holiday",
        "hour": "Hour",
        "predict": "Predict Risk",
        "risk_result": "Predicted Risk Level",
        "warning_low": "Low risk. Please still follow traffic laws and drive carefully.",
        "warning_medium": "Medium risk. Please reduce speed and keep safe distance.",
        "warning_high": "High risk. Please be very careful, reduce speed, and avoid risky travel if possible.",
    },
    "ខ្មែរ": {
        "title": "ប្រព័ន្ធព្យាករណ៍ហានិភ័យគ្រោះថ្នាក់ចរាចរណ៍នៅកម្ពុជា",
        "subtitle": "Machine Learning + ផែនទីកម្ពុជា + សារព្រមានជាភាសាខ្មែរ",
        "total_records": "ចំនួនទិន្នន័យសរុប",
        "total_accidents": "ចំនួនគ្រោះថ្នាក់សរុប",
        "total_injuries": "ចំនួនអ្នករងរបួស",
        "total_fatalities": "ចំនួនអ្នកស្លាប់",
        "prediction": "ព្យាករណ៍ហានិភ័យ",
        "dashboard": "ផ្ទាំងវិភាគទិន្នន័យ",
        "map": "ផែនទីហានិភ័យគ្រោះថ្នាក់",
        "province": "ខេត្ត/រាជធានី",
        "vehicle": "ប្រភេទយានយន្ត",
        "weather": "អាកាសធាតុ",
        "road_type": "ប្រភេទផ្លូវ",
        "road_condition": "ស្ថានភាពផ្លូវ",
        "lighting": "ពន្លឺ",
        "age": "ក្រុមអាយុអ្នកបើកបរ",
        "helmet": "ការពាក់មួកសុវត្ថិភាព",
        "seatbelt": "ការពាក់ខ្សែក្រវ៉ាត់",
        "speeding": "ការបើកបរលើសល្បឿន",
        "alcohol": "ពាក់ព័ន្ធនឹងគ្រឿងស្រវឹង",
        "holiday": "ថ្ងៃឈប់សម្រាក",
        "hour": "ម៉ោង",
        "predict": "ព្យាករណ៍",
        "risk_result": "កម្រិតហានិភ័យដែលបានព្យាករណ៍",
        "warning_low": "ហានិភ័យទាប។ សូមគោរពច្បាប់ចរាចរណ៍ និងបើកបរដោយប្រុងប្រយ័ត្ន។",
        "warning_medium": "ហានិភ័យមធ្យម។ សូមបន្ថយល្បឿន និងរក្សាគម្លាតសុវត្ថិភាព។",
        "warning_high": "ហានិភ័យខ្ពស់។ សូមប្រុងប្រយ័ត្នខ្លាំង បន្ថយល្បឿន និងជៀសវាងការធ្វើដំណើរបើមិនចាំបាច់។",
    }
}

T = TEXT[language]

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.title(T["title"])
st.write(T["subtitle"])

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Please run: python train_model.py")
    st.stop()

model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
label_encoder = model_data["label_encoder"]
features = model_data["features"]

tab1, tab2, tab3 = st.tabs([
    T["dashboard"],
    T["prediction"],
    T["map"]
])

with tab1:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(T["total_records"], f"{len(df):,}")
    col2.metric(T["total_accidents"], f"{df['accident_count'].sum():,}")
    col3.metric(T["total_injuries"], f"{df['injuries'].sum():,}")
    col4.metric(T["total_fatalities"], f"{df['fatalities'].sum():,}")

    st.subheader(T["province"] + " Risk Summary")

    province_summary = df.groupby("province_en", as_index=False).agg(
        total_accidents=("accident_count", "sum"),
        total_injuries=("injuries", "sum"),
        total_fatalities=("fatalities", "sum"),
        avg_risk_score=("risk_score", "mean")
    )

    province_summary["avg_risk_score"] = province_summary["avg_risk_score"].round(2)

    fig1 = px.bar(
        province_summary.sort_values("total_accidents", ascending=False),
        x="province_en",
        y="total_accidents",
        title="Total Accidents by Province"
    )
    st.plotly_chart(fig1, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        fig2 = px.pie(
            df,
            names="vehicle_type_en",
            title="Accidents by Vehicle Type"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        fig3 = px.histogram(
            df,
            x="hour",
            color="risk_level_en",
            title="Risk Level by Hour"
        )
        st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.line(
        df.groupby(["year", "month"], as_index=False)["accident_count"].sum(),
        x="month",
        y="accident_count",
        color="year",
        title="Monthly Accident Trend"
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.dataframe(province_summary, use_container_width=True)

with tab2:
    st.subheader(T["prediction"])

    col1, col2 = st.columns(2)

    with col1:
        province = st.selectbox(T["province"], sorted(df["province_en"].unique()))
        vehicle = st.selectbox(T["vehicle"], sorted(df["vehicle_type_en"].unique()))
        weather = st.selectbox(T["weather"], sorted(df["weather_en"].unique()))
        road_type = st.selectbox(T["road_type"], sorted(df["road_type_en"].unique()))
        road_condition = st.selectbox(T["road_condition"], sorted(df["road_condition_en"].unique()))
        hour = st.slider(T["hour"], 0, 23, 18)

    with col2:
        lighting = st.selectbox(T["lighting"], sorted(df["lighting_en"].unique()))
        age = st.selectbox(T["age"], sorted(df["driver_age_group_en"].unique()))
        helmet_used = st.selectbox(T["helmet"], sorted(df["helmet_used_en"].unique()))
        seatbelt_used = st.selectbox(T["seatbelt"], sorted(df["seatbelt_used_en"].unique()))
        speeding = st.selectbox(T["speeding"], sorted(df["speeding_en"].unique()))
        alcohol_related = st.selectbox(T["alcohol"], sorted(df["alcohol_related_en"].unique()))
        holiday = st.selectbox(T["holiday"], sorted(df["holiday_en"].unique()))
        day_of_week = st.selectbox("Day of Week / ថ្ងៃ", sorted(df["day_of_week"].unique()))

    if 18 <= hour <= 23 or 0 <= hour <= 5:
        time_period = "Night"
    elif 6 <= hour <= 9:
        time_period = "Morning Rush"
    elif 16 <= hour <= 18:
        time_period = "Evening Rush"
    else:
        time_period = "Day"

    input_data = pd.DataFrame([{
        "province_en": province,
        "day_of_week": day_of_week,
        "hour": hour,
        "time_period_en": time_period,
        "vehicle_type_en": vehicle,
        "weather_en": weather,
        "road_type_en": road_type,
        "road_condition_en": road_condition,
        "lighting_en": lighting,
        "driver_age_group_en": age,
        "helmet_used_en": helmet_used,
        "seatbelt_used_en": seatbelt_used,
        "speeding_en": speeding,
        "alcohol_related_en": alcohol_related,
        "holiday_en": holiday
    }])

    if st.button(T["predict"]):
        pred = model.predict(input_data)[0]
        risk_en = label_encoder.inverse_transform([pred])[0]

        risk_km_map = {
            "Low": "ទាប",
            "Medium": "មធ្យម",
            "High": "ខ្ពស់"
        }

        risk_km = risk_km_map.get(risk_en, risk_en)

        if risk_en == "Low":
            st.success(f"{T['risk_result']}: {risk_en} / {risk_km}")
            st.info(T["warning_low"])
        elif risk_en == "Medium":
            st.warning(f"{T['risk_result']}: {risk_en} / {risk_km}")
            st.info(T["warning_medium"])
        else:
            st.error(f"{T['risk_result']}: {risk_en} / {risk_km}")
            st.info(T["warning_high"])

with tab3:
    st.subheader(T["map"])

    map_type = st.radio(
        "Map Type / ប្រភេទផែនទី",
        ["Heatmap", "Marker Map"],
        horizontal=True
    )

    sample_df = df.sample(min(3000, len(df)), random_state=42)

    m = folium.Map(
        location=[12.5657, 104.9910],
        zoom_start=7,
        tiles="OpenStreetMap"
    )

    if map_type == "Heatmap":
        heat_data = sample_df[["latitude", "longitude", "risk_score"]].values.tolist()
        HeatMap(
            heat_data,
            radius=12,
            blur=15,
            max_zoom=10
        ).add_to(m)
    else:
        high_risk_df = sample_df[sample_df["risk_level_en"] == "High"].head(500)

        for _, row in high_risk_df.iterrows():
            popup_text = f"""
            <b>{row['province_en']} / {row['province_km']}</b><br>
            Vehicle: {row['vehicle_type_en']} / {row['vehicle_type_km']}<br>
            Weather: {row['weather_en']} / {row['weather_km']}<br>
            Risk: {row['risk_level_en']} / {row['risk_level_km']}<br>
            Score: {row['risk_score']}
            """

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=5,
                popup=popup_text,
                fill=True
            ).add_to(m)

    st_folium(m, width=1200, height=600)

    st.caption("Map uses sample points from the dataset for faster loading.")