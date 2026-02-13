import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from io import BytesIO

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Delivery Intelligence",
    layout="wide",
    page_icon="🚀"
)

st.title("🚀 Smart Delivery Intelligence System")
st.markdown("AI-powered ETA Prediction • Risk Assessment • Operational Optimization")

# =====================================================
# LOAD MODEL
# =====================================================
model = joblib.load("../models/model.pkl")
model_columns = joblib.load("../models/columns.pkl")

# =====================================================
# SIDEBAR INPUTS
# =====================================================
st.sidebar.header("📦 Delivery Inputs")

city = st.sidebar.selectbox(
    "Select City",
    ["BANG", "CHEN", "HYD", "MYS", "INDO"]
)

age = st.sidebar.slider("Driver Age", 18, 60, 25)
rating = st.sidebar.slider("Driver Rating", 1.0, 5.0, 4.5)
distance = st.sidebar.slider("Delivery Distance (km)", 0.5, 30.0, 5.0)

vehicle_type = st.sidebar.selectbox("Vehicle Type", ["motorcycle", "scooter"])

# =====================================================
# CITY COORDINATES FOR MAP
# =====================================================
city_coordinates = {
    "BANG": (12.9716, 77.5946),
    "CHEN": (13.0827, 80.2707),
    "HYD": (17.3850, 78.4867),
    "MYS": (12.2958, 76.6394),
    "INDO": (22.7196, 75.8577)
}

restaurant_lat, restaurant_lon = city_coordinates[city]
delivery_lat = restaurant_lat + (distance * 0.01)
delivery_lon = restaurant_lon + (distance * 0.01)

# =====================================================
# FEATURE ENGINEERING
# =====================================================
def prepare_input(age, rating, distance, city):

    efficiency_score = age * rating
    distance_squared = distance ** 2
    rating_distance_interaction = rating * distance
    age_distance_interaction = age * distance

    city_encoding_map = {
        "BANG": 35,
        "CHEN": 33,
        "HYD": 32,
        "MYS": 28,
        "INDO": 30
    }

    input_dict = {
        "Delivery_person_Age": age,
        "Delivery_person_Ratings": rating,
        "distance_km": distance,
        "efficiency_score": efficiency_score,
        "distance_squared": distance_squared,
        "rating_distance_interaction": rating_distance_interaction,
        "age_distance_interaction": age_distance_interaction,
        "City_encoded": city_encoding_map.get(city, 32)
    }

    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df)
    df = df.reindex(columns=model_columns, fill_value=0)

    return df

# =====================================================
# SESSION STATE
# =====================================================
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# =====================================================
# PREDICTION BUTTON
# =====================================================
if st.button("🚀 Generate Delivery Intelligence Report"):

    input_df = prepare_input(age, rating, distance, city)
    log_pred = model.predict(input_df)[0]
    prediction = float(np.expm1(log_pred))

    st.session_state.prediction = prediction

# =====================================================
# RESULTS
# =====================================================
if st.session_state.prediction is not None:

    prediction = st.session_state.prediction
    lower = max(prediction - 6, 0)
    upper = prediction + 6
    delay_probability = min((prediction / 60) * 100, 100)

    # KPI CARDS
    col1, col2, col3 = st.columns(3)
    col1.metric("📦 ETA (min)", f"{prediction:.2f}")
    col2.metric("⚠ Delay Risk (%)", f"{delay_probability:.1f}")
    col3.metric("📊 Confidence Range", f"{lower:.1f} - {upper:.1f}")

    # =====================================================
    # GAUGE
    # =====================================================
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={'text': "Estimated Delivery Time"},
        gauge={'axis': {'range': [0, 60]}}
    ))
    st.plotly_chart(gauge, use_container_width=True)

    # =====================================================
    # DELAY RISK BAR (FIXED)
    # =====================================================
    st.subheader("⚠ Delay Risk Level")
    st.progress(float(delay_probability) / 100)

    # =====================================================
    # LIVE MAP
    # =====================================================
    st.subheader("🗺 Live Route Visualization")

    m = folium.Map(location=[restaurant_lat, restaurant_lon], zoom_start=12)

    folium.Marker(
        [restaurant_lat, restaurant_lon],
        tooltip="Restaurant",
        icon=folium.Icon(color="green")
    ).add_to(m)

    folium.Marker(
        [delivery_lat, delivery_lon],
        tooltip="Delivery Location",
        icon=folium.Icon(color="red")
    ).add_to(m)

    folium.PolyLine(
        [[restaurant_lat, restaurant_lon], [delivery_lat, delivery_lon]],
        color="blue",
        weight=4
    ).add_to(m)

    st_folium(m, width=900)

    # =====================================================
    # AI SUMMARY
    # =====================================================
    st.subheader("🧠 AI Generated Operational Summary")

    if delay_probability > 70:
        summary = f"The delivery is expected to take {prediction:.1f} minutes with HIGH delay probability. Immediate operational optimization recommended."
    elif delay_probability > 40:
        summary = f"The delivery is projected at {prediction:.1f} minutes with moderate delay risk. Monitor route and driver performance."
    else:
        summary = f"The delivery is projected at {prediction:.1f} minutes with low delay probability. Current allocation appears efficient."

    st.info(summary)

    # =====================================================
    # DRIVER COMPARISON
    # =====================================================
    st.subheader("👥 Driver Comparison")

    colA, colB = st.columns(2)

    with colA:
        ageA = st.slider("Driver A Age", 18, 60, 30)
        ratingA = st.slider("Driver A Rating", 1.0, 5.0, 4.2)

    with colB:
        ageB = st.slider("Driver B Age", 18, 60, 40)
        ratingB = st.slider("Driver B Rating", 1.0, 5.0, 4.8)

    predA = float(np.expm1(model.predict(prepare_input(ageA, ratingA, distance, city))[0]))
    predB = float(np.expm1(model.predict(prepare_input(ageB, ratingB, distance, city))[0]))

    st.write(f"Driver A ETA: {predA:.2f} min")
    st.write(f"Driver B ETA: {predB:.2f} min")

    # =====================================================
    # WHAT-IF SIMULATION
    # =====================================================
    st.subheader("🔄 What-If Simulation")

    new_distance = st.slider("Simulate Different Distance", 0.5, 30.0, distance)
    sim_eta = float(np.expm1(model.predict(prepare_input(age, rating, new_distance, city))[0]))

    st.write(f"If distance changes to {new_distance} km → ETA becomes {sim_eta:.2f} min")

    # =====================================================
    # PDF EXPORT (FIXED VERSION)
    # =====================================================
    st.subheader("📄 Download Intelligence Report")

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Smart Delivery Intelligence Report", styles["Heading1"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(summary, styles["Normal"]))

    doc.build(elements)

    pdf_data = buffer.getvalue()
    buffer.close()

    st.download_button(
        label="Download PDF Report",
        data=pdf_data,
        file_name="Delivery_Intelligence_Report.pdf",
        mime="application/pdf"
    )
