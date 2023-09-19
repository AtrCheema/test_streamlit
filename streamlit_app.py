
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="NGBoost", #page_icon="🖼️",
    #initial_sidebar_state="collapsed"
)

st.title('Adsorption Capacity of biochar for PO4')


with st.form('key1'):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pyrol_temp = st.number_input('Pyrolysis Temperature', value=10)
    with col2:
        heat_rate = st.number_input("Heating Rate", value=10)
    with col3:
        pyrol_time = st.number_input("Pyrolysis Time", value=10)
    with col4:
        c = st.number_input("Carbon (%)", value=10)

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        o = st.number_input("O (%)", value=10)
    with col6:
        surface_area = st.number_input("Surface Area", value=10)
    with col7:
        ads_time = st.number_input("Adsorption Time", value=10)
    with col8:
        ci = st.number_input("Initial Concentration", value=10)

    col9, col10, col11, col12 = st.columns(4)
    with col5:
        sol_ph = st.number_input("Solution pH", value=10)
    with col6:
        rpm = st.number_input("rpm", value=10)
    with col7:
        vol = st.number_input("Volume (L)", value=10)
    with col8:
        loading = st.number_input("Loading (g)", value=10)

    col13, col14, col15, col16 = st.columns(4)
    with col5:
        ads_temp = st.number_input("Adsorption Temp", value=10)
    with col6:
        ion_conc = st.number_input("Ion Concentration", value=10)
    with col7:
        adsorbent = st.number_input("Adsorbent", value=10)


    st.form_submit_button(label="Predict")

st.write(pyrol_temp + pyrol_time + ci)
st.text_area(label="Upper Bound", value=ci + 10)
st.text_area(label="Lower Bound", value=ci - 10)

