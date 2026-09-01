import os
import requests
import streamlit as st

API_URL = os.getenv('API_URL', "http://localhost:8000")

st.set_page_config(page_title="Term Deposit Prediction",layout="centered")
st.title("Banking Client Deposit Predictor")

with st.form("client Form"):
    age = st.slider("Client age", 18,90,35)
    balance = st.number_input("Yearly Balance",value=50000)
    duration = st.number_input("Last Contact Duration(seconds)",value = 240)
    campaign = st.number_input("Campaign Contacts",min_value=1,value=2)
    job = st.selectbox('Job Type',["management","technician","entrepreneur","blue-collar","retired"])
    housing = st.selectbox("Housing Loan",['yes','no'])

    submitted = st.form_submit_button("Run Prediction")

    if submitted:
        #convert INR back to the dataset's original base scale
        model_balance = balance / 85

        payload = {
            "age": age, 
            "balance": model_balance,  # Send the converted balance to the API
            "duration": duration,
            "campaign": campaign, 
            "job": job, 
            "housing": housing 
        }       

        try:
            with st.spinner("Querying API...."):
                res = requests.post(f"{API_URL}/predict",json=payload)

                if res.status_code == 200:
                    data  = res.json()
                    if data["subscription_prediction"] == 1:
                        st.success(f"**Likely to Subscribe** (Probability : {data['deposit_probability']})")
                    else:
                        st.warning(f"**Unlikely to Subscribe** (Probability: {data['deposit_probability']})")
                else:
                    st.error(f"API Error: {res.text}") 

        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to FastAPI backend at {API_URL}. Ensure service is active.")