import streamlit as st

import requests

from config import API_URL

st.set_page_config(

    page_title="House Price Prediction",

    page_icon="🏠",

    layout="centered",

)
st.title(
    "🏠 House Price Prediction"
)

st.write(
    "Enter the house details below."
)

overall_quality = st.slider(

    "Overall Quality",

    min_value=1,

    max_value=10,

    value=5,

)
st.sidebar.header(
    "About"
)

st.sidebar.write(

    """
    House Price Prediction

    Machine Learning Portfolio Project

    Built using:

    • Scikit-Learn

    • FastAPI

    • Streamlit

    """

)
living_area = st.number_input(

    "Living Area (sq ft)",

    min_value=100,

    value=1500,

)

garage_cars = st.number_input(

    "Garage Capacity",

    min_value=0,

    max_value=6,

    value=2,

)

basement = st.number_input(

    "Basement Area",

    min_value=0,

    value=800,

)

year = st.number_input(

    "Year Built",

    min_value=1800,

    max_value=2030,

    value=2000,

)

predict = st.button(
    "Predict Price"
)

payload = {

    "OverallQual": overall_quality,

    "GrLivArea": living_area,

    "GarageCars": garage_cars,

    "TotalBsmtSF": basement,

    "YearBuilt": year,

}
if predict:

    try:

        response = requests.post(
            API_URL,
            json=payload,
)

        response.raise_for_status()

        prediction = response.json()

        st.success(

            f"Successfully connected to the prediction API."

        )

    except requests.exceptions.RequestException:

        st.error(

            "Could not connect to the prediction API."

        )
if predict:

    prediction = response.json()

    st.success(

        f"Estimated House Price: ${prediction['predicted_price']:,.2f}"

    )