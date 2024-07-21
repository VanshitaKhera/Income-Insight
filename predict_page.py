import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the Salary""")
    st.write("""Please provide the following details to estimate the salary of a software developer.""")

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education_levels = (
        "Less than a Bachelors",
        "Bachelor's degree",
        "Master's degree", 
        "Post grad",
    )

    # Custom styling
    st.markdown(
        """
        <style>
        .stSelectbox, .stSlider {
            margin-bottom: 20px;
        }
        .stButton {
            margin-top: 20px;
        }
        </style>
        """, unsafe_allow_html=True
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education_levels)
    experience = st.slider("Years of Experience", 0, 50, 3)

    st.write(
        """
        Adjust the slider to reflect your years of professional coding experience.
        """
    )

    ok = st.button("Compute Salary")

    if ok:
        X = np.array([[country, education, experience]])
        X[:, 0] = le_country.transform(X[:, 0])
        X[:, 1] = le_education.transform(X[:, 1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:,.2f}")
