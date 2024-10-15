import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go

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
        .stButton>button {
            background-color: #6c757d;
            color: white;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #5a6268;
            color: white;
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
        
        # Create an animated bar chart
        fig = go.Figure(
            data=[
                go.Bar(
                    x=["Salary"],
                    y=[salary[0]],
                    marker_color='indianred',
                    text=[f"${salary[0]:,.2f}"],
                    textposition='auto'
                )
            ],
            layout=go.Layout(
                title="Estimated Salary",
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        buttons=[dict(label="Play", method="animate", args=[None])],
                    )
                ],
                yaxis=dict(range=[0, max(salary[0] * 1.2, 100000)]),
                xaxis=dict(range=[-0.5, 0.5]),
            ),
            frames=[
                go.Frame(
                    data=[
                        go.Bar(
                            x=["Salary"],
                            y=[v],
                            marker_color='indianred',
                            text=[f"${v:,.2f}"],
                            textposition='auto'
                        )
                    ]
                ) for v in np.linspace(0, salary[0], 20)
            ]
        )

        st.plotly_chart(fig)

        st.write("""
            ### Insights
            - Salaries vary significantly based on country and education level.
            - Experience plays a crucial role in determining salary.
            - This prediction model uses machine learning to provide an estimate based on historical data.
        """)

# Sidebar for additional information
st.sidebar.header("Additional Information")
st.sidebar.write("""
    - This prediction model is based on data from a variety of countries.
    - It takes into account education level and years of experience.
    - Use this tool to get a rough estimate of expected salary.
""")

if __name__ == "__main__":
    show_predict_page()
