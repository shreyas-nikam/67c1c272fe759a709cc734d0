import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="QuCreate Financial Distress Predictor", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Financial Distress Predictor")
st.divider()

# --- Introduction and Explanation ---
st.markdown("""
    ## Welcome to the Financial Distress Predictor Lab!

    This interactive application predicts the likelihood of financial distress for a company based on key financial ratios. 
    It's designed to be a hands-on demonstration of machine learning applications in finance, as discussed in the 
    **AI and Big Data in Investments** course by QuantUniversity, specifically relating to the challenges and applications of AI in finance.

    **How to use this lab:**
    1.  **Enter Financial Ratios:** Use the input form in the sidebar to enter values for different financial ratios for a hypothetical company.
    2.  **Get Prediction:** Click the 'Predict Distress' button to see the model's prediction of financial distress probability.
    3.  **Explore Visualizations:** Observe the charts and graphs that dynamically update to show feature importance and data distributions.
    4.  **Understand the Concepts:** Read the explanations provided throughout the application to learn about financial ratios, the machine learning model, and the insights generated.

    **Educational Purpose:**
    This application is for educational purposes to illustrate the concepts of machine learning in finance. It uses a simplified model and synthetic data. 
    For deeper understanding and advanced applications, consider exploring the full **AI and Big Data in Investments** course from QuantUniversity.

    Let's get started by inputting some financial data!
""")

st.sidebar.header("Input Financial Ratios")
st.sidebar.markdown("Enter the financial ratios for the company you want to assess.")

# --- Explanation of Financial Ratios ---
with st.expander("Understanding Financial Ratios"):
    st.write("""
        Financial ratios are vital tools for analyzing a company's financial performance and health. 
        They provide insights into various aspects of a company's operations, such as its ability to manage debt, generate profit, and efficiently use assets. 
        Here's a brief explanation of the ratios used in this application:

        - **Debt-to-Equity Ratio:** 
          - Formula: Total Liabilities / Shareholder's Equity
          - Description: Measures a company's financial leverage. A higher ratio indicates that the company has financed more of its assets with debt, which can imply higher financial risk.
          - Typical Range: Generally, a ratio around 1.0 to 1.5 is considered healthy, but it varies significantly by industry.

        - **Current Ratio:** 
          - Formula: Current Assets / Current Liabilities
          - Description: A liquidity ratio that measures a company's ability to pay short-term obligations. A higher ratio indicates that a company is more capable of meeting its short-term liabilities.
          - Typical Range: A ratio of 1.5 to 2.0 is often considered healthy for many industries.

        - **Return on Assets (ROA):** 
          - Formula: Net Income / Total Assets
          - Description: Measures how profitable a company is relative to its total assets. It gives an idea of how efficiently management is using its assets to generate earnings.
          - Typical Range: Varies widely by industry. Generally, a ROA of 5% or higher is considered good.

        - **Profit Margin:** 
          - Formula: Net Profit / Revenue
          - Description: Indicates how much net profit a company makes for every dollar of revenue. A higher profit margin means the company is more profitable.
          - Typical Range: Also varies by industry, but a profit margin of 10% or higher is often seen as healthy.

        **Important Note:** These ratios should be interpreted in the context of the company's industry and compared to industry averages and competitors. 
        This application provides a simplified prediction model for educational purposes and should not be used for actual financial decision-making without consulting with a financial professional and conducting thorough analysis.
    """)

# --- User Input Form in Sidebar ---
debt_to_equity = st.sidebar.number_input("Debt-to-Equity Ratio", min_value=0.0, format="%.2f", value=1.5, step=0.1, help="Total Liabilities divided by Shareholder's Equity. Higher values indicate higher leverage.")
current_ratio = st.sidebar.number_input("Current Ratio", min_value=0.0, format="%.2f", value=1.2, step=0.1, help="Current Assets divided by Current Liabilities. Measures short-term liquidity.")
roa = st.sidebar.number_input("Return on Assets (ROA)", format="%.2f", value=0.05, step=0.01, help="Net Income divided by Total Assets. Measures profitability relative to total assets.")
profit_margin = st.sidebar.number_input("Profit Margin", format="%.2f", value=0.10, step=0.01, help="Net Profit divided by Revenue. Indicates percentage of revenue remaining after all expenses.")

# --- Synthetic Data Generation ---
@st.cache_data
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    debt_to_equity_distressed = np.random.normal(3.0, 1.0, num_samples//2)
    debt_to_equity_non_distressed = np.random.normal(1.0, 0.5, num_samples//2)
    current_ratio_distressed = np.random.normal(0.8, 0.3, num_samples//2)
    current_ratio_non_distressed = np.random.normal(1.8, 0.4, num_samples//2)
    roa_distressed = np.random.normal(0.02, 0.03, num_samples//2)
    roa_non_distressed = np.random.normal(0.08, 0.04, num_samples//2)
    profit_margin_distressed = np.random.normal(0.05, 0.02, num_samples//2)
    profit_margin_non_distressed = np.random.normal(0.15, 0.05, num_samples//2)

    data = pd.DataFrame({
        'Debt-to-Equity Ratio': np.concatenate([debt_to_equity_distressed, debt_to_equity_non_distressed]),
        'Current Ratio': np.concatenate([current_ratio_distressed, current_ratio_non_distressed]),
        'ROA': np.concatenate([roa_distressed, roa_non_distressed]),
        'Profit Margin': np.concatenate([profit_margin_distressed, profit_margin_non_distressed]),
        'Financial Distress': np.concatenate([np.ones(num_samples//2), np.zeros(num_samples//2)])
    })
    return data

synthetic_data = generate_synthetic_data()

# --- Model Training ---
@st.cache_resource
def train_logistic_regression_model(data):
    X = data[['Debt-to-Equity Ratio', 'Current Ratio', 'ROA', 'Profit Margin']]
    y = data['Financial Distress']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)
    return model

model = train_logistic_regression_model(synthetic_data)

# --- Prediction ---
if st.sidebar.button("Predict Distress"):
    input_data = pd.DataFrame({
        'Debt-to-Equity Ratio': [debt_to_equity],
        'Current Ratio': [current_ratio],
        'ROA': [roa],
        'Profit Margin': [profit_margin]
    })
    prediction_probability = model.predict_proba(input_data)[:, 1][0]
    distress_probability_percentage = prediction_probability * 100

    st.subheader("Prediction Results")
    st.write(f"Based on the financial ratios provided, the predicted probability of financial distress is: **{distress_probability_percentage:.2f}%**")

    if prediction_probability > 0.5:
        st.warning("This indicates a higher likelihood of financial distress according to the model.")
    else:
        st.success("This indicates a lower likelihood of financial distress according to the model.")

    st.markdown("---")

    # --- Feature Importance Visualization ---
    st.subheader("Feature Importance")
    st.write("This chart shows the importance of each financial ratio in the model's prediction. Feature importance is derived from the coefficients of the Logistic Regression model. The magnitude of the coefficient indicates the strength of the feature's impact, and the sign indicates the direction (positive or negative correlation with financial distress).")

    feature_importance = pd.DataFrame({'Feature': ['Debt-to-Equity Ratio', 'Current Ratio', 'ROA', 'Profit Margin'],
                                     'Importance': model.coef_[0]})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    fig_importance, ax_importance = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax_importance, palette="viridis")
    ax_importance.set_title('Feature Importance in Financial Distress Prediction')
    ax_importance.set_xlabel('Coefficient Magnitude (Logistic Regression)')
    ax_importance.set_ylabel('Financial Ratio')
    st.pyplot(fig_importance)

    st.markdown("---")

    # --- Interactive Data Visualization ---
    st.subheader("Data Distribution and Ratio Relationships")
    st.write("Explore the distribution of financial ratios in our synthetic dataset and how they relate to financial distress. These visualizations help understand the underlying patterns the model learns from.")

    ratio_choice = st.selectbox("Choose a financial ratio to visualize:",
                                ['Debt-to-Equity Ratio', 'Current Ratio', 'ROA', 'Profit Margin'],
                                help="Select a ratio to see its distribution for distressed and non-distressed companies.")

    fig_hist, ax_hist = plt.subplots()
    sns.histplot(data=synthetic_data, x=ratio_choice, hue='Financial Distress', kde=True, ax=ax_hist, palette=['coral', 'skyblue'])
    ax_hist.set_title(f'Distribution of {ratio_choice} by Financial Distress Status')
    ax_hist.set_xlabel(ratio_choice)
    ax_hist.set_ylabel('Frequency')
    ax_hist.legend(title='Financial Distress', labels=['Distressed', 'Non-Distressed'])
    st.pyplot(fig_hist)

    st.write(f"This histogram shows how the distribution of the **{ratio_choice}** differs between companies labeled as financially distressed and non-distressed in our synthetic dataset. By observing the peaks and overlaps, you can get a sense of how this ratio might be indicative of financial health according to the generated data.")


st.divider()
st.write("© 2025 QuantUniversity. All Rights Reserved.")
st.caption("The purpose of this demonstration is solely for educational use and illustration. "
           "To access the full legal documentation, please visit this link. Any reproduction of this demonstration "
           "requires prior written consent from QuantUniversity.")
