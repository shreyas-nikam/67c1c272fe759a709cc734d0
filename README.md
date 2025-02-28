
# QuLab: Financial Distress Predictor

## Description

QuLab: Financial Distress Predictor is a Streamlit application designed to demonstrate the application of machine learning in finance. It predicts the probability of financial distress for a company based on key financial ratios: Debt-to-Equity Ratio, Current Ratio, Return on Assets (ROA), and Profit Margin.

This application is developed as a hands-on educational tool for the "AI and Big Data in Investments" course by QuantUniversity. It uses a simplified logistic regression model trained on synthetic data for illustrative purposes and should not be used for real-world financial decision-making.

The application allows users to:
- Input financial ratios for a hypothetical company through an interactive sidebar.
- Obtain a prediction of financial distress probability based on a trained machine learning model.
- Visualize feature importance to understand which financial ratios are most influential in the prediction.
- Explore data distributions of financial ratios for distressed and non-distressed companies in a synthetic dataset.
- Learn about the significance of each financial ratio and its interpretation in financial analysis.

This tool serves as an educational resource to understand the practical applications of AI and machine learning in finance, particularly in risk assessment.

## Installation

To run this application, you need to have Python installed on your system. It is recommended to use Python 3.8 or higher.

1.  **Clone the repository (if applicable):**
    If you have access to a repository containing the application files, clone it using git:
    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```

2.  **Navigate to the project directory:**
    If you downloaded the application files, navigate to the directory containing the `your_script_name.py` file (replace `your_script_name.py` with the actual name of your Streamlit script file).

    ```bash
    cd <project_directory>
    ```

3.  **Create a virtual environment (optional but recommended):**
    It's best practice to create a virtual environment to isolate project dependencies.

    ```bash
    python -m venv venv
    ```
    or
    ```bash
    python3 -m venv venv
    ```

4.  **Activate the virtual environment:**

    -   **On Windows:**
        ```bash
        venv\Scripts\activate
        ```

    -   **On macOS and Linux:**
        ```bash
        source venv/bin/activate
        ```

5.  **Install the required packages:**
    Install the necessary Python libraries using pip. You can either create a `requirements.txt` file with the following content in your project directory:

    ```
    streamlit
    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn
    ```

    And then run:
    ```bash
    pip install -r requirements.txt
    ```

    Alternatively, install the packages individually using:
    ```bash
    pip install streamlit pandas numpy scikit-learn matplotlib seaborn
    ```

## Usage

1.  **Run the Streamlit application:**
    From your terminal, in the project directory where your Streamlit script (`your_script_name.py`) is located, run the following command:

    ```bash
    streamlit run your_script_name.py
    ```
    Replace `your_script_name.py` with the actual name of your Python script file.

2.  **Access the application in your browser:**
    Once the application is running, Streamlit will provide a local URL in your terminal, usually `http://localhost:8501`. Open this URL in your web browser to access the Financial Distress Predictor application.

3.  **Input Financial Ratios:**
    -   In the left sidebar, you will find input fields for the following financial ratios:
        -   **Debt-to-Equity Ratio:** Enter the company's debt-to-equity ratio.
        -   **Current Ratio:** Enter the company's current ratio.
        -   **Return on Assets (ROA):** Enter the company's return on assets.
        -   **Profit Margin:** Enter the company's profit margin.
    -   Use the number input fields to adjust the values. Default values are provided as starting points. You can also refer to the tooltips for each input field for a brief explanation of the ratio.

4.  **Predict Financial Distress:**
    -   After entering the desired financial ratios, click the **"Predict Distress"** button located in the sidebar.

5.  **View Prediction Results:**
    -   The main panel of the application will display the prediction results under the "Prediction Results" section.
    -   The predicted probability of financial distress will be shown as a percentage.
    -   Based on the probability, a message will indicate the likelihood of financial distress:
        -   A **warning message** will appear if the predicted probability is above 50%, suggesting a higher likelihood of financial distress.
        -   A **success message** will appear if the predicted probability is 50% or below, suggesting a lower likelihood of financial distress.

6.  **Explore Visualizations:**
    -   **Feature Importance:** A bar chart will be displayed under the "Feature Importance" section, illustrating the importance of each financial ratio in the prediction model. This helps understand which ratios are most influential in determining financial distress.
    -   **Data Distribution and Ratio Relationships:** Under the "Data Distribution and Ratio Relationships" section, you can use the dropdown menu to select a financial ratio. A histogram will then visualize the distribution of the chosen ratio for both financially distressed and non-distressed companies from the synthetic dataset used to train the model. This visualization helps understand how different ratios are distributed across these two groups.

7.  **Understand Financial Ratios:**
    -   For detailed explanations of each financial ratio used in the application, expand the **"Understanding Financial Ratios"** section located at the top of the main panel. This section provides definitions, formulas, typical ranges, and interpretations for each ratio.

## Credits

Developed by QuantUniversity as part of the **AI and Big Data in Investments** course.

Visit [https://www.quantuniversity.com/](https://www.quantuniversity.com/) for more information about QuantUniversity and their courses.

## License

© 2025 QuantUniversity. All Rights Reserved.

This application is for educational use only. Any reproduction or commercial use requires prior written consent from QuantUniversity. For full legal documentation, please visit [link to legal documentation if available].
