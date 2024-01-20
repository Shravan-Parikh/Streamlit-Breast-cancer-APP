import streamlit as st

def show_information_page():
    st.header("How It Works")
    st.write("The Breast Cancer Predictor App utilizes a logistic regression model from the scikit-learn library in Python. The model is trained on a dataset obtained from the University of Wisconsin, specifically the Breast Cancer Wisconsin (Diagnostic) dataset, which is available on Kaggle. This dataset plays a crucial role in enhancing the accuracy of our predictions.")

    # Limitations
    st.header("Limitations")
    st.write("While our app is a powerful tool for risk assessment, it comes with certain limitations:")
    st.write("- The app is currently designed for use in a laboratory setting to fetch real-time data. Data input is manual, facilitated through sliders in the sidebar.")
    st.write("- It is essential to highlight that the app is not a substitute for professional medical advice. The predictions are meant for risk assessment purposes only.")

    # References and Sources
    st.header("References and Sources")
    st.write("We believe in transparency and have based our app on reputable sources:")
    st.write("- The dataset used for training the prediction model is sourced from the Breast Cancer Wisconsin (Diagnostic) dataset from Kaggle.")
    st.write("- Our machine learning model is built upon the logistic regression algorithm provided by the scikit-learn library.")


def Disclamer():
    st.header("Disclaimer")
    st.write("**Disclaimer:**")
    st.write("The predictions provided by the Breast Cancer Predictor App are not infallible and should not be used as the sole basis for medical decisions. It is crucial to consult with healthcare professionals for personalized advice tailored to your specific health needs.")

    # Breast Cancer Awareness
    st.header("Breast Cancer Awareness")
    st.write("We are committed to raising awareness about breast cancer and promoting early detection:")
    st.write("- Learn more about breast cancer awareness, early detection, and preventive measures.")
    st.write("- Explore reputable resources on breast health through the following links:")
    st.write("  - [American Cancer Society](https://www.cancer.org/)")
    st.write("  - [National Breast Cancer Foundation](https://www.nationalbreastcancer.org/)")
    st.write("Remember, your health is a priority, and early awareness can make a significant impact on breast cancer outcomes.")

# Run the app
def main():
    st.set_page_config(
        page_title="Breast Cancer Information",
        page_icon=":female-doctor",
        layout="wide",
        initial_sidebar_state='collapsed'
    )
    column_1, column_2 = st.columns([1,1])

    with column_1:
        show_information_page()

    with column_2:
        Disclamer()




if __name__ == "__main__":
    main()
