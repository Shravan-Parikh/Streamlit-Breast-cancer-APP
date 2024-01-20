import streamlit as st
import pickle as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

import os

def show_information_page():
    st.title("Breast Cancer Information")
    
    st.header("What is Breast Cancer?")
    st.write("Breast cancer is a type of cancer that starts in the cells of the breast.")

    st.header("Benign vs. Malignant")
    st.write("Benign tumors are non-cancerous, while malignant tumors are cancerous and can spread to other parts of the body.")

    st.header("Causes of Breast Cancer")
    st.write("The exact cause of breast cancer is not known, but factors such as genetics, age, and hormonal changes can contribute.")

    st.header("Prevention")
    st.write("Early detection through regular screenings, maintaining a healthy lifestyle, and knowing your family history can help in prevention.")

    st.header("Breast Cancer Awareness")
    st.write("Increasing awareness about breast cancer, promoting regular check-ups, and supporting research are crucial for fighting the disease.")

def get_clean_data():

    current_directory = os.path.dirname(os.path.realpath(__file__))
    csv_file_path = os.path.join(current_directory, "../model/data/data.csv")
    data = pd.read_csv(csv_file_path)
    data = data.drop(['Unnamed: 32','id'],axis=1)

    data['diagnosis']= data['diagnosis'].map({ 'M':1,'B': 0})

    return data 

def add_sidebar():
    st.sidebar.header("Cell Nuclei Details")

    

    data=get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]
    
    input_dict ={}

    

    for label , key in slider_labels:
        input_dict[key]=st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )

    return input_dict

def get_scaled_value(input_dict):
    data = get_clean_data()
    x = data.drop(['diagnosis'],axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = x[key].max()
        min_val = x[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict 

def get_radar_chart(input_data):

    input_data=get_scaled_value(input_data)

    categories = ['Radius','Texture','Perimeter','Area','Smoothness',
                  'Compactness','Concavity',
                  'Concave Points','Symmetry','Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'],
            input_data['texture_mean'],
            input_data['perimeter_mean'],
            input_data['area_mean'],
            input_data['smoothness_mean'],
            input_data['compactness_mean'],
            input_data['concavity_mean'],
            input_data['concave points_mean'],
            input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']

        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
         r=[
            input_data['radius_se'],
            input_data['texture_se'],
            input_data['perimeter_se'],
            input_data['area_se'],
            input_data['smoothness_se'],
            input_data['compactness_se'],
            input_data['concavity_se'],
            input_data['concave points_se'],
            input_data['symmetry_se'],
            input_data['fractal_dimension_se']

        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
         r=[
            input_data['radius_worst'],
            input_data['texture_worst'],
            input_data['perimeter_worst'],
            input_data['area_worst'],
            input_data['smoothness_worst'],
            input_data['compactness_worst'],
            input_data['concavity_worst'],
            input_data['concave points_worst'],
            input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']

        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )

    return fig

def add_prediction(input_data):
    directory = os.path.dirname(os.path.realpath(__file__))
    model_file_path = os.path.join(directory, "../model/model.pkl")
    scaler_file_path = os.path.join(directory, "../model/scaler.pkl")
    model= pickle.load(open(model_file_path,"rb"))   
    scaler= pickle.load(open(scaler_file_path,"rb"))  

    input_array = np.array(list(input_data.values())).reshape(1,-1)

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Bengin</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis Malicious'>Malicious</span>", unsafe_allow_html=True)

    st.write("Probality of being Benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probality of being Malicious: ", model.predict_proba(input_array_scaled)[0][1])

    st.write("This app can asssit medical professionals in making a diagnosis, but should not he used as a subsititue fro a professional diagnosis")
     

def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()),unsafe_allow_html=True)
    

    input_data = add_sidebar()

    # st.write(input_data)

    with st.container():
        c1,c2= st.columns([10,1])
        with c1 :
            st.title("Breast Cancer Predictor")
        with c2 :
            if st.button("Info"):
                st.switch_page("pages/info.py")
        st.write("Connect this app to your cytology lab to help diagnose breast cancer form tissue sample. This app will predict using Machine Learning model whether a breast mass is benign or malignant based on the measurements it recives from your cytosis lab. update the measurements by hand using the sliders in sidebar.")
        

    column_1, column_2 = st.columns([4,1])

    with column_1:
        radar_chart=get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with column_2:
        
        add_prediction(input_data)




if __name__ =='__main__':
    main() 