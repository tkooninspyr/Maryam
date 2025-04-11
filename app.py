import streamlit as st 
import pandas as pd
import numpy as np 
#import xgboost 
import joblib 
#import shap 
# Attempt to load the model
try:
    model = joblib.load('final_xgb_model.pk1')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")

# Load the transformed test datasets
Test_X_transformed = np.load('Test_X_transformed.npy') 
Test_y_transformed = np.load('Test_y_transformed.npy')

# Load feature names
feature_names = np.load('feature_names.npy', allow_pickle=True)

# Create a DataFrame for easy manipulation
test_df = pd.DataFrame (Test_X_transformed, columns=feature_names)
test_df['y'] = Test_y_transformed
st.title('XGBoost Model Prediction for Subcription')
st.write("Note: This Streamlit app loads random samples from the test set, makes predictions using a pretrained XGBoost model, and compare the prediction with True Labels") 

# Button to choose 10 random data points
if st.button('Choose 10 Random Test Data Points'):
    sample_data = test_df.sample (10)
    st.session_state.sample_data = sample_data # Store in session state
    st.write("Selected Test Data Points: ")
    st.dataframe(sample_data)

# Debug: Check if the session state persists
if 'sample_data' in st.session_state:
    st.write("10 Data Points are Randomly Selected.")
    st.write("Please Press on Predict to Get Predictions of XGBoost Model")

# Button to predict
if 'sample_data' in st.session_state and st.button('Predict'):
    # Predict and display predictions
    input = st.session_state.sample_data.iloc[:, -1].to_numpy()
    predictions = model.predict(input)
    #predictions
    st.session_state.sample_data['Predicted Label'] = predictions
    st.session_state.sample_data['Correct Prediction'] = np.where(st.session_state.sample_data['Predicted Label'] == st.session_state.sampel_data['y'], 'Yes')
    #st.write('i am here')
    #st.session_state.sample_data
    def highlight_cols(x):
        red = 'background-color: red;'
        green = 'background-color: green;'
        return [green if v == 'Yes' else red if v == 'No' else '' for v in x]

    st.dataframe(st.session_state.sample_data.style.apply (highlight_cols, subset=['Correct Prediction']))
else:
    if 'sample_data' not in st.session_state:
        st.write("Please select data points first by clicking 'Choose 10 Random Test Data Points'.")

