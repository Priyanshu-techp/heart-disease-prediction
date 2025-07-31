import pandas as pd
import streamlit as st
import joblib 
import numpy as np 

st.title('Customer Revenue Prediction')

model = joblib.load('model deploy/model.pkl')

# Input field
def user_input():
    Administrative = st.slider('Administrative', 0,30, 15, step = 1)
    Administrative_Duration = st.slider('Administrative_Duration', 0.0, 200.0, 50.0)
    Informational = st.slider('Informational', 0, 20, 8, step= 1)
    Informational_Duration = st.slider('Informational_Duration', 0, 300, 100, step= 30)
    ProductRelated = st.slider('ProductRelated', 0, 500, 250, step = 50)
    ProductRelated_Duration = st.slider('ProductRelated_Duration', 0,1000, 500, step = 100)
    BounceRates = st.slider('BounceRates', 0.0, 1.0, 0.5, step = 0.01)
    ExitRates = st.slider('ExitRates', 0.0, 1.0, 0.5, step = 0.01)
    PageValues = st.slider('PageValues', 0, 100, 30, step =  10)
    SpecialDay = st.slider('SpecialDay', 0.0, 1.0, 0.3)
    OperatingSystems = st.slider('OperatingSystems', 0, 10, 3, step =  1)
    Browser = st.slider('Browser', 0, 20, 10, step =  1)
    Region = st.slider('Region', 0, 10, 3, step =  1)
    TrafficType = st.slider('TrafficType', 0, 30, 10, step =  5)
    Month = st.selectbox('Month', ['Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov','Dec'])
    VisitorType = st.selectbox('VisitorType', ['Returning_Visitor', 'New_Visitor', 'Other'])
    Weekend = st.selectbox('Weekend', ['True', 'False'])

    input_data = {
        'Administrative': Administrative,
        'Administrative_Duration': Administrative_Duration,
        'Informational': Informational,
        'Informational_Duration':	Informational_Duration,
        'ProductRelated': ProductRelated,
        'ProductRelated_Duration':	ProductRelated_Duration,
        'BounceRates'	:  BounceRates,
        'ExitRates'	: ExitRates,
        'PageValues': PageValues,	 
        'SpecialDay': SpecialDay,	
        'OperatingSystems':	OperatingSystems,
        'Browser': Browser,
        'Region': Region,
        'TrafficType': TrafficType,	
        'Month'	: Month,
        'VisitorType': VisitorType,
        'Weekend': Weekend,
    }
    return input_data 

# Ab data lo
user_data = user_input()

# Prepare for prediction
def prepare_input(data):
    df = pd.DataFrame([data]) 

    # Encode categorical variables (same as during training)
    df['Month'] = df['Month'].map({
        'Feb': 'Month_Feb', 'Mar': 'Month_Mar', 'Apr': 'Month_Apr', 'May': 'Month_May',
        'June': 'Month_June', 'Jul': 'Month_Jul', 'Aug': 'Month_Aug', 'Sep': 'Month_Sep',
        'Oct': 'Month_Oct', 'Nov': 'Month_Nov', 'Dec': 'Month_Dec'
    })

    df['VisitorType'] = df['VisitorType'].map({
        'Returning_Visitor': 'VisitorType_Returning_Visitor',
        'New_Visitor': 'VisitorType_New_Visitor',
        'Other': 'VisitorType_Other'
    })

    df['Weekend'] = df['Weekend'].map({'True': 1, 'False': 0})

    # One-hot encode Month and VisitorType
    for col in ['Month_Aug', 'Month_Dec', 'Month_Feb', 'Month_Jul', 'Month_June', 'Month_Mar', 'Month_May', 'Month_Nov', 'Month_Oct', 'Month_Sep']:
        df[col] = 1 if col == df['Month'].values[0] else 0

    for col in ['VisitorType_New_Visitor', 'VisitorType_Other', 'VisitorType_Returning_Visitor']:
        df[col] = 1 if col == df['VisitorType'].values[0] else 0

    df.drop(['Month', 'VisitorType'], axis=1, inplace=True)

    return df

if st.button('🔮 Predict Revenue', type="primary"):
    input_df = prepare_input(user_data)
    # ✅ Move these inside
    st.write(input_df)


    if input_df is not None and not input_df.empty:
        prediction = model.predict(input_df)
        result = 'Will Purchase' if prediction[0] == 1 else 'Will Not Purchase'
        st.subheader(f'Prediction: {result}')
    else:
        st.error("⚠️ Input data is invalid or empty.")






# Run with:
# streamlit run "Model deploy/app.py"

