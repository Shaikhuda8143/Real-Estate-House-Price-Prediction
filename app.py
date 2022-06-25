import pandas as pd
import streamlit as st 
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

st.title('Real estate prediction')
html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Real estate ML App </h2>
</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)

st.sidebar.header('User Input Parameters')

def user_input_features():
    house_age = st.sidebar.number_input('x2_h_age')
    distance_MTR = st.sidebar.number_input('x3_MRT_dist')
    stores_nearby= st.sidebar.selectbox('x4_stores',range(0,100))
    data = {
            'house_age':house_age,
            'distance_MTR':distance_MTR,
            'stores_nearby':stores_nearby}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

dt = pd.read_csv('Real estate.csv')
dt.rename({'X1 transaction date':'x1_trans_date','X2 house age':'x2_h_age','X3 distance to the nearest MRT station':'x3_MRT_dist','X4 number of convenience stores':'x4_stores','X5 latitude':'x5_latitude','X6 longitude':'x6_longitude','Y house price of unit area':'house_price'},inplace=True,axis = 1)



X = dt.iloc[:,[1,2,3]]
Y = dt.iloc[:,-1]
clf = LinearRegression()
clf.fit(X,Y)

prediction = clf.predict(df)
print(prediction)

st.subheader('Predicted Result')
st.write(prediction)
