import pandas as pd
import streamlit as st

df = pd.DataFrame({"Name1": [1, 2, 3], "Name2": [1, 2, 3]})

my_dict = {"Weight": ["kg", "tons", "lbs"], "Speed": ["m/s", "km/hr"]}


col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    name = st.selectbox("Choose name", options=df.columns, key=1)
with col2:
    prop = st.selectbox("Choose Property", options=my_dict.keys(), key=2)
with col3:
    unit = st.selectbox("Choose Unit", options=my_dict[prop], key=3)