import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
st.set_page_config(page_title = "CAClysis",page_icon="🦈",layout="wide")

# Load the dataset
costs_data = pd.read_csv('cac_dataset\customer_acquistion_costs.csv')



# st.metric(label="Temperature", value="70 °F", delta="1.2 °F")
st.image("elements\pictures\cfm_logo.gif")
st.title(":blue[_CAC_] Analysis :chart:")
st.subheader("Detailed EDA on Customer Acquistion Costs (FOOD-MART)", divider="rainbow")


left, middle, right = st.columns(3)


left.button("Click to view the dataset",icon="😃", key="check_data", use_container_width=True)
if st.session_state.check_data:
    st.dataframe(costs_data, width=1400, height=250)  

middle.button("Data Information",icon="😃", key="info_data", use_container_width=True)
if st.session_state.info_data:
    st.write("No. of data points(rows): ", costs_data.shape[0])
    st.write("No. of data features(columns): ", costs_data.shape[1])
    st.write("No. of categorical features: ", costs_data.select_dtypes(include='object').shape[1])
    st.write("No. of numerical features: ", costs_data.select_dtypes(include='number').shape[1])
    st.write("No. of missing values: ", costs_data.isnull().sum().sum())

right.button("Data Description",icon="😃", key="desc_data", use_container_width=True)
if st.session_state.desc_data:
    describe_dataframe = costs_data.describe().T
    st.dataframe(describe_dataframe.style.background_gradient(subset=['mean'],
                                                    cmap='mako_r').background_gradient(subset=['std'],
                                                                                         cmap='mako_r').background_gradient(subset=['50%'],
                                                                                                                             cmap='mako_r'),
                                                                                                                             width=1400, 
                                                                                                                             height=250)
    inference_dataframe = pd.DataFrame({"Inference": ["The store_sqft column has the highest mean of around 27988 while the low fat column has the lowest mean of 0.35 which suggests scaling to avoid issues when using distance based algorithms ."]}, index=[1])
    st.dataframe(inference_dataframe, width=1400, height=50, use_container_width=True)
                                                                                                                                                                                                                                       



# st.button("Click to view the dataset", key="check_data")
# if st.session_state.check_data:
#     st.dataframe(costs_data, width=1250, height=250)
#     st.write("No. of data points(rows): ", costs_data.shape[0])
#     st.write("No. of data features(columns): ", costs_data.shape[1])
# st.button("Collapse", key="collapse_data")
# if st.session_state.collapse_data:
    

# 1. Histogram for Customer Income Distribution
# sns.histplot(cac_data['avg. yearly_income'], kde=False, color="skyblue", ax=axes[0, 0])
# axes[0, 0].set_title('Income Distribution of Customers', fontsize=14)
# axes[0, 0].set_xlabel('Income (USD)', fontsize=12)
# axes[0, 0].set_ylabel('Frequency', fontsize=12)
# axes[0, 0].tick_params(axis='x', rotation=30)
# for i in axes[0,0].containers:
#     axes[0,0].bar_label(i, label_type = 'edge', fontsize = 10)  # Rotate x labels for better readability