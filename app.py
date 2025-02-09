import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import skew, kurtosis
import plotly.express as px
import plotly.figure_factory as ff

st.set_page_config(page_title = "CAClysis",page_icon="🦈",layout="wide")

# Load the dataset
costs_data = pd.read_csv(r'cac_dataset\customer_acquisition_costs.csv')
costs_data = costs_data.rename(columns={'avg. yearly_income': 'yearly_income'})
costs_data = costs_data.drop(["avg_cars_at home(approx).1"], axis=1)
costs_columns = list(costs_data.columns)

#float-type features
float_features = costs_data.select_dtypes(exclude="object").columns
float_categories = costs_data[float_features].nunique()
float_cat_df = pd.DataFrame({"float_Features": float_features, "Count of each categories":list(float_categories)})

#object-type features
object_features = costs_data.select_dtypes(include="object").columns
object_categories = costs_data[object_features].nunique()
object_cat_df = pd.DataFrame({"object_Features": object_features, "Count of each categories":list(object_categories)})



numerical_features = ['store_sales(in millions)', 'store_cost(in millions)','SRP','gross_weight', 'net_weight','cost']
len_num = len(numerical_features)
numerical_categories = costs_data[numerical_features].nunique()
numerical_df = pd.DataFrame({"Features": numerical_features, "Count of each categories":list(numerical_categories)})


categorical_features = [col for col in costs_columns if col not in numerical_features]
len_cat = len(categorical_features)
categorical_categories = costs_data[categorical_features].nunique()
categorical_df = pd.DataFrame({"Features": categorical_features, "Count of each categories":list(categorical_categories)})





# st.metric(label="Temperature", value="70 °F", delta="1.2 °F")
with st.container(border=True):
    left,right = st.columns(2, vertical_alignment="center")
    right.image(r"elements\pictures\cfm_logo.gif",use_container_width=True)
    left.title(":blue[_CAC_] Analysis :chart:")
    left.markdown("#### Detailed EDA on Customer Acquistion Costs (FOOD-MART)")
    # left.subheader("Detailed EDA on Customer Acquistion Costs (FOOD-MART)")


with st.container(border=True):
    if "disabled" not in st.session_state:
        st.session_state.disabled = False

    st.checkbox("Select the box to explore data", key="disabled")
    if st.session_state.disabled:
        left, middle, right = st.columns(3) #Created three sub containers for each function

        left.button("Click to view the dataset", key="check_data", use_container_width=True) #Created button to view dataset
        if st.session_state.check_data:
            st.dataframe(costs_data, width=1400, height=250)  

        middle.button("Data Information", key="info_data", use_container_width=True) #Created button to view info about data
        if st.session_state.info_data:
            info_df = pd.DataFrame({"No. of data points(rows)": [costs_data.shape[0]],
                                    "No. of data features(columns)": [costs_data.shape[1]],
                                    "No. of object-datatype features": [costs_data.select_dtypes(include='object').shape[1]],
                                    "No. of float-datatype features": [costs_data.select_dtypes(exclude='object').shape[1]],
                                    "No. of missing values": [costs_data.isnull().sum().sum()]})
            st.dataframe(info_df, width=1400, height=50,hide_index=True, use_container_width=True)

            with st.expander("Expand to know about object-type features"):
                obj_df = pd.DataFrame({"Inference": ["These features are of \"Object\" datatypes. All these features need to be encoded or using model like CATBoost which can handle categories internally."]})
                st.dataframe(obj_df, width=1400, height=50,hide_index=True, use_container_width=True)
                st.dataframe(object_cat_df.T, width=1400, height=100,hide_index=True, use_container_width=True)
                st.bar_chart(object_cat_df,x="object_Features",y="Count of each categories",use_container_width=True)


            with st.expander("Expand to know about float-type features"):
                flt_df = pd.DataFrame({"Inference": ["These features are of \"Float\" datatypes. Some features are evidenlty categorical which don't need encoding."]})
                st.dataframe(flt_df, width=1400, height=50,hide_index=True, use_container_width=True)
                st.dataframe(float_cat_df.T, width=1400, height=100,hide_index=True, use_container_width=True)
                st.bar_chart(float_cat_df,x="float_Features",y="Count of each categories",use_container_width=True)

        right.button("Data Description", key="desc_data", use_container_width=True) #Created button to view description of data
        if st.session_state.desc_data:
            describe_dataframe = costs_data.describe().T
            st.dataframe(describe_dataframe.style.background_gradient(subset=['mean'],
                                                            cmap='mako_r').background_gradient(subset=['std'],
                                                                                                cmap='mako_r').background_gradient(subset=['50%'],
                                                                                                                                    cmap='mako_r'),
                                                                                                                                    width=1400, 
                                                                                                                                    height=250)
            inference_dataframe = pd.DataFrame({"Inference": ["The \"store_sqft\" column has the highest mean of around 27988 while the \"low_fat\" column has the lowest mean of 0.35 which suggests scaling to avoid issues when using distance based algorithms ."]})
            st.dataframe(inference_dataframe, width=1400, height=50,hide_index=True, use_container_width=True)
                                                                                                                                                                                                                                        
with st.container(border=True):
    if "disabled_two" not in st.session_state:
        st.session_state.disabled_two = False

    st.checkbox("Select the box to explore more on categorical data", key="disabled_two")
    if st.session_state.disabled_two:
            option = st.selectbox("Select column to check frequency of data in each category.",categorical_features,key="cat_option")
            dict = {ind : costs_data[option].value_counts()[ind] for ind in costs_data[option].unique()}
            option_cat_df = pd.DataFrame(dict.items(), columns = [option, "Count"])
            st.dataframe(option_cat_df.T, width=1400, height=100,hide_index=True, use_container_width=True)
            tab1, tab2, tab3 = st.tabs(["Bar", "Line", "Area"])
            with tab1:
                st.bar_chart(option_cat_df,x=option, y= "Count",use_container_width=True)
            with tab2:
                st.line_chart(option_cat_df,x=option, y= "Count",use_container_width=True)
            with tab3:
                st.area_chart(option_cat_df,x=option, y= "Count",use_container_width=True)


with st.container(border=True):
    if "disabled_three" not in st.session_state:
        st.session_state.disabled_three = False

    st.checkbox("Select the box to explore more on numerical data", key="disabled_three")
    if st.session_state.disabled_three:
        option = st.selectbox("Select column to check distribution of data.",numerical_features,key="num_option")
        op_mean = np.mean(costs_data[option])
        op_median = np.median(costs_data[option])
        op_std = np.std(costs_data[option])
        op_var = np.var(costs_data[option])
        op_skew = skew(costs_data[option],axis=0, bias=True)
        op_1q = costs_data[option].quantile(0.25)
        op_3q = costs_data[option].quantile(0.75)
        op_iqr = op_3q - op_1q
        op_upper = min((op_3q + 1.5*(op_iqr)),costs_data[option].max())
        op_lower = max((op_1q - 1.5*(op_iqr)),costs_data[option].min())
        op_outliers = ((costs_data[option] < op_lower) | (costs_data[option] > op_upper)).sum()
        if op_skew==0:
            op_skewness = "Normally distributed"
            skew_inference = "Following feature doesnot need to be scaled since the feature is \"{}\".".format(op_skewness)
        elif op_skew>0:
            op_skewness = "Positively skewed"
            skew_inference = "Following feature needs to be scaled since the feature is \"{}\".".format(op_skewness)
        else:
            op_skewness = "Negatively skewed"
            skew_inference = "Following feature needs to be scaled since the feature is \"{}\".".format(op_skewness)
        op_kurt = kurtosis(costs_data[option],axis=0, bias=True)
        stats_df = pd.DataFrame({"Mean": [op_mean],
                                 "Median":[op_median],
                                "Standard-Deviation": [op_std],
                                "Variance": [op_var],
                                "Skew": [op_skew],
                                "Skewness":[op_skewness],
                                "Kurosis": [op_kurt],
                                "Q1": [op_1q],
                                "Q3": [op_3q],
                                "IQR": [op_iqr],
                                "Upper Fence":[op_upper],
                                "Lower Fence": [op_lower],
                                "Outliers count":[op_outliers]})
        st.dataframe(stats_df, width=1400, height=50,hide_index=True, use_container_width=True)
        left, right = st.columns(2)
        plot_data = costs_data[option]
        # fig = ff.create_distplot(hist_data=hist_data, group_labels=[option])
        # st.plotly_chart(fig, use_container_width=True)
        fig = px.histogram(plot_data)
        left.plotly_chart(fig)  
        fig = px.box(plot_data, orientation='h')
        right.plotly_chart(fig)

        if op_outliers > 0:
            out_inference = "The following feature \"{}\" has {}  outliers, which might affect models during training.".format(option, op_outliers)
        else:
            out_inference = "The following feature \"{}\"  has no outliers.".format(option)
        inference_dataframe = pd.DataFrame({"Inference": [out_inference,skew_inference]})
        st.dataframe(inference_dataframe, width=1400, height=100,hide_index=True, use_container_width=True)



with st.container(border=True):
    if "disabled_four" not in st.session_state:
        st.session_state.disabled_four = False

    st.checkbox("Select the box to play with data", key="disabled_four")
    if st.session_state.disabled_four:
        color_list = list(['unit_sales(in millions)','total_children','avg_cars_at home(approx)','num_children_at_home','recyclable_package',
 'low_fat','coffee_bar','video_store','salad_bar','prepared_food','florist','food_family','sales_country','marital_status',
 'gender','education','member_card','occupation','houseowner','store_type'])
        option_cat = st.selectbox("Select categorical feature",categorical_features,key="cat_select")
        # if option_cat in color_list:
        #     st.write("happens")
        #     color_list = color_list.remove(option_cat)
        option_num = st.selectbox("Select numerical feature",numerical_features,key="num_select")
        option_color = st.radio("Select column for colour :",color_list, horizontal=True, key="color_select")
        
        # option_color = st.radio("Select a category for fun hue.",)
        with st.container(border=True):
            st.bar_chart(costs_data, x=option_cat, y=option_num,color=option_color,stack=False,use_container_width=True)


        

# 1. Histogram for Customer Income Distribution
# sns.histplot(cac_data['avg. yearly_income'], kde=False, color="skyblue", ax=axes[0, 0])
# axes[0, 0].set_title('Income Distribution of Customers', fontsize=14)
# axes[0, 0].set_xlabel('Income (USD)', fontsize=12)
# axes[0, 0].set_ylabel('Frequency', fontsize=12)
# axes[0, 0].tick_params(axis='x', rotation=30)
# for i in axes[0,0].containers:
#     axes[0,0].bar_label(i, label_type = 'edge', fontsize = 10)  # Rotate x labels for better readability