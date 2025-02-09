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
from streamlit_extras.switch_page_button import switch_page
from streamlit_option_menu import option_menu

st.set_page_config(page_title = "EDA",page_icon="🦈",layout="wide",initial_sidebar_state="collapsed")

# Load the dataset
costs_data = pd.read_csv(r'cac_dataset/customer_acquisition_costs.csv')
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
    right.image(r"elements/pictures/cfm_logo.gif",use_container_width=True)
    left.title(":blue[_CAC_] Analysis :chart:")
    left.markdown("#### Detailed EDA on Customer Acquistion Costs (FOOD-MART)")
    # left.subheader("Detailed EDA on Customer Acquistion Costs (FOOD-MART)")

with st.container():
    selected = option_menu(menu_title=None,options=["Playground", "EDA", "Train", "Prediction"],
                                icons=['house', "graph-up-arrow","cloud-upload","signal"],menu_icon="cast",
                                default_index=1,orientation="horizontal",styles={"nav-link":
                                                                                  {"text-align": "left","--hover-color": "#eee",}
                                                                                  ,"nav-link-selected": 
                                                                                  {"background-color": "green"}})
    if selected == "Playground":
        st.switch_page(r"app.py")
    if selected == "Train":
        st.switch_page(r"pages/train.py")
    if selected == "Prediction":
        st.switch_page(r"pages/prediction.py")


with st.container(border=True):
    if "disabled_one" not in st.session_state:
        st.session_state.disabled_one = False

    st.checkbox(":blue[_Exploration 1_] || Insights on Top 10 food categories sold.", key="disabled_one")
    if st.session_state.disabled_one:
        base_features = ["sales_country","marital_status","gender","member_card", "occupation" ,"yearly_income","store_type","media_type"]
        option_feature = st.selectbox("Select a feature",base_features,key="feature_option")

        category_list = list(costs_data[option_feature].unique())
        option_category = st.radio("Select category",category_list,horizontal=True,key="category_option")
        feature_data = costs_data[costs_data[option_feature]==option_category] 
        prod_sales = feature_data.groupby('food_category')
        sales_by_product = prod_sales['unit_sales(in millions)'].sum().reset_index()
        sales_by_product = sales_by_product.sort_values(by='unit_sales(in millions)',ascending=False)[:10]

        out_inference = "The most sold category of food for {} = \"{}\" is {} with unit sales of {} million".format(option_feature,
                                                                                                             option_category,
                                                                                                             sales_by_product.iloc[0][0],
                                                                                                             np.round(sales_by_product.iloc[0][1],2))
        inference_dataframe = pd.DataFrame({"Inference": [out_inference]})
        st.dataframe(inference_dataframe, width=1400, height=50,hide_index=True, use_container_width=True)

        title_s = "food categories vs unit sales for {} ".format(option_category)
        fig = px.line(sales_by_product,x="food_category", y="unit_sales(in millions)", title = title_s, markers=True)
        fig.update_traces(line_color='red', line_width=2)
        st.plotly_chart(fig)
        


with st.container(border=True):
    if "disabled_two" not in st.session_state:
        st.session_state.disabled_two = False

    st.checkbox(":blue[_Exploration 2_] || Insights on Top 10 food categories sold in stores with special facilities.", key="disabled_two")
    if st.session_state.disabled_two:

        base_features = ["coffee_bar","video_store","salad_bar","prepared_food","florist"]
        option_feature = st.selectbox("Select a special facility for Top 10 food category.",base_features,key="feature_option")

        category_list = list(costs_data[option_feature].unique())
        option_category = st.radio("Select category",category_list,horizontal=True,key="category_option")
        feature_data = costs_data[costs_data[option_feature]==option_category] 
        prod_sales = feature_data.groupby('food_category')
        sales_by_product = prod_sales['unit_sales(in millions)'].sum().reset_index()
        sales_by_product = sales_by_product.sort_values(by='unit_sales(in millions)',ascending=False)[:10]

        if option_category == 1:
            out_inference = "The most sold category of food for stores with \"{}\" is {} having unit sales of {} million".format(option_feature,
                                                                                                             sales_by_product.iloc[0][0],
                                                                                                             np.round(sales_by_product.iloc[0][1],2))
        else:
            out_inference = "The most sold category of food for stores without \"{}\" is {} having unit sales of {} million".format(option_feature,
                                                                                                             sales_by_product.iloc[0][0],
                                                                                                             np.round(sales_by_product.iloc[0][1],2))
        inference_dataframe = pd.DataFrame({"Inference": [out_inference]})
        st.dataframe(inference_dataframe, width=1400, height=50,hide_index=True, use_container_width=True)
        title_s = "food categories vs unit sales for {} = {}".format(option_feature, bool(option_category))
        fig = px.line(sales_by_product,x="food_category", y="unit_sales(in millions)", title = title_s, markers=True)
        fig.update_traces(line_color='green', line_width=2)
        st.plotly_chart(fig)


with st.container(border=True):
    if "disabled_three" not in st.session_state:
        st.session_state.disabled_three = False

    st.checkbox(":blue[_Exploration 3_] || Segmentation of membership-card holders.", key="disabled_three")
    if st.session_state.disabled_three:

        # country_list = list(costs_data['sales_country'].unique())
        # option = st.selectbox("Select country.",country_list,key="country_option")
        # country_data = costs_data[costs_data['sales_country']==option] 
        revenue_data = costs_data.copy()
        revenue_data["revenue(in millions)"] = costs_data['store_sales(in millions)']-costs_data['store_cost(in millions)']
        card_sales = revenue_data.groupby('member_card')
        revenue_by_card = card_sales["revenue(in millions)"].sum().reset_index()
        revenue_by_card = revenue_by_card.sort_values(by="revenue(in millions)",ascending=False)
        out_inference = "The {} card holders generate the most revenue of {} million $".format(revenue_by_card.iloc[0][0],
                                                                                                np.round(revenue_by_card.iloc[0][1],2))
        inference_dataframe = pd.DataFrame({"Inference": [out_inference,"Targeting promotions at this group could help generate higher returns."]})
        st.dataframe(inference_dataframe, width=1400, height=100,hide_index=True, use_container_width=True)
        title_s = "member_card vs store_revenue"
        fig = px.line(revenue_by_card,x="member_card", y="revenue(in millions)",title = title_s, markers=True)
        fig.update_traces(line_color='violet', line_width=2)
        st.plotly_chart(fig)

# with st.container(border=True):
#     if "disabled_three" not in st.session_state:
#         st.session_state.disabled_three = False

#     st.checkbox(":blue[_Insight 3_] : Top 10 food categories sold country-wise.", key="disabled_three")
#     if st.session_state.disabled_three:
#         country_list = list(costs_data['sales_country'].unique())
#         option = st.selectbox("Select country.",country_list,key="country_option")
#         country_data = costs_data[costs_data['sales_country']==option] 
#         prod_sales = country_data.groupby('food_category')
#         sales_by_product = prod_sales['unit_sales(in millions)'].mean().reset_index()
#         sales_by_product = sales_by_product.sort_values(by='unit_sales(in millions)',ascending=False)[1:11]
#         title_s = "food categories vs unit sales for {}".format(option)
#         fig = px.line(sales_by_product,x="food_category", y="unit_sales(in millions)",title = title_s, markers=True)
#         fig.update_traces(line_color='red', line_width=2)
#         st.plotly_chart(fig)
