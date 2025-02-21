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
import pickle
import time

st.set_page_config(page_title = "Prediction",page_icon="🦈",layout="wide",initial_sidebar_state="collapsed")

# Load the dataset
costs_data = pd.read_csv(r'cac_dataset/customer_acquisition_costs.csv')
costs_data = costs_data.drop(["avg_cars_at home(approx).1"], axis=1)
data_predictor = pd.DataFrame(columns = costs_data.columns)
data_predictor = data_predictor.drop("cost",axis=1)
costs_columns = list(costs_data.columns)



# st.metric(label="Temperature", value="70 °F", delta="1.2 °F")
with st.container(border=True):
    left,right = st.columns(2, vertical_alignment="center")
    right.image(r"elements/pictures/cfm_logo.gif",use_container_width=True)
    left.title(":blue[_CAC_] Analysis :chart:")
    left.markdown("#### Detailed EDA on Customer Acquistion Costs (FOOD-MART)")
    # left.subheader("Detailed EDA on Customer Acquistion Costs (FOOD-MART)")

with st.container():
    selected = option_menu(menu_title=None,options=["Playground", "EDA", "Preprocess", "Prediction"],
                                icons=['house', "graph-up-arrow","cloud-upload","signal"],menu_icon="cast",
                                default_index=3,orientation="horizontal",styles={"nav-link":
                                                                                  {"text-align": "left","--hover-color": "#eee",}
                                                                                  ,"nav-link-selected": 
                                                                                  {"background-color": "green"}})
    # # if selected == "Home":
        # # st.switch_page("main.py")
    if selected == "EDA":
        st.switch_page(r"pages/eda.py")
    if selected == "Preprocess":
        st.switch_page(r"pages/preprocess.py")
    if selected == "Playground":
        st.switch_page(r"app.py")




def preprocess_test(data, binary_encoder, scaler):

    data["avg. yearly_income"] = data['avg. yearly_income'].str.replace(r'\$', '', regex=True) #cleaned yearly income
    def media_cleaner(value):
        if 'Daily' in value:
            value = "Daily Media"
        elif 'Sunday' in value:
            value = "Sunday Media"
        return value
    data["media_type"] = data["media_type"].apply(media_cleaner) #cleaned mediatype

    # Map the data value to strength score
    education_mapping = {"Partial High School": 1,"High School Degree": 2,"Partial College": 3,"Bachelors Degree": 4,"Graduate Degree": 5}
    houseowner_mapping = {"Y": 1, "N": 0}
    occupation_mapping = {'Manual' :1,'Skilled Manual':2,'Clerical':3,'Professional':4,'Management':5}
    member_card_mapping = {"Normal": 1,"Bronze": 2,"Silver": 3,"Golden": 4}
    income_mapping = {"10K - 30K": 1,"30K - 50K": 2,"50K - 70K": 3,"70K - 90K": 4,"90K - 110K": 5,"110K - 130K": 6,"130K - 150K": 7,"150K +": 8}
    car_mapping = {"0 car": 0.0,"1 car": 1.0,"2 car": 2.0,"3 car": 3.0,"4 car": 4.0}
    weights = {"education_encoded": 0.1,"houseowner_encoded": 0.15,"member_card_encoded": 0.2,"income_encoded": 0.3, "occupation_encoded": 0.1,"cars_at_home" : 0.15}

    data["profile_strength"] = (
        (data["education"].map(education_mapping)) * weights["education_encoded"] +
        (data["houseowner"].map(houseowner_mapping)) * weights["houseowner_encoded"] +
        (data["member_card"].map(member_card_mapping)) * weights["member_card_encoded"] +
        (data["avg. yearly_income"].map(income_mapping)) * weights["income_encoded"] +
        (data["occupation"].map(occupation_mapping)) * weights["occupation_encoded"] +
        (data["avg_cars_at home(approx)"]) * weights["cars_at_home"] ) 

    ordinal_cat = ["education","occupation","member_card","avg. yearly_income","store_type"]
    dic = {}
    for i in range(len(ordinal_cat)):
        dic[ordinal_cat[i]] = data[ordinal_cat[i]].unique()


    codes = [0,1,2,3,4,5,6,7]
    ed_map = pd.DataFrame({"education" : ["Partial High School","High School Degree","Partial College","Bachelors Degree","Graduate Degree"],
                        "encoding_values" : codes[:5]})
    mem_map = pd.DataFrame({"member_card":["Normal","Bronze","Silver","Golden"],
                            "encoding_values" : codes[:4]})
    occ_map = pd.DataFrame({"occupation" : ['Manual', 'Skilled Manual', 'Clerical', 'Professional', 'Management'],
                        "encoding_values" : codes[:5]})
    inc_map = pd.DataFrame({"avg. yearly_income" :['10K - 30K', '30K - 50K', '50K - 70K', '70K - 90K','90K - 110K', '110K - 130K', '130K - 150K', '150K +'],
                            "encoding_values" : codes})
    store_map = pd.DataFrame({"store_type":['Small Grocery', 'Mid-Size Grocery', 'Supermarket', 'Gourmet Supermarket', 'Deluxe Supermarket'],
                            "encoding_values" : codes[:5]})

    ordinal_order = [ed_map[ordinal_cat[0]],occ_map[ordinal_cat[1]],mem_map[ordinal_cat[2]],inc_map[ordinal_cat[3]],store_map[ordinal_cat[4]]]

    for i in range(len(ordinal_order)):
        data[ordinal_cat[i]] = pd.Categorical(data[ordinal_cat[i]], categories=ordinal_order[i], ordered=True).codes


    nominal_cat = data.select_dtypes('object').columns
    data = binary_encoder.transform(data)

    preprocessed_data = pd.DataFrame(scaler.transform(data), index=data.index, columns=data.columns)
    return preprocessed_data


  

st.success("_Store Details_ :house:")
with st.container(border = True):
    state1, state2, state3, state4= st.columns(4)
    with state1:
        with st.container():
            sales_country = costs_data["sales_country"].unique()
            country_option = st.selectbox("Select Country", sales_country, key="sales_country")
    with state2:
        with st.container():
            store_state = costs_data[costs_data["sales_country"] == country_option]["store_state"].unique()
            state_option = st.selectbox("Select State", store_state, key="store_state")
    with state3:
        with st.container():
            store_city = costs_data[costs_data["store_state"] == state_option]["store_city"].unique()
            city_option = st.selectbox("Select City", store_city, key="store_city")
    with state4:
        with st.container():
            store_type = costs_data[costs_data["store_city"] == city_option]["store_type"].unique()
            store_option = st.selectbox("Select Store type", store_type, key="store_type")


    area2, area3, area4 = st.columns(3)
    with area2:
        with st.container():
            grocery_area = st.number_input("Grocery Area (sqft)", value=0, placeholder="Enter Area...")
    with area3:
        with st.container():
            meat_area = st.number_input("Meat Area (sqft)", value=0, placeholder="Enter Area...")
    with area4:
        with st.container():
            frozen_area = st.number_input("Frozen Area (sqft)", value=0, placeholder="Enter Area...")



st.success("_Additional Store Facilities_ :coffee:")
with st.container(border = True):
    fac1, fac2, fac3, fac4, fac5= st.columns(5)
    with fac1:
        with st.container():
            coffee = costs_data["coffee_bar"].unique()
            coffee_str = ["No", "Yes"]
            coffee_dic = {coffee_str[0]:0.0,
                    coffee_str[1]:1.0}
            coffee_op = st.selectbox("Coffee Bar", coffee_str, key="coffee")
            coffee_option = coffee_dic[coffee_op]
    with fac2:
        with st.container():
            video = costs_data["video_store"].unique()
            video_str = ["No", "Yes"]
            video_dic = {video_str[0]:0.0,
                    video_str[1]:1.0}
            video_op = st.selectbox("Video Store", video_str, key="video")
            video_option = video_dic[video_op]
    with fac3:
        with st.container():
            salad = costs_data["salad_bar"].unique()
            salad_str = ["No", "Yes"]
            salad_dic = {salad_str[0]:0.0,
                    salad_str[1]:1.0}
            salad_op = st.selectbox("Salad Bar", salad_str, key="salad")
            salad_option = salad_dic[salad_op]
    with fac4:
        with st.container():
            prep_food = costs_data["prepared_food"].unique()
            prep_str = ["No", "Yes"]
            prep_dic = {prep_str[0]:0.0,
                    prep_str[1]:1.0}
            prep_op = st.selectbox("Prepared Food", prep_str, key="prep_food")
            prep_food_option = prep_dic[prep_op]
    with fac5:
        with st.container():
            florist = costs_data["florist"].unique()
            flo_str = ["No", "Yes"]
            flo_dic = {flo_str[0]:0.0,
                    flo_str[1]:1.0}
            florist_op = st.selectbox("Florist",flo_str, key="florist")
            florist_option = flo_dic[florist_op]



st.success("_Food Product Details_ :pizza:")
with st.container(border = True):
    food1, food2, food3, food4, food5, food6 = st.columns(6)
    with food1:
        with st.container():
            food_family = costs_data[costs_data["store_type"] == store_option]["food_family"].unique()
            fam_option = st.selectbox("Select Food Family", food_family, key="food_family")
    with food2:
        with st.container():
            food_department = costs_data[costs_data["food_family"] == fam_option]["food_department"].unique()
            dept_option = st.selectbox("Select Food Department", food_department, key="food_dept")
    with food3:
        with st.container():
            food_category = costs_data[costs_data["food_department"] == dept_option]["food_category"].unique()
            cat_option = st.selectbox("Select Food Category", food_category, key="food_cat")
    with food4:
        with st.container():
            brand_name = costs_data[costs_data["food_category"] == cat_option]["brand_name"].unique()
            brand_option = st.selectbox("Select Brand", brand_name, key="brand_name")
    with food5:
        with st.container():
            low_fat = costs_data[costs_data["brand_name"] == brand_option]["low_fat"].unique()
            fat_str = ["No", "Yes"]
            fat_dic = {fat_str[0]:0.0,
                    fat_str[1]:1.0}
            fat_op = st.selectbox("Select Low Fat option", fat_str, key="low_fat")
            fat_option = fat_dic[fat_op]
    with food6:
        with st.container():
            recyclable_package = costs_data[costs_data["brand_name"] == brand_option]["recyclable_package"].unique()
            rec_str = ["No", "Yes"]
            rec_dic = {rec_str[0]:0.0,
                    rec_str[1]:1.0}
            recycle_op = st.selectbox("Select Recyclability option", rec_str, key="recyclable_package")
            recycle_option = rec_dic[recycle_op]



st.success("_Customer Details_ :man:")
with st.container(border = True):
    cus1, cus2, cus3, cus4 = st.columns(4)
    with cus1:
        with st.container():
            gender = costs_data["gender"].unique()
            gender_option = st.selectbox("Select Gender",gender, key="gender")
    with cus2:
        with st.container():
            marital_status = costs_data["marital_status"].unique()
            marital_option = st.selectbox("Select Marital Status",marital_status, key="marital_status")
    with cus3:
        with st.container():
            total_children = sorted(costs_data["total_children"].unique())
            total_children_option = st.selectbox("Select Total Children",total_children, key="total_children")
    with cus4:
        with st.container():
            children_home = sorted(costs_data[costs_data["total_children"]==total_children_option]["num_children_at_home"].unique())
            children_home_option = st.selectbox("Select Children at home",children_home, key="children_home")

    qual1, qual2, qual3, qual4, qual5, qual6, qual7 = st.columns(7)   
    with qual1:
        with st.container():
            education = costs_data["education"].unique()
            education_option = st.selectbox("Select Education",education, key="education")
    with qual2:
        with st.container():
            occupation = costs_data["occupation"].unique()
            occupation_option = st.selectbox("Select Occupation",occupation, key="occupation")
    with qual3:
        with st.container():
            houseowner = costs_data["houseowner"].unique()
            houseowner_option = st.selectbox("Select whether Houseowner",houseowner, key="houseowner")
    with qual4:
        with st.container():
            car = costs_data["avg_cars_at home(approx)"].unique()
            car_option = st.selectbox("Select No. of Cars",car, key="car")
    with qual5:
        with st.container():
            income = costs_data["avg. yearly_income"].unique()
            income_option = st.selectbox("Select Income range",income, key="income")
    with qual6:
        with st.container():
            member = costs_data["member_card"].unique()
            member_option = st.selectbox("Select Card",member, key="member")
    with qual7:
        with st.container():
            promotion = costs_data["promotion_name"].unique()
            promotion_option = st.selectbox("Select Promotion Available",promotion, key="promotion")


st.success("_Inventory Details_ :dollar:")
with st.container(border = True):
    sale1, sale2, sale3, sale4 = st.columns(4)
    with sale1:
        with st.container():
            unit_sale = st.number_input("Unit Sales (in million)", value=0, placeholder="Enter Unit Sales...")
            # unit_sale = sorted(costs_data["unit_sales(in millions)"].unique())
            # unit_option = st.selectbox("Unit Sales (in million)",unit_sale, key="unit_sale")
    with sale2:
        with st.container():
            srp = st.number_input("Unit Retail Price", value=0, placeholder="Enter Unit Price...")
            # srp_option = st.selectbox("Select SRP",srp, key="srp")
    with sale3:
        with st.container():
            case = st.number_input("Units per Case", value=0, placeholder="Enter Units/Case...")
            # case = sorted(costs_data["units_per_case"].unique())
            # case_option = st.selectbox("Select Units per Box",case, key="case")
    with sale4:
        with st.container():
            store_cost = st.number_input("Store cost (in million)", value=0, placeholder="Enter store Cost...")

    sale5, sale6, sale7 = st.columns(3)
    with sale5:
        with st.container():
            media = costs_data["media_type"].unique()
            media_option = st.selectbox("Select Media Campaign",media, key="media")
    with sale6:
        with st.container():
            gross = st.number_input("Gross Weight (lbs.)", value=0, placeholder="Enter Gross Weight...")
    with sale7:
        with st.container():
            net = st.number_input("Net Weight (lbs.)", value=0, placeholder="Enter Net Weight...")


store_sales = (unit_sale*srp)
store_area = grocery_area + meat_area + frozen_area

form_input = [cat_option, dept_option, fam_option, store_sales, store_cost, unit_sale, promotion_option, country_option, marital_option,
              gender_option, total_children_option, education_option, member_option, occupation_option, houseowner_option,car_option, income_option,
              children_home_option, brand_option, srp, gross, net, recycle_option, fat_option, case, store_option, city_option, state_option,
              store_area, grocery_area, frozen_area, meat_area, coffee_option, video_option, salad_option, prep_food_option, florist_option,
              media_option]
data_predictor.loc[len(data_predictor)] = form_input
st.dataframe(data_predictor, hide_index=True)

with open(r'models/binary_encoder' , 'rb') as be:
    binary_encoder = pickle.load(be)
with open(r'models/scaler' , 'rb') as sc:
    scaler = pickle.load(sc)
with open(r'models/model_pkl' , 'rb') as sc:
    prediction_model = pickle.load(sc)

encoded_data = preprocess_test(data_predictor,binary_encoder, scaler)
# st.dataframe(encoded_data, hide_index=True)
value = -1
left, middle, right = st.columns(3)
with middle:
    st.button("Predict", key="predict", use_container_width=True) #Created button to view description of data
    if st.session_state.predict:
        with st.spinner("Predicting...", show_time=True):
            time.sleep(2)
        st.success("Prediction Generated")
        value = np.round(prediction_model.predict(encoded_data)[0], 2)
if value!=-1:
    left,right = st.columns([0.7,0.3], border=True)
    left.markdown("#### :green[The Costs on media campaign complying to given details is predicted to be : ]")
    right.markdown("# :blue[{}] $ :dollar:".format(value))
