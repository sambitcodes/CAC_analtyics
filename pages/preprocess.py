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
import time
import category_encoders as cat_encoder

st.set_page_config(page_title = "Preprocess",page_icon="🦈",layout="wide",initial_sidebar_state="collapsed")

# Load the dataset
costs_data = pd.read_csv(r'cac_dataset/customer_acquisition_costs.csv')
train_data = pd.read_csv(r'cac_dataset/customer_acquisition_costs.csv')
repeat_data = train_data.drop(["avg_cars_at home(approx).1"], axis=1) # removed reptitive
year_inc = pd.DataFrame({'avg. yearly_income':repeat_data['avg. yearly_income'].unique()})
cleaned_year_inc =  year_inc['avg. yearly_income'].str.replace(r'\$', '', regex=True).unique()
media_tp = pd.DataFrame({'media_type':repeat_data['media_type'].unique()})
def media_cleaner(value):
    if 'Daily' in value:
        value = "Daily Media"
    elif 'Sunday' in value:
        value = "Sunday Media"
    return value
cleaned_media_tp = pd.DataFrame({'media_type':media_tp['media_type'].apply(media_cleaner).unique()})
costs_columns = list(costs_data.columns)
cleaned_data = train_data.drop(["avg_cars_at home(approx).1"], axis=1)
cleaned_data["avg. yearly_income"] = cleaned_data['avg. yearly_income'].str.replace(r'\$', '', regex=True) #cleaned yearly income
cleaned_data["media_type"] = cleaned_data["media_type"].apply(media_cleaner) #cleaned mediatype 

ordinal_cat = ["education","occupation","member_card",
               "avg. yearly_income","store_type"]
dic = {}
for i in range(len(ordinal_cat)):
    dic[ordinal_cat[i]] = cleaned_data[ordinal_cat[i]].unique()
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

ed_encoded=cleaned_data.copy()
ed_encoded[ordinal_cat[0]] = pd.Categorical(ed_encoded[ordinal_cat[0]], categories=ed_map[ordinal_cat[0]], ordered=True).codes
occ_encoded=ed_encoded.copy()
occ_encoded[ordinal_cat[1]] = pd.Categorical(occ_encoded[ordinal_cat[1]], categories=occ_map[ordinal_cat[1]], ordered=True).codes
mem_encoded=occ_encoded.copy()
mem_encoded[ordinal_cat[2]] = pd.Categorical(mem_encoded[ordinal_cat[2]], categories=mem_map[ordinal_cat[2]], ordered=True).codes
inc_encoded=mem_encoded.copy()
inc_encoded[ordinal_cat[3]] = pd.Categorical(inc_encoded[ordinal_cat[3]], categories=inc_map[ordinal_cat[3]], ordered=True).codes
store_encoded=inc_encoded.copy()
store_encoded[ordinal_cat[4]] = pd.Categorical(store_encoded[ordinal_cat[4]], categories=store_map[ordinal_cat[4]], ordered=True).codes

nominal_cat = ['food_category', 'food_department', 'food_family', 'promotion_name',
       'sales_country', 'marital_status', 'gender', 'houseowner', 'brand_name',
       'store_city', 'store_state', 'media_type']
dic_2 = {}
for i in range(len(nominal_cat)):
    dic_2[nominal_cat[i]] = store_encoded[nominal_cat[i]].unique()




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
    selected = option_menu(menu_title=None,options=["Playground", "EDA", "Preprocess", "Prediction"],
                                icons=['house', "graph-up-arrow","cloud-upload","signal"],menu_icon="cast",
                                default_index=2,orientation="horizontal",styles={"nav-link":
                                                                                  {"text-align": "left","--hover-color": "#eee",}
                                                                                  ,"nav-link-selected": 
                                                                                  {"background-color": "green"}})
    if selected == "EDA":
        st.switch_page(r"pages/eda.py")
    if selected == "Playground":
        st.switch_page(r"app.py")
    if selected == "Prediction":
        st.switch_page(r"pages/prediction.py")


with st.container(border=True):
    if "disabled1" not in st.session_state:
        st.session_state.disabled1 = False

    st.checkbox("Select the box to show original data", key="disabled1")
    if st.session_state.disabled1: 
        st.dataframe(train_data, width=1400, height=150) 

        with st.container(border=True): ### Null Values
            if "disabled1_1" not in st.session_state:
                st.session_state.disabled1_1 = False

            st.checkbox("Will Null values make my day pathetic ? ", key="disabled1_1")
            if st.session_state.disabled1_1: 
                st.dataframe(pd.DataFrame(train_data.isna().sum()).T, width=1400, height=50, hide_index=True) 
                st.dataframe(pd.DataFrame({"Inference" : ["The data doesn't have any missing value.Pretty unrealisitic times!"]}), width=1400, height=50, hide_index=True)
        
        with st.container(border=True):   ### Repeat columns
            if "disabled1_2" not in st.session_state:
                st.session_state.disabled1_2 = False

            st.checkbox("Repetitive column is just a bone for overfitting !", key="disabled1_2")
            if st.session_state.disabled1_2:
                left_rep, right_rep = st.columns(2) 
                with left_rep:
                    st.dataframe(train_data.iloc[:10, 12:20].style.set_properties(subset=["avg_cars_at home(approx).1","avg_cars_at home(approx)"], **{'background-color': 'yellow'}),
                                    width=1400,
                                    height=200,
                                        hide_index=True) 
                    st.dataframe(pd.DataFrame({"Inference" : ["The 'avg_cars_at home(approx).1' is a complete duplicate of 'avg_cars_at home(approx)'"]}), width=1400, height=50, hide_index=True)
                
                with right_rep:  
                    st.button("Remove", key="rem_repeat", use_container_width=True) #Created button to view description of data
                    if st.session_state.rem_repeat:
                        with st.spinner("Removing...", show_time=True):
                            time.sleep(2)
                        st.success("Removed!!!")
                        st.dataframe(costs_data.iloc[:10, 12:20].style.set_properties(subset=["avg_cars_at home(approx)"], **{'background-color': 'blue'}),
                                    width=1400,
                                    height=200,
                                        hide_index=True)
            
        with st.container(border=True):   ### Clean categories
            if "disabled1_3" not in st.session_state:
                st.session_state.disabled1_3 = False

            st.checkbox("Cleaner columns make preprocessing easy !", key="disabled1_3")
            if st.session_state.disabled1_3:
                left_rep, right_rep = st.columns(2) 
                with left_rep:
                    left, right = st.columns(2)
                    left.dataframe(year_inc,width=1400,height=200,hide_index=True) 
                    right.dataframe(media_tp,width=1400,height=200,hide_index=True) 
                    st.dataframe(pd.DataFrame({"Inference" : ["The 'yearly income' column has $ in categories. Cleaning $ up for simplicity.", "The 'media_type' column has repetitive categories which can be merged into one of a kind."]}),
                                  width=1400, height=100, hide_index=True)
                
                with right_rep:  
                    st.button("Clean", key="clean_col", use_container_width=True) #Created button to view description of data
                    if st.session_state.clean_col:
                        with st.spinner("Cleaning...", show_time=True):
                            time.sleep(2)
                        st.success("Cleaning done!!!")

                        left, right = st.columns(2)
                        left.dataframe(cleaned_year_inc,width=1400,height=200,hide_index=True) 
                        right.dataframe(cleaned_media_tp,width=1400,height=200,hide_index=True) 
                        st.dataframe(pd.DataFrame({"Changes" : ["The 'yearly income' column has no $ in categories", "Two new categories, Daily Media and Sunday media have replaced repetitive categories."]}),
                                  width=1400, height=100, hide_index=True)
            
with st.container(border=True):
    if "disabled2" not in st.session_state:
        st.session_state.disabled2 = False

    st.checkbox("Encoding the non-encoded ORDINAL categorical variables", key="disabled2")
    if st.session_state.disabled2: 
        st.dataframe(cleaned_data, width=1400, height=150) 

        with st.container(border=True): ### Extract ordinal
            if "disabled2_1" not in st.session_state:
                st.session_state.disabled2_1 = False

            st.checkbox("Extract the ordinal categorical variables !", key="disabled2_1")
            if st.session_state.disabled2_1: 
                col1, col2, col3, col4, col5 = st.columns(5)
                 
                col1.dataframe(pd.DataFrame({list(dic.keys())[0] : list(dic.values())[0]}), width=1400, height=150, hide_index=True) 
                col1.dataframe(pd.DataFrame({'No. of categories':[len(list(dic.values())[0])]}),hide_index=True, use_container_width=True)
                    
                col2.dataframe(pd.DataFrame({list(dic.keys())[1] : list(dic.values())[1]}), width=1400, height=150, hide_index=True) 
                col2.dataframe(pd.DataFrame({'No. of categories':[len(list(dic.values())[1])]}),hide_index=True, use_container_width=True)
                    
                col3.dataframe(pd.DataFrame({list(dic.keys())[2] : list(dic.values())[2]}), width=1400, height=150, hide_index=True) 
                col3.dataframe(pd.DataFrame({'No. of categories':[len(list(dic.values())[2])]}),hide_index=True, use_container_width=True)
                    
                col4.dataframe(pd.DataFrame({list(dic.keys())[3] : list(dic.values())[3]}), width=1400, height=150, hide_index=True) 
                col4.dataframe(pd.DataFrame({'No. of categories':[len(list(dic.values())[3])]}),hide_index=True, use_container_width=True)
                    
                col5.dataframe(pd.DataFrame({list(dic.keys())[4] : list(dic.values())[4]}), width=1400, height=150, hide_index=True) 
                col5.dataframe(pd.DataFrame({'No. of categories':[len(list(dic.values())[4])]}),hide_index=True, use_container_width=True)
        


        with st.container(border=True): ### Encode education
            if "disabled2_2" not in st.session_state:
                st.session_state.disabled2_2 = False

            st.checkbox("Encode the 'education' column !", key="disabled2_2")
            if st.session_state.disabled2_2:
                left, right = st.columns(2)
                with left:
                    st.dataframe(ed_map, use_container_width=True, hide_index=True)
                    st.dataframe(pd.DataFrame({"Inference" : ["The 'education' column has 5 categories", "The order of categories are preserved as 'partial high school' is mapped with value 0 while 'Graduate degree' with value 4"]}),
                                   height=100, hide_index=True, use_container_width=True)
                with right:
                    st.button("Encode", key="clean_col", use_container_width=True) #Created button to view description of data
                    if st.session_state.clean_col:
                        with st.spinner("Encoding...", show_time=True):
                            time.sleep(2)
                        st.success("Encoded!")
                        st.dataframe(ed_encoded.iloc[:10,9:14].style.set_properties(subset=["education"], **{'background-color': 'blue'}),
                                     width=1400, height=200, hide_index=True)
            
        with st.container(border=True): ### Encoded occupation
            if "disabled2_3" not in st.session_state:
                st.session_state.disabled2_3 = False

            st.checkbox("Encode the 'occupation' column !", key="disabled2_3")
            if st.session_state.disabled2_3:
                left, right = st.columns(2)
                with left:
                    st.dataframe(occ_map, use_container_width=True, hide_index=True)
                    st.dataframe(pd.DataFrame({"Inference" : ["The 'occupation' column has 5 categories", "The order of categories are preserved as 'manual' is mapped with value 0 while 'management' with value 4"]}),
                                   height=100, hide_index=True, use_container_width=True)
                with right:
                    st.button("Encode", key="clean_col", use_container_width=True) #Created button to view description of data
                    if st.session_state.clean_col:
                        with st.spinner("Encoding...", show_time=True):
                            time.sleep(2)
                        st.success("Encoded!!!")
                        st.dataframe(occ_encoded.iloc[:10,10:15].style.set_properties(subset=["occupation"], **{'background-color': 'blue'}),
                                     width=1400, height=200, hide_index=True)    
                        
        
        with st.container(border=True): ### Encoded membercard
            if "disabled2_4" not in st.session_state:
                st.session_state.disabled2_4 = False

            st.checkbox("Encode the 'member_card' column !", key="disabled2_4")
            if st.session_state.disabled2_4:
                left, right = st.columns(2)
                with left:
                    st.dataframe(mem_map, use_container_width=True, hide_index=True)
                    st.dataframe(pd.DataFrame({"Inference" : ["The 'member-card' column has 4 categories", "The order of categories are preserved as 'normal' is mapped with value 0 while 'golden' with value 3"]}),
                                   height=100, hide_index=True, use_container_width=True)
                with right:
                    st.button("Encode", key="clean_col", use_container_width=True) #Created button to view description of data
                    if st.session_state.clean_col:
                        with st.spinner("Encoding...", show_time=True):
                            time.sleep(2)
                        st.success("Encoded!!!")
                        st.dataframe(mem_encoded.iloc[:10,10:15].style.set_properties(subset=["member_card"], **{'background-color': 'blue'}),
                                     width=1400, height=200, hide_index=True)    
        
        with st.container(border=True): ### Encoded yearly income
            if "disabled2_5" not in st.session_state:
                st.session_state.disabled2_5 = False

            st.checkbox("Encode the 'yearly_income' column !", key="disabled2_5")
            if st.session_state.disabled2_5:
                left, right = st.columns(2)
                with left:
                    st.dataframe(inc_map, use_container_width=True, hide_index=True)
                    st.dataframe(pd.DataFrame({"Inference" : ["The 'yearly_income' column has 8 categories", "The order of categories are preserved as '10k-30k' is mapped with value 0 while '150k+' with value 7"]}),
                                   height=100, hide_index=True, use_container_width=True)
                with right:
                    st.button("Encode ", key="clean_col", use_container_width=True) #Created button to view description of data
                    if st.session_state.clean_col:
                        with st.spinner("Encoding...", show_time=True):
                            time.sleep(2)
                        st.success("Encoded!!!")
                        st.dataframe(inc_encoded.iloc[:10,14:19].style.set_properties(subset=["avg. yearly_income"], **{'background-color': 'blue'}),
                                     width=1400, height=200, hide_index=True)  
                        
        
        with st.container(border=True): ### Encoded store_type
            if "disabled2_6" not in st.session_state:
                st.session_state.disabled2_6 = False

            st.checkbox("Encode the 'store_type' column !", key="disabled2_6")
            if st.session_state.disabled2_6:
                left, right = st.columns(2)
                with left:
                    st.dataframe(store_map, use_container_width=True, hide_index=True)
                    st.dataframe(pd.DataFrame({"Inference" : ["The 'store_type' column has 5 categories", "The order of categories are preserved as '' is mapped with value 0 while '150k+' with value 7"]}),
                                   height=100, hide_index=True, use_container_width=True)
                with right:
                    st.button("Encode", key="clean_col", use_container_width=True) #Created button to view description of data
                    if st.session_state.clean_col:
                        with st.spinner("Encoding...", show_time=True):
                            time.sleep(2)
                        st.success("Encoded!!!")
                        st.dataframe(store_encoded.iloc[:10,23:28].style.set_properties(subset=["store_type"], **{'background-color': 'blue'}),
                                     width=1400, height=200, hide_index=True)  


with st.container(border=True):
    if "disabled3" not in st.session_state:
        st.session_state.disabled3 = False

    st.checkbox("Encoding the non-encoded NOMINAL categorical variables", key="disabled3")
    if st.session_state.disabled3: 
        st.dataframe(store_encoded, width=1400, height=150)
    
        with st.container(border=True): ### Extract ordinal
            if "disabled3_1" not in st.session_state:
                st.session_state.disabled3_1 = False

            st.checkbox("Extract the nominal categorical variables !", key="disabled3_1")
            if st.session_state.disabled3_1: 
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                col1.dataframe(pd.DataFrame({list(dic_2.keys())[0] : list(dic_2.values())[0]}), width=1400, height=150, hide_index=True) 
                col1.dataframe(pd.DataFrame({'No. of categories':[len(list(dic_2.values())[0])]}),hide_index=True, use_container_width=True)
                col1.dataframe(pd.DataFrame({list(dic_2.keys())[6] : list(dic_2.values())[6]}), width=1400, height=150, hide_index=True) 
                col1.dataframe(pd.DataFrame({'No. of categories':[len(list(dic_2.values())[6])]}),hide_index=True, use_container_width=True)
                    
                col2.dataframe(pd.DataFrame({list(dic_2.keys())[1] : list(dic_2.values())[1]}), width=1400, height=150, hide_index=True) 
                col2.dataframe(pd.DataFrame({'No. of categories':[len(list(dic_2.values())[1])]}),hide_index=True, use_container_width=True)
                col2.dataframe(pd.DataFrame({list(dic_2.keys())[7] : list(dic_2.values())[7]}), width=1400, height=150, hide_index=True) 
                col2.dataframe(pd.DataFrame({'No. of categories':[len(list(dic_2.values())[7])]}),hide_index=True, use_container_width=True)
                    
                col3.dataframe(pd.DataFrame({list(dic_2.keys())[2] : list(dic_2.values())[2]}), width=1400, height=150, hide_index=True) 
                col3.dataframe(pd.DataFrame({'No. of categories':[len(list(dic_2.values())[2])]}),hide_index=True, use_container_width=True)
                col3.dataframe(pd.DataFrame({list(dic_2.keys())[8] : list(dic_2.values())[8]}), width=1400, height=150, hide_index=True) 
                col3.dataframe(pd.DataFrame({'No. of categories':[len(list(dic_2.values())[8])]}),hide_index=True, use_container_width=True)
                    
                col4.dataframe(pd.DataFrame({list(dic_2.keys())[3] : list(dic_2.values())[3]}), width=1400, height=150, hide_index=True) 
                col4.dataframe(pd.DataFrame({'No. of categories':[len(list(dic_2.values())[3])]}),hide_index=True, use_container_width=True)
                col4.dataframe(pd.DataFrame({list(dic_2.keys())[9] : list(dic_2.values())[9]}), width=1400, height=150, hide_index=True) 
                col4.dataframe(pd.DataFrame({'No. of categories':[len(list(dic_2.values())[9])]}),hide_index=True, use_container_width=True)
                    
                col5.dataframe(pd.DataFrame({list(dic_2.keys())[4] : list(dic_2.values())[4]}), width=1400, height=150, hide_index=True) 
                col5.dataframe(pd.DataFrame({'No. of categories':[len(list(dic_2.values())[4])]}),hide_index=True, use_container_width=True)
                col5.dataframe(pd.DataFrame({list(dic_2.keys())[10] : list(dic_2.values())[10]}), width=1400, height=150, hide_index=True) 
                col5.dataframe(pd.DataFrame({'No. of categories':[len(list(dic_2.values())[10])]}),hide_index=True, use_container_width=True)

                col6.dataframe(pd.DataFrame({list(dic_2.keys())[5] : list(dic_2.values())[5]}), width=1400, height=150, hide_index=True) 
                col6.dataframe(pd.DataFrame({'No. of categories':[len(list(dic_2.values())[5])]}),hide_index=True, use_container_width=True)
                col6.dataframe(pd.DataFrame({list(dic_2.keys())[11] : list(dic_2.values())[11]}), width=1400, height=150, hide_index=True) 
                col6.dataframe(pd.DataFrame({'No. of categories':[len(list(dic_2.values())[11])]}),hide_index=True, use_container_width=True)
                
                            
        with st.container(border=True): ### Extract ordinal
            if "disabled3_2" not in st.session_state:
                st.session_state.disabled3_2 = False

            st.checkbox("Encode the 'food_category' column !", key="disabled3_2")
            if st.session_state.disabled3_2:
                left, right = st.columns(2)
                with left:
                    st.dataframe(ed_map, use_container_width=True, hide_index=True)
                    st.dataframe(pd.DataFrame({"Inference" : ["The 'education' column has 5 categories", "The order of categories are preserved as 'partial high school' is mapped with value 0 while 'Graduate degree' with value 4"]}),
                                   height=100, hide_index=True, use_container_width=True)
                with right:
                    st.button("Encode", key="clean_col", use_container_width=True) #Created button to view description of data
                    if st.session_state.clean_col:
                        with st.spinner("Encoding...", show_time=True):
                            time.sleep(2)
                        st.success("Encoded!")
                    
                
                    
                
        