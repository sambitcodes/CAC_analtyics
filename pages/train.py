# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from scipy.stats import skew, kurtosis
# import plotly.express as px
# import plotly.figure_factory as ff
# from streamlit_extras.switch_page_button import switch_page
# from streamlit_option_menu import option_menu

# st.set_page_config(page_title = "Train",page_icon="🦈",layout="wide",initial_sidebar_state="collapsed")

# # Load the dataset
# costs_data = pd.read_csv(r'cac_dataset/customer_acquisition_costs.csv')
# costs_data = costs_data.rename(columns={'avg. yearly_income': 'yearly_income'})
# costs_data = costs_data.drop(["avg_cars_at home(approx).1"], axis=1)
# costs_columns = list(costs_data.columns)

# #float-type features
# float_features = costs_data.select_dtypes(exclude="object").columns
# float_categories = costs_data[float_features].nunique()
# float_cat_df = pd.DataFrame({"float_Features": float_features, "Count of each categories":list(float_categories)})

# #object-type features
# object_features = costs_data.select_dtypes(include="object").columns
# object_categories = costs_data[object_features].nunique()
# object_cat_df = pd.DataFrame({"object_Features": object_features, "Count of each categories":list(object_categories)})



# numerical_features = ['store_sales(in millions)', 'store_cost(in millions)','SRP','gross_weight', 'net_weight','cost']
# len_num = len(numerical_features)
# numerical_categories = costs_data[numerical_features].nunique()
# numerical_df = pd.DataFrame({"Features": numerical_features, "Count of each categories":list(numerical_categories)})


# categorical_features = [col for col in costs_columns if col not in numerical_features]
# len_cat = len(categorical_features)
# categorical_categories = costs_data[categorical_features].nunique()
# categorical_df = pd.DataFrame({"Features": categorical_features, "Count of each categories":list(categorical_categories)})




# # st.metric(label="Temperature", value="70 °F", delta="1.2 °F")
# with st.container(border=True):
#     left,right = st.columns(2, vertical_alignment="center")
#     right.image(r"elements/pictures/cfm_logo.gif",use_container_width=True)
#     left.title(":blue[_CAC_] Analysis :chart:")
#     left.markdown("#### Detailed EDA on Customer Acquistion Costs (FOOD-MART)")
#     # left.subheader("Detailed EDA on Customer Acquistion Costs (FOOD-MART)")

# with st.container():
#     selected = option_menu(menu_title=None,options=["Playground", "EDA", "Train", "Prediction"],
#                                 icons=['house', "graph-up-arrow","cloud-upload","signal"],menu_icon="cast",
#                                 default_index=2,orientation="horizontal",styles={"nav-link":
#                                                                                   {"text-align": "left","--hover-color": "#eee",}
#                                                                                   ,"nav-link-selected": 
#                                                                                   {"background-color": "green"}})
#     if selected == "EDA":
#         st.switch_page(r"pages/eda.py")
#     if selected == "Playground":
#         st.switch_page(r"app.py")
#     if selected == "Prediction":
#         st.switch_page(r"pages/prediction.py")




