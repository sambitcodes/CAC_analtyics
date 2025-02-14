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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_extras.switch_page_button import switch_page
from streamlit_option_menu import option_menu
import addfips


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


#numerical features
numerical_features = ['store_sales(in millions)', 'store_cost(in millions)','SRP','gross_weight', 'net_weight','cost']
len_num = len(numerical_features)
numerical_categories = costs_data[numerical_features].nunique()
numerical_df = pd.DataFrame({"Features": numerical_features, "Count of each categories":list(numerical_categories)})

#categorical features
categorical_features = [col for col in costs_columns if col not in numerical_features]
len_cat = len(categorical_features)
categorical_categories = costs_data[categorical_features].nunique()
categorical_df = pd.DataFrame({"Features": categorical_features, "Count of each categories":list(categorical_categories)})




# st.metric(label="Temperature", value="70 °F", delta="1.2 °F")
#heading
with st.container(border=True):
    left,right = st.columns(2, vertical_alignment="center")
    right.image(r"elements/pictures/cfm_logo.gif",use_container_width=True)
    left.title(":blue[_CAC_] Analysis :chart:")
    left.markdown("#### Detailed EDA on Customer Acquistion Costs (FOOD-MART)")
    # left.subheader("Detailed EDA on Customer Acquistion Costs (FOOD-MART)")

#page tabs
with st.container():
    selected = option_menu(menu_title=None,options=["Playground", "EDA", "Preprocess", "Prediction"],
                                icons=['house', "graph-up-arrow","cloud-upload","signal"],menu_icon="cast",
                                default_index=1,orientation="horizontal",styles={"nav-link":
                                                                                  {"text-align": "left","--hover-color": "#eee",}
                                                                                  ,"nav-link-selected": 
                                                                                  {"background-color": "green"}})
    if selected == "Playground":
        st.switch_page(r"app.py")
    if selected == "Preprocess":
        st.switch_page(r"pages/preprocess.py")
    if selected == "Prediction":
        st.switch_page(r"pages/prediction.py")

#top 10 food
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
        sales_by_product = prod_sales['unit_sales(in millions)'].mean().reset_index()
        sales_by_product = sales_by_product.sort_values(by='unit_sales(in millions)',ascending=False)[:10]
        st.dataframe(sales_by_product.T, use_container_width=True)
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
        

#top 10 food with facilities
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
        sales_by_product = prod_sales['unit_sales(in millions)'].mean().reset_index()
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


#member-card
with st.container(border=True):
    if "disabled_three" not in st.session_state:
        st.session_state.disabled_three = False

    st.checkbox(":blue[_Exploration 3_] || Segmentation of membership-card holders.", key="disabled_three")
    if st.session_state.disabled_three:
 
        revenue_data = costs_data.copy()
        revenue_data["revenue(in millions)"] = costs_data['store_sales(in millions)']-costs_data['store_cost(in millions)']
        card_sales = revenue_data.groupby('member_card')
        revenue_by_card = card_sales["revenue(in millions)"].sum().reset_index().sort_values(by="revenue(in millions)",ascending=False)
        # cost_by_card = card_sales[""]
        out_inference = "The {} card holders generate the most revenue of {} million $".format(revenue_by_card.iloc[0][0],
                                                                                                np.round(revenue_by_card.iloc[0][1],2))
        inference_dataframe = pd.DataFrame({"Inference": [out_inference,"Targeting promotions at this group could help generate higher returns."]})
        st.dataframe(inference_dataframe, width=1400, height=100,hide_index=True, use_container_width=True)
        title_s = "member_card vs store_revenue"
        fig = px.line(revenue_by_card,x="member_card", y="revenue(in millions)",title = title_s, markers=True)
        fig.update_traces(line_color='violet', line_width=2)
        st.plotly_chart(fig)


#store-cost
with st.container(border=True):
    if "disabled_four" not in st.session_state:
        st.session_state.disabled_four = False

    st.checkbox(":blue[_Exploration 4_] || Insights on food categories,food department and food family that cost the stores most.", key="disabled_four")
    if st.session_state.disabled_four:

        base_features = ["food_category","food_department","food_family"]
        option_feature = st.selectbox("Select one of the following feature.",base_features,key="feature_option")

        store_cost = costs_data.groupby(option_feature)
        cost_by_category = store_cost['store_cost(in millions)'].sum().reset_index()
        sales_by_category = store_cost['store_sales(in millions)'].sum().reset_index()
        
        if option_feature != "food_family":
            cost_by_category = cost_by_category.sort_values(by='store_cost(in millions)',ascending=False)[:10]
            sales_by_category = sales_by_category.sort_values(by='store_sales(in millions)',ascending=False)[:10]
        else:
            cost_by_category = cost_by_category.sort_values(by='store_cost(in millions)',ascending=False)
            sales_by_category = sales_by_category.sort_values(by='store_sales(in millions)',ascending=False)
        
        revenue_by_category = sales_by_category.join(cost_by_category.set_index(option_feature), on=option_feature)
        revenue_by_category["revenue(in millions)"] = revenue_by_category["store_sales(in millions)"] - revenue_by_category["store_cost(in millions)"]

        finance_features = {0:"store_cost(in millions)",
                            1:"store_sales(in millions)",
                            2:"revenue(in millions)"}
        option_finance = st.pills("Select which insight you wanna look at.",finance_features.keys(),format_func=lambda option: finance_features[option],selection_mode="multi",key="finance_option")
        if len(option_finance)==1:
            primary_option = finance_features[option_finance[0]]
            title_s = "{} of a store for the {}".format(primary_option,option_feature)
            fig = px.line(revenue_by_category,x=option_feature, y=primary_option, title = title_s, markers=True)
            fig.update_traces(line_color='yellow', line_width=2)
            st.plotly_chart(fig)
        
        elif len(option_finance)==2:
            primary_option = finance_features[option_finance[0]]
            secondary_option = finance_features[option_finance[1]]
            x = revenue_by_category[option_feature]
            y1 = revenue_by_category[primary_option]
            y2 = revenue_by_category[secondary_option]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y1,
                    mode='lines+markers',line_color='red',
                    name=primary_option))
            fig.add_trace(go.Scatter(x=x, y=y2,
                    mode='lines+markers',line_color='yellow',
                    name=secondary_option))
            st.plotly_chart(fig)
        
        elif len(option_finance)==3:
            primary_option = finance_features[option_finance[0]]
            secondary_option = finance_features[option_finance[1]]
            tertiary_option = finance_features[option_finance[2]]
            x = revenue_by_category[option_feature]
            y1 = revenue_by_category[primary_option]
            y2 = revenue_by_category[secondary_option]
            y3 = revenue_by_category[tertiary_option]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y1,
                    mode='lines+markers',line_color='red',
                    name=primary_option))
            fig.add_trace(go.Scatter(x=x, y=y2,
                    mode='lines+markers',line_color='yellow',
                    name=secondary_option))
            fig.add_trace(go.Scatter(x=x, y=y3,
                    mode='lines+markers',line_color='violet',
                    name=tertiary_option))
            st.plotly_chart(fig)


            cost_inference = "The cost of maintaining a store is highest for {} = {} million $.".format(cost_by_category.iloc[0][0], np.round(cost_by_category.iloc[0][1],2))
            sale_inference = "This makes sense because {} also witnesses the highest sales of {} million $".format(sales_by_category.iloc[0][0], np.round(sales_by_category.iloc[0][1],2))
            rev_inference = "And thus {} generates the highest revenue of {} million $".format(revenue_by_category.iloc[0][0], np.round(revenue_by_category.iloc[0][3],2))
            inference_dataframe = pd.DataFrame({"Inference": [cost_inference,sale_inference,rev_inference]})
            st.dataframe(inference_dataframe, width=1400, height=140,hide_index=True, use_container_width=True)


#country-wise
with st.container(border=True):
    if "disabled_five" not in st.session_state:
        st.session_state.disabled_five = False

    st.checkbox(":blue[_Exploration 5_] || Insights on country wise contibution in store sales, costs and revenues.", key="disabled_five")
    if st.session_state.disabled_five:

        country_data = costs_data.groupby('sales_country')
        cost_by_country = country_data['store_cost(in millions)'].sum().reset_index().sort_values(by='store_cost(in millions)',ascending=False)
        sales_by_country = country_data['store_sales(in millions)'].sum().reset_index().sort_values(by='store_sales(in millions)',ascending=False)
        revenue_by_country = sales_by_country.join(cost_by_country.set_index('sales_country'), on='sales_country')
        revenue_by_country["revenue(in millions)"] = revenue_by_country["store_sales(in millions)"] - revenue_by_country["store_cost(in millions)"]
        st.dataframe(revenue_by_country,use_container_width=True, hide_index=True)

        labels = revenue_by_country["sales_country"]
        values_cost = revenue_by_country["store_cost(in millions)"]
        values_sale = revenue_by_country["store_sales(in millions)"]
        values_rev = revenue_by_country["revenue(in millions)"]
        fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}]])
        fig.add_trace(go.Pie(labels=labels, values=values_cost, name="Store Costs"),1, 1)
        fig.add_trace(go.Pie(labels=labels, values=values_sale, name="Store Sales"),1, 2)
        fig.add_trace(go.Pie(labels=labels, values=values_rev, name="Store Revenue"),1, 3)
        fig.update_traces(hole=.4, hoverinfo="label+value+name")
        fig.update_layout(title_text="Country-Wise Stores Performance",
                            annotations=[dict(text='COST', x=sum(fig.get_subplot(1, 1).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center"),
                                        dict(text='SALES', x=sum(fig.get_subplot(1, 2).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center"),
                                        dict(text='REV', x=sum(fig.get_subplot(1, 3).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center")])
        st.plotly_chart(fig)



#gender, marital, children
with st.container(border=True):
    if "disabled_six" not in st.session_state:
        st.session_state.disabled_six = False

    st.checkbox(":blue[_Exploration 6_] || Gender, Marriage, childrens changing trend in sales ?", key="disabled_six")
    if st.session_state.disabled_six:

        base_features = ["gender","marital_status","num_children_at_home"]
        option_feature = st.selectbox("Select one of the following feature.",base_features,key="feature_option")

        store_cost = costs_data.groupby(option_feature)
        cost_by_category = store_cost['store_cost(in millions)'].sum().reset_index().sort_values(by='store_cost(in millions)',ascending=False)
        sales_by_category = store_cost['store_sales(in millions)'].sum().reset_index().sort_values(by='store_sales(in millions)',ascending=False) 
        revenue_by_category = sales_by_category.join(cost_by_category.set_index(option_feature), on=option_feature)
        revenue_by_category["revenue(in millions)"] = revenue_by_category["store_sales(in millions)"] - revenue_by_category["store_cost(in millions)"]
        
        if option_feature == "num_children_at_home":
            finance_features = {0:"store_cost(in millions)",
                                1:"store_sales(in millions)",
                                2:"revenue(in millions)"}
            option_finance = st.pills("Select which insight you wanna look at.",finance_features.keys(),format_func=lambda option: finance_features[option],selection_mode="multi",key="finance_option")
            if len(option_finance)==1:
                primary_option = finance_features[option_finance[0]]
                title_s = "{} of a store for the {}".format(primary_option,option_feature)
                fig = px.line(revenue_by_category,x=option_feature, y=primary_option, title = title_s, markers=True)
                fig.update_traces(line_color='yellow', line_width=2)
                st.plotly_chart(fig)
            
            elif len(option_finance)==2:
                primary_option = finance_features[option_finance[0]]
                secondary_option = finance_features[option_finance[1]]
                x = revenue_by_category[option_feature]
                y1 = revenue_by_category[primary_option]
                y2 = revenue_by_category[secondary_option]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y1,
                        mode='lines+markers',line_color='red',
                        name=primary_option))
                fig.add_trace(go.Scatter(x=x, y=y2,
                        mode='lines+markers',line_color='yellow',
                        name=secondary_option))
                st.plotly_chart(fig)
            
            elif len(option_finance)==3:
                primary_option = finance_features[option_finance[0]]
                secondary_option = finance_features[option_finance[1]]
                tertiary_option = finance_features[option_finance[2]]
                x = revenue_by_category[option_feature]
                y1 = revenue_by_category[primary_option]
                y2 = revenue_by_category[secondary_option]
                y3 = revenue_by_category[tertiary_option]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y1,
                        mode='lines+markers',line_color='red',
                        name=primary_option))
                fig.add_trace(go.Scatter(x=x, y=y2,
                        mode='lines+markers',line_color='yellow',
                        name=secondary_option))
                fig.add_trace(go.Scatter(x=x, y=y3,
                        mode='lines+markers',line_color='violet',
                        name=tertiary_option))
                st.plotly_chart(fig)
        
        else:
            labels = revenue_by_category[option_feature]
            values_cost = revenue_by_category["store_cost(in millions)"]
            values_sale = revenue_by_category["store_sales(in millions)"]
            values_rev = revenue_by_category["revenue(in millions)"]
            fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}]])
            fig.add_trace(go.Pie(labels=labels, values=values_cost, name="Store Costs"),1, 1)
            fig.add_trace(go.Pie(labels=labels, values=values_sale, name="Store Sales"),1, 2)
            fig.add_trace(go.Pie(labels=labels, values=values_rev, name="Store Revenue"),1, 3)
            fig.update_traces(hole=.4, hoverinfo="label+value+name")
            fig.update_layout(title_text="Stores Performance for {}".format(option_feature),
                                annotations=[dict(text='COST', x=sum(fig.get_subplot(1, 1).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center"),
                                            dict(text='SALES', x=sum(fig.get_subplot(1, 2).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center"),
                                            dict(text='REV', x=sum(fig.get_subplot(1, 3).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center")])
            st.plotly_chart(fig)
        

        gender_inference = "The sale performance remains almost same for the given genders."
        marital_inference = "Being married doesn't have visible affect on store performance"
        child_inference = "The scale of revenue sees to shift downwards for parents with more children at home than customers with no children.Offers could be made for parents of 1 or more children."
        inference_dataframe = pd.DataFrame({"Inference": [gender_inference,marital_inference,child_inference]})
        st.dataframe(inference_dataframe, width=1400, height=140,hide_index=True, use_container_width=True)



#house_owner, cars
with st.container(border=True):
    if "disabled_seven" not in st.session_state:
        st.session_state.disabled_seven = False

    st.checkbox(":blue[_Exploration 7_] || Does owning a car or a house have any affect on sales?", key="disabled_seven")
    if st.session_state.disabled_seven:

        base_features = ["houseowner","avg_cars_at home(approx)", "Together"]
        option_feature = st.selectbox("Select one of the following feature.",base_features,key="feature_option2")
        
        if option_feature == "Together":

            asset_data = costs_data.groupby(['houseowner','avg_cars_at home(approx)'])
            sales_by_asset = asset_data['store_sales(in millions)'].sum().reset_index().sort_values(by='store_sales(in millions)',ascending=False)
            costs_by_asset = asset_data['store_cost(in millions)'].sum().reset_index().sort_values(by='store_cost(in millions)',ascending=False)
            revenue_by_asset = sales_by_asset.join(costs_by_asset.set_index(['houseowner','avg_cars_at home(approx)']), on=['houseowner','avg_cars_at home(approx)'])
            revenue_by_asset["revenue(in millions)"] = revenue_by_asset["store_sales(in millions)"] - revenue_by_asset["store_cost(in millions)"]
            x = "avg_cars_at home(approx)"
            y = "revenue(in millions)"
            z = "houseowner"
            fig = px.bar(revenue_by_asset,x=z,y=y,color=x, title = "Revenue by customers with car and house.")
            st.plotly_chart(fig)



        else:

            store_cost = costs_data.groupby(option_feature)
            cost_by_category = store_cost['store_cost(in millions)'].sum().reset_index().sort_values(by='store_cost(in millions)',ascending=False)
            sales_by_category = store_cost['store_sales(in millions)'].sum().reset_index().sort_values(by='store_sales(in millions)',ascending=False) 
            revenue_by_category = sales_by_category.join(cost_by_category.set_index(option_feature), on=option_feature)
            revenue_by_category["revenue(in millions)"] = revenue_by_category["store_sales(in millions)"] - revenue_by_category["store_cost(in millions)"]
            labels = revenue_by_category[option_feature]
            values_cost = revenue_by_category["store_cost(in millions)"]
            values_sale = revenue_by_category["store_sales(in millions)"]
            values_rev = revenue_by_category["revenue(in millions)"]
            fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}]])
            fig.add_trace(go.Pie(labels=labels, values=values_cost, name="Store Costs"),1, 1)
            fig.add_trace(go.Pie(labels=labels, values=values_sale, name="Store Sales"),1, 2)
            fig.add_trace(go.Pie(labels=labels, values=values_rev, name="Store Revenue"),1, 3)
            fig.update_traces(hole=.4, hoverinfo="label+value+name")
            fig.update_layout(title_text="Stores Performance for {}".format(option_feature),
                                    annotations=[dict(text='COST', x=sum(fig.get_subplot(1, 1).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center"),
                                                dict(text='SALES', x=sum(fig.get_subplot(1, 2).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center"),
                                                dict(text='REV', x=sum(fig.get_subplot(1, 3).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center")])
            st.plotly_chart(fig)
            

        house_inference = "House owners are more likely to shop at these stores."
        car_inference = "Revenue increases with increase in number of cars till 3 cars and then decreases. May be they choose other stores."
        inference_dataframe = pd.DataFrame({"Inference": [house_inference,car_inference]})
        st.dataframe(inference_dataframe, width=1400, height=100,hide_index=True, use_container_width=True)

#education and occupation
with st.container(border=True):
    if "disabled_eight" not in st.session_state:
        st.session_state.disabled_eight = False

    st.checkbox(":blue[_Exploration 8_] || Does occupation or educational background moves sales?", key="disabled_eight")
    if st.session_state.disabled_eight:

        base_features = ["occupation","education", "Together"]
        option_feature = st.selectbox("Select one of the following feature.",base_features,key="feature_option3")
        
        if option_feature == "Together":
            asset_data = costs_data.groupby(['occupation','education'])
            sales_by_asset = asset_data['store_sales(in millions)'].sum().reset_index().sort_values(by='store_sales(in millions)',ascending=False)
            costs_by_asset = asset_data['store_cost(in millions)'].sum().reset_index().sort_values(by='store_cost(in millions)',ascending=False)
            revenue_by_asset = sales_by_asset.join(costs_by_asset.set_index(['occupation','education']), on=['occupation','education'])
            revenue_by_asset["revenue(in millions)"] = revenue_by_asset["store_sales(in millions)"] - revenue_by_asset["store_cost(in millions)"]
            x = "occupation"
            y = "revenue(in millions)"
            z = "education"
            fig = px.bar(revenue_by_asset,x=z,y=y,color=x, title = "Revenue by customers with different occupation and education.")
            st.plotly_chart(fig)



        else:
            store_cost = costs_data.groupby(option_feature)
            cost_by_category = store_cost['store_cost(in millions)'].sum().reset_index().sort_values(by='store_cost(in millions)',ascending=False)
            sales_by_category = store_cost['store_sales(in millions)'].sum().reset_index().sort_values(by='store_sales(in millions)',ascending=False) 
            revenue_by_category = sales_by_category.join(cost_by_category.set_index(option_feature), on=option_feature)
            revenue_by_category["revenue(in millions)"] = revenue_by_category["store_sales(in millions)"] - revenue_by_category["store_cost(in millions)"]
            
            labels = revenue_by_category[option_feature]
            values_cost = revenue_by_category["store_cost(in millions)"]
            values_sale = revenue_by_category["store_sales(in millions)"]
            values_rev = revenue_by_category["revenue(in millions)"]
            fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}]])
            fig.add_trace(go.Pie(labels=labels, values=values_cost, name="Store Costs"),1, 1)
            fig.add_trace(go.Pie(labels=labels, values=values_sale, name="Store Sales"),1, 2)
            fig.add_trace(go.Pie(labels=labels, values=values_rev, name="Store Revenue"),1, 3)
            fig.update_traces(hole=.4, hoverinfo="label+value+name")
            fig.update_layout(title_text="Stores Performance for {}".format(option_feature),
                                    annotations=[dict(text='COST', x=sum(fig.get_subplot(1, 1).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center"),
                                                dict(text='SALES', x=sum(fig.get_subplot(1, 2).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center"),
                                                dict(text='REV', x=sum(fig.get_subplot(1, 3).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center")])
            st.plotly_chart(fig)
            

        ed_inference = "People with partial college degree i.e. may be currently in college contribute most to revenue "
        occ_inference = "Professionals are more likely to buy more from stores."
        inference_dataframe = pd.DataFrame({"Inference": [ed_inference,occ_inference]})
        st.dataframe(inference_dataframe, width=1400, height=100,hide_index=True, use_container_width=True)

#member-card
with st.container(border=True):
    if "disabled_nine" not in st.session_state:
        st.session_state.disabled_nine = False

    st.checkbox(":blue[_Exploration 9_] || How much income bracket affects sales", key="disabled_nine")
    if st.session_state.disabled_nine:
 
        income_cost = costs_data.groupby('yearly_income')
        cost_by_income = income_cost['store_cost(in millions)'].sum().reset_index().sort_values(by='store_cost(in millions)',ascending=False)
        sales_by_income = income_cost['store_sales(in millions)'].sum().reset_index().sort_values(by='store_sales(in millions)',ascending=False) 
        revenue_by_income = sales_by_income.join(cost_by_income.set_index('yearly_income'), on='yearly_income')
        revenue_by_income["revenue(in millions)"] = revenue_by_income["store_sales(in millions)"] - revenue_by_income["store_cost(in millions)"]
            
        title_s = "store revenue for different income group"
        fig_l = px.line(revenue_by_income,x="yearly_income", y="revenue(in millions)",title = title_s, markers=True)
        fig_l.update_traces(line_color='violet', line_width=2)
        st.plotly_chart(fig_l)

        labels = revenue_by_income['yearly_income']
        values_cost = revenue_by_income["store_cost(in millions)"]
        values_sale = revenue_by_income["store_sales(in millions)"]
        values_rev = revenue_by_income["revenue(in millions)"]
        fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}]])
        fig.add_trace(go.Pie(labels=labels, values=values_cost, name="Store Costs"),1, 1)
        fig.add_trace(go.Pie(labels=labels, values=values_sale, name="Store Sales"),1, 2)
        fig.add_trace(go.Pie(labels=labels, values=values_rev, name="Store Revenue"),1, 3)
        fig.update_traces(hole=.4, hoverinfo="label+value+name")
        fig.update_layout(title_text="Stores Performance for yearly_income",
                                    annotations=[dict(text='COST', x=sum(fig.get_subplot(1, 1).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center"),
                                                dict(text='SALES', x=sum(fig.get_subplot(1, 2).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center"),
                                                dict(text='REV', x=sum(fig.get_subplot(1, 3).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center")])
        st.plotly_chart(fig)
        income_inference = "Higher income doesn't necessarily drive higher revenue. But sending promotional messages to encourage high earning customers can increase sale. "
        inference_dataframe = pd.DataFrame({"Inference": [income_inference]})
        st.dataframe(inference_dataframe, width=1400, height=50,hide_index=True, use_container_width=True)

#low-fat
with st.container(border=True):
    if "disabled_ten" not in st.session_state:
        st.session_state.disabled_ten = False

    st.checkbox(":blue[_Exploration 10_] || Is low fat food really a phenomenon?", key="disabled_ten")
    if st.session_state.disabled_ten:
        
        feat = 'low_fat'
        fat_cost = costs_data.groupby(feat)
        cost_by_fat = fat_cost['store_cost(in millions)'].sum().reset_index().sort_values(by='store_cost(in millions)',ascending=False)
        sales_by_fat = fat_cost['store_sales(in millions)'].sum().reset_index().sort_values(by='store_sales(in millions)',ascending=False) 
        revenue_by_fat = sales_by_fat.join(cost_by_fat.set_index(feat), on=feat)
        revenue_by_fat["revenue(in millions)"] = revenue_by_fat["store_sales(in millions)"] - revenue_by_fat["store_cost(in millions)"]

        labels = revenue_by_fat[feat]
        values_cost = revenue_by_fat["store_cost(in millions)"]
        values_sale = revenue_by_fat["store_sales(in millions)"]
        values_rev = revenue_by_fat["revenue(in millions)"]
        fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}]])
        fig.add_trace(go.Pie(labels=labels, values=values_cost, name="Store Costs"),1, 1)
        fig.add_trace(go.Pie(labels=labels, values=values_sale, name="Store Sales"),1, 2)
        fig.add_trace(go.Pie(labels=labels, values=values_rev, name="Store Revenue"),1, 3)
        fig.update_traces(hole=.4, hoverinfo="label+value+name")
        fig.update_layout(title_text="Stores Performance for low_fat food",
                                    annotations=[dict(text='COST', x=sum(fig.get_subplot(1, 1).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center"),
                                                dict(text='SALES', x=sum(fig.get_subplot(1, 2).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center"),
                                                dict(text='REV', x=sum(fig.get_subplot(1, 3).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center")])
        st.plotly_chart(fig)

        fat_data = costs_data[costs_data['low_fat']==1]
        fat_cost = fat_data.groupby("food_category")
        cost_by_fat = fat_cost['store_cost(in millions)'].sum().reset_index().sort_values(by='store_cost(in millions)',ascending=False)
        sales_by_fat = fat_cost['store_sales(in millions)'].sum().reset_index().sort_values(by='store_sales(in millions)',ascending=False) 
        revenue_by_fat = sales_by_fat.join(cost_by_fat.set_index("food_category"), on="food_category")
        revenue_by_fat["revenue(in millions)"] = revenue_by_fat["store_sales(in millions)"] - revenue_by_fat["store_cost(in millions)"]
        title_s = "Low fat Food Categories with highest sales"
        fig_l = px.line(revenue_by_fat,x="food_category", y="revenue(in millions)",title = title_s, markers=True)
        fig_l.update_traces(line_color='yellow', line_width=2)
        st.plotly_chart(fig_l)

        fat_inference = "Low fat food may be in the lower spectrum but even 38% of the total food sales is a huge portion."
        inference_dataframe = pd.DataFrame({"Inference": [fat_inference]})
        st.dataframe(inference_dataframe, width=1400, height=50,hide_index=True, use_container_width=True)

#recyclable_package
with st.container(border=True):
    if "disabled_eleven" not in st.session_state:
        st.session_state.disabled_eleven = False

    st.checkbox(":blue[_Exploration 11_] || Recyclability - The new way to go?", key="disabled_eleven")
    if st.session_state.disabled_eleven:
        
        feat = 'recyclable_package'
        fat_cost = costs_data.groupby(feat)
        cost_by_fat = fat_cost['store_cost(in millions)'].sum().reset_index().sort_values(by='store_cost(in millions)',ascending=False)
        sales_by_fat = fat_cost['store_sales(in millions)'].sum().reset_index().sort_values(by='store_sales(in millions)',ascending=False) 
        revenue_by_fat = sales_by_fat.join(cost_by_fat.set_index(feat), on=feat)
        revenue_by_fat["revenue(in millions)"] = revenue_by_fat["store_sales(in millions)"] - revenue_by_fat["store_cost(in millions)"]

        labels = revenue_by_fat[feat]
        values_cost = revenue_by_fat["store_cost(in millions)"]
        values_sale = revenue_by_fat["store_sales(in millions)"]
        values_rev = revenue_by_fat["revenue(in millions)"]
        fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}]])
        fig.add_trace(go.Pie(labels=labels, values=values_cost, name="Store Costs"),1, 1)
        fig.add_trace(go.Pie(labels=labels, values=values_sale, name="Store Sales"),1, 2)
        fig.add_trace(go.Pie(labels=labels, values=values_rev, name="Store Revenue"),1, 3)
        fig.update_traces(hole=.4, hoverinfo="label+value+name")
        fig.update_layout(title_text="Stores Performance for recyclable_package food",
                                    annotations=[dict(text='COST', x=sum(fig.get_subplot(1, 1).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center"),
                                                dict(text='SALES', x=sum(fig.get_subplot(1, 2).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center"),
                                                dict(text='REV', x=sum(fig.get_subplot(1, 3).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center")])
        st.plotly_chart(fig)

        fat_data = costs_data[costs_data['recyclable_package']==1]
        fat_cost = fat_data.groupby("food_category")
        cost_by_fat = fat_cost['store_cost(in millions)'].sum().reset_index().sort_values(by='store_cost(in millions)',ascending=False)
        sales_by_fat = fat_cost['store_sales(in millions)'].sum().reset_index().sort_values(by='store_sales(in millions)',ascending=False) 
        revenue_by_fat = sales_by_fat.join(cost_by_fat.set_index("food_category"), on="food_category")
        revenue_by_fat["revenue(in millions)"] = revenue_by_fat["store_sales(in millions)"] - revenue_by_fat["store_cost(in millions)"]
        title_s = "Recyclably packaged Food Categories with highest sales"
        fig_l = px.line(revenue_by_fat,x="food_category", y="revenue(in millions)",title = title_s, markers=True)
        fig_l.update_traces(line_color='yellow', line_width=2)
        st.plotly_chart(fig_l)

        rec_inference = "Recyclable packages visibly witness more sales. Filling up stocks with recycalble packages could make sales move faster."
        inference_dataframe = pd.DataFrame({"Inference": [rec_inference]})
        st.dataframe(inference_dataframe, width=1400, height=50,hide_index=True, use_container_width=True)


with st.container(border=True):
    if "disabled_twelve" not in st.session_state:
        st.session_state.disabled_twelve = False

    st.checkbox(":blue[_Exploration 12_] || Do Coupon Help ?", key="disabled_twelve")
    if st.session_state.disabled_twelve:
        
        feat = 'promotion_name'
        coup_cost = costs_data.groupby(feat)
        cost_by_coup = coup_cost['store_cost(in millions)'].sum().reset_index().sort_values(by='store_cost(in millions)',ascending=False)[:10]
        sales_by_coup = coup_cost['store_sales(in millions)'].sum().reset_index().sort_values(by='store_sales(in millions)',ascending=False)[:10] 
        revenue_by_coup = sales_by_coup.join(cost_by_coup.set_index(feat), on=feat)
        revenue_by_coup["revenue(in millions)"] = (revenue_by_coup["store_sales(in millions)"] - revenue_by_coup["store_cost(in millions)"])[:10]
        
        title_s = "Coupons bringing highest sales"
        fig_l = px.line(revenue_by_coup,x=feat, y="revenue(in millions)",title = title_s, markers=True)
        fig_l.update_traces(line_color='red', line_width=2)
        st.plotly_chart(fig_l)
        labels = revenue_by_coup[feat]
        values_cost = revenue_by_coup["store_cost(in millions)"]
        values_sale = revenue_by_coup["store_sales(in millions)"]
        values_rev = revenue_by_coup["revenue(in millions)"]

        fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}]])
        fig.add_trace(go.Pie(labels=labels, values=values_cost, name="Store Costs"),1, 1)
        fig.add_trace(go.Pie(labels=labels, values=values_sale, name="Store Sales"),1, 2)
        fig.add_trace(go.Pie(labels=labels, values=values_rev, name="Store Revenue"),1, 3)
        fig.update_traces(hole=.4, hoverinfo="label+value+name")
        fig.update_layout(title_text="Stores Performance for {} food".format(feat),
                                    annotations=[dict(text='COST', x=sum(fig.get_subplot(1, 1).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center"),
                                                dict(text='SALES', x=sum(fig.get_subplot(1, 2).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center"),
                                                dict(text='REV', x=sum(fig.get_subplot(1, 3).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center")])
        st.plotly_chart(fig)

with st.container(border=True):
    if "disabled_thirteen" not in st.session_state:
        st.session_state.disabled_thirteen = False

    st.checkbox(":blue[_Exploration 13_] || Favourable State of USA witnessing most revenue.", key="disabled_thirteen")
    if st.session_state.disabled_thirteen:
        fip_data = costs_data[costs_data['sales_country']=='USA']
        fips = pd.DataFrame(fip_data.groupby(['store_state']).size().reset_index().iloc[:,[0,1]])
        af = addfips.AddFIPS()
        for index, row in fips.iterrows():
            fips.at[index,'FIPS']=af.get_state_fips(fips.at[index,'store_state'])
        data = fips.merge(fip_data, how='inner', on=['store_state'])
        state_data = data.groupby(['FIPS',"store_state"])
        sales_by_state = state_data['store_sales(in millions)'].sum().reset_index().sort_values(by='store_sales(in millions)',ascending=False)
        costs_by_state = state_data['store_cost(in millions)'].sum().reset_index().sort_values(by='store_cost(in millions)',ascending=False)
        revenue_by_state = sales_by_state.join(costs_by_state.set_index(['FIPS',"store_state"]), on=['FIPS',"store_state"])
        revenue_by_state["revenue(in millions)"] = (revenue_by_state["store_sales(in millions)"] - revenue_by_state["store_cost(in millions)"])
   
        revenue_by_state['Sales Category'] = pd.qcut(revenue_by_state['store_sales(in millions)'], q=3, labels=['Low', 'Medium', 'High']) 
        fig = px.choropleth(revenue_by_state, 
                    locations='store_state',  # Using FIPS codes
                    locationmode='USA-states',  # Recognizing FIPS for USA states
                    color='Sales Category',  # Color based on sales
                    hover_name='store_state',
                    title='Store Sales in WA, OR, and CA',
                    scope='usa',
                    color_discrete_map={'Low': 'red', 'Medium': 'yellow', 'High': 'blue'})
        fig.update_layout(geo=dict(bgcolor="black"),  # Change map background color
                        paper_bgcolor="black",  # Change outer background color
                        font=dict(color="white"))
        
        st.plotly_chart(fig)
        
        title_s = "State with highest sales"
        fig_l = px.line(revenue_by_state,x="store_state", y="revenue(in millions)",title = title_s, markers=True)
        fig_l.update_traces(line_color='red', line_width=2)
        st.plotly_chart(fig_l)

        labels = revenue_by_state["store_state"]
        values_cost = revenue_by_state["store_cost(in millions)"]
        values_sale = revenue_by_state["store_sales(in millions)"]
        values_rev = revenue_by_state["revenue(in millions)"]

        fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}]])
        fig.add_trace(go.Pie(labels=labels, values=values_cost, name="Store Costs"),1, 1)
        fig.add_trace(go.Pie(labels=labels, values=values_sale, name="Store Sales"),1, 2)
        fig.add_trace(go.Pie(labels=labels, values=values_rev, name="Store Revenue"),1, 3)
        fig.update_traces(hole=.4, hoverinfo="label+value+name")
        fig.update_layout(title_text="Stores Performance for {} ".format("USA"),
                                    annotations=[dict(text='COST', x=sum(fig.get_subplot(1, 1).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center"),
                                                dict(text='SALES', x=sum(fig.get_subplot(1, 2).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center"),
                                                dict(text='REV', x=sum(fig.get_subplot(1, 3).x) / 2, y=0.5,font_size=20, showarrow=False, xanchor="center")])
        st.plotly_chart(fig)


with st.container(border=True):
    if "disabled_fourteen" not in st.session_state:
        st.session_state.disabled_fourteen = False

    st.checkbox(":blue[_Exploration 14_] || Costs of acquiring customers using member-cards per unit sale.", key="disabled_fourteen")
    if st.session_state.disabled_fourteen:
        feat = "member_card"
        costs_data["cac_unit"] = costs_data['cost']/costs_data['unit_sales(in millions)']
        feature_data = costs_data.groupby(feat)
        cost_by_feat = feature_data["cac_unit"].mean().reset_index().sort_values(by="cac_unit",ascending=False)
        st.dataframe(cost_by_feat.T, use_container_width=True)
        title_s = "Mean cost  of acquiring customer per unit sale using {}".format(feat)
        fig = px.bar(cost_by_feat, x = feat, y = "cac_unit",title = title_s)
        fig.update_traces(marker_color='red')
        # fig.update_traces(line_color='violet', line_width=2)
        st.plotly_chart(fig)
        
with st.container(border=True):
    if "disabled_15" not in st.session_state:
        st.session_state.disabled_15 = False

    st.checkbox(":blue[_Exploration 15_] || Costs of acquiring customers using promotions", key="disabled_15")
    if st.session_state.disabled_15:
        feat = "promotion_name"
        costs_data["cac_unit"] = costs_data['cost']/costs_data['unit_sales(in millions)']
        feature_data = costs_data.groupby(feat)
        cost_by_feat = feature_data["cac_unit"].mean().reset_index().sort_values(by="cac_unit",ascending=False)
        cost_by_feat2 = feature_data["cac_unit"].mean().reset_index().sort_values(by="cac_unit",ascending=True)[:5]
        st.dataframe(cost_by_feat.T, use_container_width=True)
        title_s = "Mean cost of acquiring customer per unit sale using {}".format(feat)
        fig = px.line(cost_by_feat, x = feat, y = "cac_unit",title = title_s, markers=True)
        fig.update_traces(line_color='red', line_width=2, marker_color = 'yellow')
        st.plotly_chart(fig)

        title_t = "5 {} with lowest cost of acquiring customer per unit sold".format(feat)
        fig = px.bar(cost_by_feat2, x = feat, y = "cac_unit",title = title_t)
        fig.update_traces(marker_color='blue')
        st.plotly_chart(fig)
        

with st.container(border=True):
    if "disabled_16" not in st.session_state:
        st.session_state.disabled_16 = False

    st.checkbox(":blue[_Exploration 16_] || Costs of acquiring customers for unit sold of food_categories", key="disabled_16")
    if st.session_state.disabled_16:
        feat = "food_category"
        costs_data["cac_unit"] = costs_data['cost']/costs_data['unit_sales(in millions)']
        feature_data = costs_data.groupby(feat)
        cost_by_feat = feature_data["cac_unit"].mean().reset_index().sort_values(by="cac_unit",ascending=False)
        cost_by_feat2 = feature_data["cac_unit"].mean().reset_index().sort_values(by="cac_unit",ascending=True)[:5]
        st.dataframe(cost_by_feat.T, use_container_width=True)
        title_s = "Mean cost of acquiring customer for unit {} sold".format(feat)
        fig = px.line(cost_by_feat, x = feat, y = "cac_unit", title = title_s, markers=True)
        fig.update_traces(line_color='red', line_width=2, marker_color = 'yellow')
        st.plotly_chart(fig)

        title_t = "5 {} with lowest cost of acquiring customer per unit sold".format(feat)
        fig = px.bar(cost_by_feat2, x = feat, y = "cac_unit", title = title_t)
        fig.update_traces(marker_color='blue')
        st.plotly_chart(fig)
        

with st.container(border=True):
    if "disabled_17" not in st.session_state:
        st.session_state.disabled_17 = False

    st.checkbox(":blue[_Exploration 17_] || Costs of acquiring customers for unit sold in a country", key="disabled_17")
    if st.session_state.disabled_17:
        feat = "sales_country"
        costs_data["cac_unit"] = costs_data['cost']/costs_data['unit_sales(in millions)']
        feature_data = costs_data.groupby(feat)
        cost_by_feat = feature_data["cac_unit"].mean().reset_index().sort_values(by="cac_unit",ascending=False)
        cost_by_feat2 = feature_data["cac_unit"].mean().reset_index().sort_values(by="cac_unit",ascending=True)[:5]
        st.dataframe(cost_by_feat.T, use_container_width=True)
        title_s = "Mean cost of acquiring customer for unit {} sold".format(feat)
        fig = px.line(cost_by_feat, x = feat, y = "cac_unit", title = title_s, markers=True)
        fig.update_traces(line_color='red', line_width=2, marker_color = 'yellow')
        st.plotly_chart(fig)

        # title_t = "5 {} with lowest cost of acquiring customer per unit sold".format(feat)
        # fig = px.bar(cost_by_feat2, x = feat, y = "cac_unit", title = title_t)
        # fig.update_traces(marker_color='blue')
        # st.plotly_chart(fig)


with st.container(border=True):
    if "disabled_18" not in st.session_state:
        st.session_state.disabled_18 = False

    st.checkbox(":blue[_Exploration 18_] || Costs of acquiring customers for unit sold based on gender/marital status/no. of children", key="disabled_18")
    if st.session_state.disabled_18:
        base_features = ["gender","marital_status","num_children_at_home"]
        feat = st.selectbox("Select one of the following feature.",base_features,key="feat")
        costs_data["cac_unit"] = costs_data['cost']/costs_data['unit_sales(in millions)']
        feature_data = costs_data.groupby(feat)
        cost_by_feat = feature_data["cac_unit"].mean().reset_index().sort_values(by="cac_unit",ascending=False)
        cost_by_feat2 = feature_data["cac_unit"].mean().reset_index().sort_values(by="cac_unit",ascending=True)[:5]
        st.dataframe(cost_by_feat.T, use_container_width=True)
        title_s = "Mean cost of acquiring customer for unit {} sold".format(feat)
        fig = px.line(cost_by_feat, x = feat, y = "cac_unit", title = title_s, markers=True)
        fig.update_traces(line_color='red', line_width=2, marker_color = 'yellow')
        st.plotly_chart(fig)

        # title_t = "5 {} with lowest cost of acquiring customer per unit sold".format(feat)
        # fig = px.bar(cost_by_feat2, x = feat, y = "cac_unit", title = title_t)
        # fig.update_traces(marker_color='blue')
        # st.plotly_chart(fig)

with st.container(border=True):
    if "disabled_19" not in st.session_state:
        st.session_state.disabled_19 = False

    st.checkbox(":blue[_Exploration 19_] || Costs of acquiring customers for unit sold based on education/occupation", key="disabled_19")
    if st.session_state.disabled_19:
        base_features = ["education","occupation"]
        feat = st.selectbox("Select one of the following feature.",base_features,key="feat")
        costs_data["cac_unit"] = costs_data['cost']/costs_data['unit_sales(in millions)']
        feature_data = costs_data.groupby(feat)
        cost_by_feat = feature_data["cac_unit"].mean().reset_index().sort_values(by="cac_unit",ascending=False)
        cost_by_feat2 = feature_data["cac_unit"].mean().reset_index().sort_values(by="cac_unit",ascending=True)[:5]
        st.dataframe(cost_by_feat.T, use_container_width=True)
        title_s = "Mean cost of acquiring customer for unit {} sold".format(feat)
        fig = px.line(cost_by_feat, x = feat, y = "cac_unit", title = title_s, markers=True)
        fig.update_traces(line_color='red', line_width=2, marker_color = 'yellow')
        st.plotly_chart(fig)

with st.container(border=True):
    if "disabled_20" not in st.session_state:
        st.session_state.disabled_20 = False

    st.checkbox(":blue[_Exploration 20_] || Costs of acquiring customers for unit sold based on assets", key="disabled_20")
    if st.session_state.disabled_20:
        base_features = ["houseowner","avg_cars_at home(approx)"]
        feat = st.selectbox("Select one of the following feature.",base_features,key="feat")
        costs_data["cac_unit"] = costs_data['cost']/costs_data['unit_sales(in millions)']
        feature_data = costs_data.groupby(feat)
        cost_by_feat = feature_data["cac_unit"].mean().reset_index().sort_values(by="cac_unit",ascending=False)
        cost_by_feat2 = feature_data["cac_unit"].mean().reset_index().sort_values(by="cac_unit",ascending=True)
        st.dataframe(cost_by_feat.T, use_container_width=True)
        title_s = "Mean cost of acquiring customer for unit {} sold".format(feat)
        fig = px.line(cost_by_feat, x = feat, y = "cac_unit", title = title_s, markers=True)
        fig.update_traces(line_color='red', line_width=2, marker_color = 'yellow')
        st.plotly_chart(fig)

with st.container(border=True):
    if "disabled_21" not in st.session_state:
        st.session_state.disabled_21 = False

    st.checkbox(":blue[_Exploration 21_] || Costs of acquiring customers for unit sold based on income bracket", key="disabled_21")
    if st.session_state.disabled_21:
        # base_features = ["gender","marital_status","num_children_at_home"]
        feat = "yearly_income"
        costs_data["cac_unit"] = costs_data['cost']/costs_data['unit_sales(in millions)']
        feature_data = costs_data.groupby(feat)
        cost_by_feat = feature_data["cac_unit"].mean().reset_index().sort_values(by="cac_unit",ascending=False)
        cost_by_feat2 = feature_data["cac_unit"].mean().reset_index().sort_values(by="cac_unit",ascending=True)[:5]
        st.dataframe(cost_by_feat.T, use_container_width=True)
        title_s = "Mean cost of acquiring customer for unit {} sold".format(feat)
        fig = px.line(cost_by_feat, x = feat, y = "cac_unit", title = title_s, markers=True)
        fig.update_traces(line_color='red', line_width=2, marker_color = 'yellow')
        st.plotly_chart(fig)

        # title_t = "5 {} with lowest cost of acquiring customer per unit sold".format(feat)
        # fig = px.bar(cost_by_feat2, x = feat, y = "cac_unit", title = title_t)
        # fig.update_traces(marker_color='blue')
        # st.plotly_chart(fig)


with st.container(border=True):
    if "disabled_22" not in st.session_state:
        st.session_state.disabled_22 = False

    st.checkbox(":blue[_Exploration 22_] || Costs of acquiring customers for unit sold of a brand.", key="disabled_22")
    if st.session_state.disabled_22:
        # base_features = ["gender","marital_status","num_children_at_home"]
        feat = "brand_name"
        costs_data["cac_unit"] = costs_data['cost']/costs_data['unit_sales(in millions)']
        feature_data = costs_data.groupby(feat)
        cost_by_feat = feature_data["cac_unit"].mean().reset_index().sort_values(by="cac_unit",ascending=False)
        cost_by_feat2 = feature_data["cac_unit"].mean().reset_index().sort_values(by="cac_unit",ascending=True)[:5]
        st.dataframe(cost_by_feat.T, use_container_width=True)
        title_s = "Mean cost of acquiring customer for unit {} sold".format(feat)
        fig = px.line(cost_by_feat, x = feat, y = "cac_unit", title = title_s, markers=True)
        fig.update_traces(line_color='red', line_width=2, marker_color = 'yellow')
        st.plotly_chart(fig)

        title_t = "5 {} with lowest cost of acquiring customer per unit sold".format(feat)
        fig = px.bar(cost_by_feat2, x = feat, y = "cac_unit", title = title_t)
        fig.update_traces(marker_color='blue')
        st.plotly_chart(fig)

with st.container(border=True):
    if "disabled_23" not in st.session_state:
        st.session_state.disabled_23 = False

    st.checkbox(":blue[_Exploration 23_] || Costs of acquiring customers for unit sold through media ads", key="disabled_23")
    if st.session_state.disabled_23:
        # base_features = ["gender","marital_status","num_children_at_home"]
        feat = "media_type"
        costs_data["cac_unit"] = costs_data['cost']/costs_data['unit_sales(in millions)']
        feature_data = costs_data.groupby(feat)
        cost_by_feat = feature_data["cac_unit"].mean().reset_index().sort_values(by="cac_unit",ascending=False)
        cost_by_feat2 = feature_data["cac_unit"].mean().reset_index().sort_values(by="cac_unit",ascending=True)[:5]
        st.dataframe(cost_by_feat.T, use_container_width=True)
        title_s = "Mean cost of acquiring customer for unit {} sold".format(feat)
        fig = px.line(cost_by_feat, x = feat, y = "cac_unit", title = title_s, markers=True)
        fig.update_traces(line_color='red', line_width=2, marker_color = 'yellow')
        st.plotly_chart(fig)

        title_t = "5 {} with lowest cost of acquiring customer per unit sold".format(feat)
        fig = px.bar(cost_by_feat2, x = feat, y = "cac_unit", title = title_t)
        fig.update_traces(marker_color='blue')
        st.plotly_chart(fig)

with st.container(border=True):
    if "disabled_24" not in st.session_state:
        st.session_state.disabled_24 = False

    st.checkbox(":blue[_Exploration 24_] || Costs of acquiring customers for unit sold in type of store", key="disabled_24")
    if st.session_state.disabled_24:
        # base_features = ["gender","marital_status","num_children_at_home"]
        feat = "store_type"
        costs_data["cac_unit"] = costs_data['cost']/costs_data['unit_sales(in millions)']
        feature_data = costs_data.groupby(feat)
        cost_by_feat = feature_data["cac_unit"].mean().reset_index().sort_values(by="cac_unit",ascending=False)
        cost_by_feat2 = feature_data["cac_unit"].mean().reset_index().sort_values(by="cac_unit",ascending=True)[:5]
        st.dataframe(cost_by_feat.T, use_container_width=True)
        title_s = "Mean cost of acquiring customer for unit {} sold".format(feat)
        fig = px.line(cost_by_feat, x = feat, y = "cac_unit", title = title_s, markers=True)
        fig.update_traces(line_color='red', line_width=2, marker_color = 'yellow')
        st.plotly_chart(fig)

        title_t = "5 {} with lowest cost of acquiring customer per unit sold".format(feat)
        fig = px.bar(cost_by_feat2, x = feat, y = "cac_unit", title = title_t)
        fig.update_traces(marker_color='blue')
        st.plotly_chart(fig)


