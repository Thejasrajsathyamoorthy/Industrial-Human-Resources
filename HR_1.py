import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
import seaborn as sns
import plotly.express as px
import joblib

I_HR_model = joblib.load("Industrial_Human_Resouces.pkl")

nlp_df = pd.read_csv("NLP_df.csv")
state_name_code = pd.read_csv("State_Name_code.csv")

# Prediction of Industry Groups

def preprocess_input(state_code, district_code, division, group, class_):
    return pd.DataFrame({
        "state_code": [state_code],
        "district_code": [district_code],
        "division": [division],
        "group": [group],
        "class": [class_]
                })


# Visualisation

def plot_top_industries(df, group_col, value_col, top_n, title, color ):
    
    top_industries = df.groupby(group_col)[value_col].sum().sort_values(ascending=False).head(top_n)
    
    w_p_fig, ax = plt.subplots(figsize=(6, 8))
    top_industries.plot(kind='barh', color=color, ax =ax)
    ax.set_title(title) 
    ax.set_xlabel("Number of Workers")
    ax.set_ylabel("Industry")
    ax.invert_yaxis()
    st.pyplot(w_p_fig)



def plot_workers_distribution_state(data):
    geo_data = data.groupby('state_code')['total_workers'].sum().reset_index()

    workers_fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='state_code', y='total_workers', data=geo_data, ax=ax, palette="viridis")
    ax.set_title("Workers Population State-wise")
    ax.set_xlabel("State Code")
    ax.set_ylabel("Total Workers")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(workers_fig)



def plot_rural_urban_distribution(df,  title="Rural & Urban Workers Distribution"):
    
    columns_to_plot = [
            'main_workers_rural_persons', 'main_workers_urban_persons',
                  'marginal_workers_rural_persons', 'marginal_workers_urban_persons'
            ]

    distribution = df[columns_to_plot].sum()
    
    colors = ['green', 'red', 'blue', 'orange']
    
    r_u_fig, ax = plt.subplots(figsize=(6, 5))
    distribution.plot(kind='bar', color=colors, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of Workers")
    st.pyplot(r_u_fig)



def plot_men_women_distribution(df,  title="Men & Women Workers Distribution"):
    
    columns_to_plot = [
            'main_workers_total_males', 'main_workers_total_females',
                    'marginal_workers_total_males', 'marginal_workers_total_females'
            ]
    
    distribution = df[columns_to_plot].sum()
    
    colors = ['green', 'red', 'blue', 'orange']
    
    m_w_fig, ax = plt.subplots(figsize=(6, 5))
    distribution.plot(kind='bar', color=colors, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of Workers")
    st.pyplot(m_w_fig)




# Streamlit Part

st.set_page_config(page_title="Industrial Human Resources Geo-visualization", page_icon=":bar_chart:", layout="wide")
st.title(":bar_chart:  Industrial Human Resouces Geo-visualization")

st.markdown("<style>div.block-container{padding-top:3rem;}</style>", unsafe_allow_html= True)

with st.sidebar:
    select = option_menu("Main menu", ["Intro","Data Exploration", "Visualization"])

if select == "Intro":

    col1, col2, col3 = st.columns([14,2,8])
    with col1:
        st.write("\n")
        st.markdown('#### :blue[_Problem Statement_ :] In India, the industrial classification of the workforce is essential to understand the distribution of the labor force across various sectors.')
        st.markdown('#### The classification of main workers and marginal workers, other than cultivators and agricultural laborers, by sex and by section, division, and class, has been traditionally used to understand the economic status and employment trends in thecountry.')
        st.markdown('#### However, the current data on this classification is outdated and may not accurately reflect the current state of the workforce.')
        st.markdown('#### The aim of this study is to update the information on the industrial classification of the main and marginal workers, other than cultivators and agricultural laborers, by sex and by section, division, and class, to provide relevant and accurate data for policy making and employment planning.')
        st.markdown("#### :blue[_Skills take away_ :] Python scripting, Data Pre-processing, EDA, Visualization, Streamlit, Machine Learning (RandomForestClassifier), NLP.")
        st.markdown("#### :blue[_Domain_:] Resource Management")

    with col3:
        st.header(":red[_Tools Used_ :]")
        col1, col2 = st.columns(2)
        with col1:
            st.image("NumPy_logo_2020.svg.png")
            st.image("Pandas_logo.svg.png")
            st.image("Plotly.jpg")

        with col2:
            st.image("Matplotlib.png")
            st.image("Scikit-learn.png")

    st.markdown("#### :blue[_Outcome_:]") 
    st.markdown("#### In this project, the population of workers for each state and its districts is analysed to provide relevant and accurate data for policy making and employment planning. Based on their work, they were catogorised based on their work nature into 7 groups using NLP, And then trained model for future usage on predicting their industry w.r.t their work nature.")
    
    
elif select == "Data Exploration":
    
    tab1, tab2 = st.tabs(['Analysis', 'Prediction'])
    
    with tab1:

        st.header("Analysis of workers population")

        col1, col2, = st.columns(2)
        with col1:
            industry = st.selectbox(
                                "Select Industry,", 
                                ['Select any Industry Group'] + list(nlp_df['industry_group'].unique()))
                            
        industry_data = nlp_df[(nlp_df['industry_group'] == industry)]

        with col2:
            state = st.selectbox("Select State,",
                                    ['Select any State'] + list(industry_data['state_code'].unique()))
        
        state_data = industry_data[(industry_data['state_code'] == state)]


        col1, col2 = st.columns(2)
        with col1:
            district = st.selectbox("Select District,",
                                    ['Select any District'] + list(state_data['district_code'].unique()))
        
        district_data = state_data[state_data['district_code'] == district]


        with col2:
            division = st.selectbox("Select Division,",
                                    ['Select any Division'] + list(district_data['division'].unique()))
        
        division_data = district_data[district_data['division'] == division]


        col1, col2 = st.columns(2)
        with col1:
            group = st.selectbox("Select Group,",
                                    ['Select any Group'] + list(division_data['group'].unique()))
        
        group_data = division_data[division_data['group'] == group]


        with col2:
            class_details = st.selectbox("Select Class,",
                                            ['Select any Class'] + list(group_data['class'].unique()))
        
        class_data = group_data[group_data['class'] == class_details]

        if not class_data.empty:
            st.write("")
            st.write(f"Workers in {industry}")
        else:
            st.write("")

            

        # Visualization of Total Workers Analysis

        total_fig = px.bar(class_data, x='total_workers', y='industry_group',
                            title="Total Workers")
        st.plotly_chart(total_fig)

        col1, col2 = st.columns(2)
        # Visualization of Total Main Workers Analysis
        with col1:
        
            total_main_workers_fig = px.bar(class_data, x='main_workers_total_persons', y='industry_group',
                                title="Main Workers")
            st.plotly_chart(total_main_workers_fig)

        # Visualization of Total Marginal Workers Analysis
        with col2:
            
            total_marginal_workers_fig = px.bar(class_data, x='marginal_workers_total_persons', y='industry_group',
                                title="Marginal Workers")
            st.plotly_chart(total_marginal_workers_fig)


        # Visualization of Male Workers Analysis

        col1, col2 = st.columns(2)
        with col1:
            males_fig = px.bar(class_data, x='total_male_workers', y='industry_group', 
                                title="Male Workers")
            st.plotly_chart(males_fig)


        # Visualization of Female Workers Analysis

        with col2:
            females_fig = px.bar(class_data, x='total_female_workers', y='industry_group',
                                    title="Female Workers")
            st.plotly_chart(females_fig)

    with tab2:
        st.header("Predict Industry Group")
        try:
            col1, col2, col3 = st.columns(3)
            with col1:
                state_code = st.text_input("Enter State Code")
            with col2:
                district_code = st.text_input("Enter District Code")
            with col3:
                division = st.text_input("Enter Division")
                
            col1, col2, col3, col4, col5 = st.columns([2,6,2.5,6,2])
            with col2:
                group = st.text_input("Enter Group")
            with col4:
                class_ = st.text_input("Enter Class")
                
            col1, col2, col3, col4, col5 = st.columns([2,6,2,6,2])
            with col2:
                st.write("")
                if st.button("Predict Industry Group", icon= '🔍'):
                    input_data = preprocess_input(state_code, district_code, division, group, class_)
                    prediction = I_HR_model.predict(input_data)[0]

                    if prediction == 0:
                        st.write(f"Predicted Industry Group : Developmental Projects")
                    elif prediction == 1:
                        st.write(f"Predicted Industry Group : Education")
                    elif prediction == 2:
                        st.write(f"Predicted Industry Group : Health sector")
                    elif prediction == 3:
                        st.write(f"Predicted Industry Group : Heavy Industries")
                    elif prediction == 4:
                        st.write(f"Predicted Industry Group : Organizational works")
                    elif prediction == 5:
                        st.write(f"Predicted Industry Group : Retail Stores")
                    elif prediction == 6:
                        st.write(f"Predicted Industry Group : Small Industries")

        except:
            st.warning('Enter the Details')



elif select == "Visualization":

    col1, col2, col3 = st.columns([1,5,3])
    with col2:
        plot_workers_distribution_state(nlp_df)
    
    with col3:
        st.dataframe(state_name_code)

    col1, col2 = st.columns(2)
    with col1:
        state = st.sidebar.selectbox("Select State,",
                                ['Select any State'] + list( nlp_df['state_code'].unique()))
    
    state_data = nlp_df[(nlp_df['state_code'] == state)]

    with col2:
        district = st.sidebar.selectbox("Select District,",
                                ['Select any District'] + list(state_data['district_code'].unique()))
    
    district_data = state_data[state_data['district_code'] == district]

    if state_data.empty:
        st.info("Select State")
    
    
    elif not state_data.empty:

        if  district_data.empty:
        
            st.write("")
            st.write("")

            st.header("Visualization of Top industries in State")

            col1, col2 = st.columns(2)        
            with col1:
                st.write("Top Industries by Main Worker Population")
                plot_top_industries(state_data, 'nic_name', 'main_workers_total_persons', 20, "Top Industries by Main Worker Population", 'purple')
            
            with col2:
                st.write("Top Industries by Marginal Worker Population")
                plot_top_industries(state_data, 'nic_name', 'marginal_workers_total_persons', 20, "Top Industries by Marginal Worker Population", 'blue')
            
            st.write("")
            st.write("")

            col1, col2, col3 = st.columns([2,6,2])
            with col2:
                st.write("Top Industries by Total Worker Population")
                plot_top_industries(state_data, 'nic_name', 'total_workers', 20, "Top Industries by Total Worker Population", 'red')
            
            st.write("")
            st.write("")

            col1,col2 = st.columns(2)
            with col1:
                st.write("Rural & Urban Workers Distribution")
                plot_rural_urban_distribution(state_data)
            
            with col2:
                st.write("Men & Women Workers Distribution")
                plot_men_women_distribution(state_data)


        elif not district_data.empty:

            st.write("")
            st.write("")

            st.header("Visualization of Top industries in District")

            col1, col2 = st.columns(2)        
            with col1:
                st.write("Top Industries by Main Worker Population")
                plot_top_industries(district_data, 'nic_name', 'main_workers_total_persons', 20, "Top Industries by Main Worker Population", 'purple')
            
            with col2:
                st.write("Top Industries by Marginal Worker Population")
                plot_top_industries(district_data, 'nic_name', 'marginal_workers_total_persons', 20, "Top Industries by Marginal Worker Population", 'blue')
            
            st.write("")
            st.write("")

            col1, col2, col3 = st.columns([2,6,2])
            with col2:
                st.write("Top Industries by Total Worker Population")
                plot_top_industries(district_data, 'nic_name', 'total_workers', 20, "Top Industries by Total Worker Population", 'red')
            
            st.write("")
            st.write("")

            col1,col2 = st.columns(2)
            with col1:
                st.write("Rural & Urban Workers Distribution")
                plot_rural_urban_distribution(district_data)
            
            with col2:
                st.write("Men & Women Workers Distribution")
                plot_men_women_distribution(district_data)

        else:
            st.info("Select State and District for Further Visualization")

        





        

