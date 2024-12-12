import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
import seaborn as sns
import plotly.express as px
import joblib

nlp_df = pd.read_csv("NLP_df.csv")

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
    sns.barplot(x='state_code', y='total_workers', data=geo_data, ax=ax, hue= 'total_workers', legend= False, palette='deep')
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
            st.image("Seaborn.png")

    st.markdown("#### :blue[_Outcome_:]") 
    st.markdown("#### In this project, the population of workers for each state and its districts is analysed to provide relevant and accurate data through visualization for policy making and employment planning. Based on their work, they were catogorised into 7 groups using NLP, And then ML model is trained for future usage on predicting their industry based on their division, group, and class.")
    
    
elif select == "Data Exploration":
    
    st.header("Analysis of workers population")

    col1, col2, = st.columns(2)
    with col1:
        industry = st.selectbox(
                            "Select Industry,", 
                            ['Select any Industry Group'] + list(nlp_df['industry_group'].unique()))
                        
    industry_data = nlp_df[(nlp_df['industry_group'] == industry)]

    with col2:
        state = st.selectbox("Select State,",
                                ['Select any State'] + list(industry_data['states'].unique()))
    
    state_data = industry_data[(industry_data['states'] == state)]


    col1, col2 = st.columns(2)
    with col1:
        district = st.selectbox("Select District,",
                                ['Select any District'] + list(state_data['districts'].unique()))
    
    district_data = state_data[state_data['districts'] == district]


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

        st.subheader(f"Workers Population of '{industry}' sector")


        # Visualization of Total Workers Analysis

        total_fig = px.bar(class_data, x='total_workers', y='industry_group',
                            title="Total Workers", color_discrete_sequence=['#A9A9A9'])
        st.plotly_chart(total_fig)


        # Visualization of Main/Marginal Workers Analysis

        col1, col2, col3 = st.columns([8,2,8])

        with col1:
        
            total_main_workers_fig = px.bar(class_data, x='industry_group', y='main_workers_total_persons',
                                color_discrete_sequence=['#FFD1DC'], title="Total Main Workers")
            st.plotly_chart(total_main_workers_fig)

        with col3:
                
            total_marginal_workers_fig = px.bar(class_data, x='industry_group', y='marginal_workers_total_persons',
                                color_discrete_sequence=['#FFDAB9'], title="Total Marginal Workers")
            st.plotly_chart(total_marginal_workers_fig)


        # Visualization of Rural/Urban Workers Analysis

        col1, col2, col3, col4, col5, col6, col7 = st.columns([5,1,5,1,5,1,5])

        with col1:
            main_rural_fig = px.bar(class_data, x='industry_group', y='main_workers_rural_persons', 
                                title="Rural Main Workers", color_discrete_sequence=['#77DD77'])
            st.plotly_chart(main_rural_fig)

        with col3:
            main_urban_fig = px.bar(class_data, x='industry_group', y='main_workers_urban_persons',
                                    title="Urban Main Workers", color_discrete_sequence=['#F5F5DC'])
            st.plotly_chart(main_urban_fig)

        with col5:
            marginal_rural_fig = px.bar(class_data, x='industry_group', y='marginal_workers_rural_persons', 
                                title="Rural Marginal Workers", color_discrete_sequence=['#77DD77'])
            st.plotly_chart(marginal_rural_fig)

        with col7:
            marginal_urban_fig = px.bar(class_data, x='industry_group', y='marginal_workers_urban_persons',
                                    title="Urban Marginal Workers", color_discrete_sequence=['#F5F5DC'])
            st.plotly_chart(marginal_urban_fig)


        # Visualization of Males/Females Workers Analysis
        

        col1, col2, col3, col4, col5, col6, col7 = st.columns([5,1,5,1,5,1,5])

        with col1:
            main_males_fig = px.bar(class_data, x='industry_group', y='main_workers_total_males', 
                            title="Male Main Workers", color_discrete_sequence=['#1E90FF'])
            st.plotly_chart(main_males_fig)

        with col3:
            main_females_fig = px.bar(class_data, x='industry_group', y='main_workers_total_females',
                                title="Female Main Workers", color_discrete_sequence=['#FF00FF'])
            st.plotly_chart(main_females_fig)
        
        with col5:
            marginal_males_fig = px.bar(class_data, x='industry_group', y='marginal_workers_total_males', 
                            title="Male Marginal Workers", color_discrete_sequence=['#1E90FF'])
            st.plotly_chart(marginal_males_fig)

        with col7:
            marginal_females_fig = px.bar(class_data, x='industry_group', y='marginal_workers_total_females',
                                title="Female Marginal Workers", color_discrete_sequence=['#FF00FF'])
            st.plotly_chart(marginal_females_fig)
    else:
        st.write("")


elif select == "Visualization":

    col1, col2, col3 = st.columns([1,5,3])
    with col2:
        plot_workers_distribution_state(nlp_df)
    
    with col3:
        st.dataframe(nlp_df['states'].unique())

    col1, col2 = st.columns(2)
    with col1:
        state = st.sidebar.selectbox("Select State,",
                                ['Select any State'] + list( nlp_df['states'].unique()))
    
        state_data = nlp_df[(nlp_df['states'] == state)]

    with col2:
        district = st.sidebar.selectbox("Select District,",
                                ['Select any District'] + list(state_data['districts'].unique()))
    
        district_data = state_data[state_data['districts'] == district]

    if state_data.empty:
        st.info("Select State in Sidebar")
    
    
    elif not state_data.empty:

        if  district_data.empty:
        
            st.write("")
            st.write("")

            st.header(f"Visualization of Workers Distribution in '{state}'")

            st.write("")
            st.write("")

            col1, col2, col3 = st.columns([10,1,10])       
            with col1:
                st.subheader("Top Industries by 'Main Workers' Population")
                plot_top_industries(state_data, 'nic_name', 'main_workers_total_persons', 20, "Top Industries by Main Workers Population", 'purple')
            
            with col3:
                st.subheader("Top Industries by 'Marginal Workers' Population")
                plot_top_industries(state_data, 'nic_name', 'marginal_workers_total_persons', 20, "Top Industries by Marginal Workers Population", 'blue')
            
            st.write("")
            st.write("")

            col1, col2, col3 = st.columns([2,6,2])
            with col2:
                st.subheader("Top Industries by 'Total Workers' Population")
                plot_top_industries(state_data, 'nic_name', 'total_workers', 20, "Top Industries by Total Workers Population", 'red')
            
            st.write("")
            st.write("")

            col1, col2, col3 = st.columns([10,1,10])       
            with col1:
                st.subheader("'Rural & Urban' Workers Distribution")
                plot_rural_urban_distribution(state_data)
            
            with col3:
                st.subheader("'Men & Women' Workers Distribution")
                plot_men_women_distribution(state_data)


        elif not district_data.empty:

            st.write("")
            st.write("")

            st.header(f"Visualization of Workers Distribution in '{district}'")

            st.write("")
            st.write("")

            col1, col2, col3 = st.columns([10,1,10])       
            with col1:
                st.subheader("Top Industries by 'Main Workers' Population")
                plot_top_industries(district_data, 'nic_name', 'main_workers_total_persons', 20, "Top Industries by Main Workers Population", 'purple')
            
            with col3:
                st.subheader("Top Industries by 'Marginal Workers' Population")
                plot_top_industries(district_data, 'nic_name', 'marginal_workers_total_persons', 20, "Top Industries by Marginal Workers Population", 'blue')
            
            st.write("")
            st.write("")

            col1, col2, col3 = st.columns([2,6,2])
            with col2:
                st.subheader("Top Industries by 'Total Workers' Population")
                plot_top_industries(district_data, 'nic_name', 'total_workers', 20, "Top Industries by Total Workers Population", 'red')
            
            st.write("")
            st.write("")

            col1, col2, col3 = st.columns([10,1,10])       
            with col1:
                st.subheader("'Rural & Urban' Workers Distribution")
                plot_rural_urban_distribution(district_data)
            
            with col3:
                st.subheader("'Men & Women' Workers Distribution")
                plot_men_women_distribution(district_data)

        else:
            st.info("Select State and District for Further Visualization")

        





        

