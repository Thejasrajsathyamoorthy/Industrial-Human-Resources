import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
import seaborn as sns
import plotly.express as px


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



def plot_workers_distribution(data):
    geo_data = data.groupby('state_code')['total_workers'].sum().reset_index()

    workers_fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='state_code', y='total_workers', data=geo_data, ax=ax, palette="viridis")
    ax.set_title("Workers Distribution by State")
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
    select = option_menu("Main menu", ["Home","Analysis", "Visualization"])

if select == "Home":
    pass
    
elif select == "Analysis":
    
    # Distribution based on Industry group and State 
    col1, col2 = st.columns(2)
    with col1:
        industry = st.selectbox("Select Industry", nlp_df['industry_group'].unique())
    
    with col2:
        state = st.selectbox("Select State", nlp_df['state_code'].unique())
    
    filtered_data = nlp_df[(nlp_df['industry_group'] == industry) & (nlp_df['state_code'] == state)]


    # Distribution based on District 
    col1, col2 = st.columns(2)
    with col1:
        district = st.selectbox("Select District", filtered_data['district_code'].unique())
    
    district_data = filtered_data[filtered_data['district_code'] == district]


    # Distribution based on Division 
    with col2:
        division = st.selectbox("Select Division", district_data['division'].unique())
    
    division_data = district_data[district_data['division'] == division]


    # Distribution based on Group 
    col1, col2 = st.columns(2)
    with col1:
        group = st.selectbox("Select Group", division_data['group'].unique())
    
    group_data = division_data[division_data['group'] == group]


    # Distribution based on Class 
    with col2:
        class_details = st.selectbox("Select Class", group_data['class'].unique())
    
    class_data = group_data[group_data['class'] == class_details]


    # Visualization of Analysis
    st.write(f"Workers in {industry}")
    fig = px.bar(class_data, x='total_workers', y='industry_group', title="Worker Distribution")
    st.plotly_chart(fig)



elif select == "Visualization":
    col1,col2 = st.columns(2)
    with col1:
        st.write("Top Industries by Main Worker Population")
        plot_top_industries(nlp_df, 'nic_name', 'main_workers_total_persons', 20, "Top Industries by Main Worker Population", 'purple')
    with col2:
        st.write("Top Industries by Marginal Worker Population")
        plot_top_industries(nlp_df, 'nic_name', 'marginal_workers_total_persons', 20, "Top Industries by Marginal Worker Population", 'blue')
    
    st.write("")
    st.write("")

    col1,col2 = st.columns(2)
    with col1:
        st.write("Top Industries by Total Worker Population")
        plot_top_industries(nlp_df, 'nic_name', 'total_workers', 20, "Top Industries by Total Worker Population", 'red')
    with col2:
        st.write("Workers Distribution by States")
        plot_workers_distribution(nlp_df)

    col1,col2 = st.columns(2)
    with col1:
        st.write("Rural & Urban Workers Distribution")
        plot_rural_urban_distribution(nlp_df)
    with col2:
        st.write("Men & Women Workers Distribution")
        plot_men_women_distribution(nlp_df)

