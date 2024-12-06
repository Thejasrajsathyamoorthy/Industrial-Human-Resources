import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu

eda_df = pd.read_csv("EDA_df.csv")



# Data Exploratin




























# Visualisation

def plot_top_industries(df, group_col, value_col, top_n, color='purple', title="Top Industries by Worker Population"):
    
    top_industries = df.groupby(group_col)[value_col].sum().sort_values(ascending=False).head(top_n)
    
    w_p_fig, ax = plt.subplots(figsize=(6, 8))
    top_industries.plot(kind='barh', color=color, ax =ax)
    ax.set_title(title)
    ax.set_xlabel("Number of Workers")
    ax.set_ylabel("Industry")
    ax.invert_yaxis()
    st.pyplot(w_p_fig)



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
    select = option_menu("Main menu", ["Home","Data Exploration", "Visualization"])




if select == "Home":
    pass



elif select == "Data Exploration":
    eda_df



elif select == "Visualization":
    col1,col2 = st.columns(2)
    with col1:
        st.write("Main Workers")
        plot_top_industries(eda_df, 'nic_name', 'main_workers_total_persons', 20)
    with col2:
        st.write("Marginal Workers")
        plot_top_industries(eda_df, 'nic_name', 'marginal_workers_total_persons', 20)
    
    st.write("")
    st.write("")

    col1,col2 = st.columns(2)
    with col1:
        st.write("Rural & Urban Workers Distribution")
        plot_rural_urban_distribution(eda_df)
    with col2:
        st.write("Men & Women Workers Distribution")
        plot_men_women_distribution(eda_df)

