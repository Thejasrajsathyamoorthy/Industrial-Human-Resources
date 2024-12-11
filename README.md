## 📊 Industrial Human Resource Geo-Visualization
### 🎯  Project Overview
This project analyzes the workforce distribution across various industries in India, using state-wise data on main and marginal workers classified by industry, sex, and geography. The project aims to provide accurate insights for policy-making and employment planning through exploratory data analysis, natural language processing, machine learning, and an interactive dashboard.

### 📄 Table of Contents
1. Problem Statement
2. Dataset Overview
3. Technologies Used
4. Approach
5. Key Features
6. Results
7. Future Improvements

### 1. Problem Statement
The current workforce classification data in India is outdated and fails to capture the latest trends. This study aims to:

* Update the classification of main and marginal workers by industry and demographics.
* Provide insights into workforce distribution across states and industry categories.
* Help policymakers and businesses make informed decisions.

### 2. Dataset Overview
 Source: State-wise CSV files containing counts of workers classified by sex, rural/urban distribution, etc.

### Key Features:
'state_code',  'district_code',  'nic_name',  'main_workers_total_persons',  'marginal_workers_total_persons', etc.

### 3. Technologies Used
  *  #### Data Manipulation and Analysis: 
      * ![Alt text]('Pandas_logo.svg.png'), ![Alt text]('NumPy_logo_2020.svg.png')


  *  #### Visualization:
      *  matplotlib, seaborn, plotly.express
  *  #### NLP: 
      *  sklearn (TfidfVectorizer, KMeans)
  *  #### Machine Learning:
      *  RandomForestClassifier
  *  #### Dashboard Development:
      *  Streamlit, Plotly
  *  #### Deployment:
      *  joblib

### 4. Approach
#### Data Preparation:
  * Merged multiple state-wise datasets into a single dataframe.
  * Cleaned and standardized column names and entries.
    
#### Exploratory Data Analysis (EDA):
  * Identified workforce distribution by industry, state, sex, and geography.
  * Created visualizations for key insights.

#### Natural Language Processing (NLP):
  * Clustered industries using TfidfVectorizer and KMeans.
  * Categorized industries into groups like retail, manufacturing, and services.

#### Machine Learning:
  * Built a RandomForestClassifier to predict industry groups based on state, district, and other features.

#### Visualization Dashboard:
  * Created an interactive dashboard with Streamlit to visualize workforce data by geography and industry.

### 5. Key Features
  *  #### EDA Insights:
      *  Workforce distribution by state, rural/urban demographics, and sex.
      *  Top industries with the highest worker populations.
        
  *  #### NLP Insights:
      *  Grouped industries into seven meaningful clusters for better understanding.
        
  *  #### Machine Learning Model:
      *  Random Forest Classifier with 90% accuracy for predicting industry groups.

  *  #### Interactive Dashboard:
      *  Visualize data dynamically across states and industries.
      *  Analyze trends with custom filters and interactive plots.

### 6. Results
  *  Top Industries by Workforce:
      *  Retail, Manufacturing, Government services, etc.
  *  State-wise Workforce Distribution:
      *  Maharashtra and Uttar Pradesh have the highest worker populations.
  *  NLP Clustering:
      *  Industries grouped into categories like Small Industries, Service Organizations, and Retail Stores.
  *  Model Performance:
      *  Accuracy: 99.98%
      *  Precision: 99.99%
      *  Recall: 99.99%
      *  F1-Score: 99.99%


### 7. Future Improvements
  *  Include real-time workforce updates.
  *  Extend industry classification with deep learning techniques.
  *  Integrate advanced geospatial visualizations.


# Contributor -  Thejas Raj A S
* GitHub Repository: https://github.com/Thejasrajsathyamoorthy/Industrial-Human-Resources
* LinkedIn: www.linkedin.com/in/thejas-raj18
