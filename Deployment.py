import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import style_metric_cards

##############################################################
# Preprocessing library
from sklearn.preprocessing import LabelEncoder,PolynomialFeatures,StandardScaler,RobustScaler,MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,root_mean_squared_error,confusion_matrix,accuracy_score,classification_report
from sklearn.decomposition import PCA
####################################################################
# Sampling library
from imblearn.combine import SMOTEENN
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
###################################################################
# Algorithm Library
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


# color schema for visualizion 
colors10 = ['#387478', '#4682B4', '#32CD32', '#FFD700','#001F3F','#B17457','#F2E5BF','#DA8359','#FFD09B','#A66E38']  # You can define your own colors
blue_1=['#2D4356', '#435B66', '#A76F6F', '#EAB2A0']
blue_2=['#0C134F', '#1D267D', '#2D4263', '#347474']
green1=['#1A1A19', '#31511E', '#859F3D', '#88C273']
brown1=['#A79277', '#D1BB9E', '#EAD8C0', '#FFF2E1']
yel_gre1=['#F3CA52', '#F6E9B2', '#0A6847', '#7ABA78']
red_tel=['#C96868', '#FADFA1', '#FFF4EA', '#7EACB5']
cofee=['#EAC696', '#C8AE7D', '#765827', '#65451F']
pastel=['#B5C0D0', '#CCD3CA', '#B4B4B8', '#B3A398']
retro=['#060047', '#B3005E', '#E90064', '#FF5F9E']
white_blue=['#04009A', '#77ACF1', '#77ACF1', '#C0FEFC']
cold_blue=['#240750', '#344C64', '#577B8D', '#57A6A1']
cold_green=['#006769', '#40A578', '#9DDE8B', '#E6FF94']
happy=['#D2E0FB', '#F9F3CC', '#D7E5CA', '#8EACCD']
sky=['#00A9FF', '#89CFF3', '#A0E9FF', '#CDF5FD']
grad_brown=['#8D7B68', '#A4907C', '#C8B6A6', '#F1DEC9']
grad_black=['#2C3333', '#2E4F4F', '#0E8388', '#CBE4DE']
grad_green=['#439A97', '#62B6B7', '#97DECE', '#CBEDD5']
grad_blue=['#164863', '#427D9D', '#9BBEC8', '#DDF2FD']
night=['#003C43', '#135D66', '#77B0AA', '#E3FEF7']


# read Data


# Configure page
st.set_page_config(page_title="Loan approval prediction",page_icon=":💶:",layout='wide',initial_sidebar_state="expanded")



st.title("💶 Loan approval prediction")
st.markdown("##")

df=pd.read_csv('Loan approval prediction.csv')
# Data Cleaning :
    
# determine the row that has the person age is 123
index_to_drop = df[df['person_age'] == 123].index
    
# Delete the row who has age 123    
df = df.drop(index_to_drop)
    
    
# determine the row that has the person_emp_length is 123
index_to_drop = df[df['person_emp_length'] == 123].index

# Delete the row who has age 123 
df = df.drop(index_to_drop)

with st.sidebar:
        selected=option_menu(
        menu_title="Main Menu",
        options=["Dataset Overview","Data Aanlysis","Modeling"],
        icons=["table", "bar-chart", "sliders"],
        menu_icon="cast", #option
        default_index=0, #option
        orientation="vertical"
        )

#switcher
st.sidebar.header("Please filter")
loan_intent=st.sidebar.multiselect(
    "Filter loan_intent",
     options=df["loan_intent"].unique(),
     default=df["loan_intent"].unique()
     )
person_home_ownership=st.sidebar.multiselect(
    "Filter person_home_ownership",
     options=df["person_home_ownership"].unique(),
     default=df["person_home_ownership"].unique()
)
loan_status=st.sidebar.multiselect(
    "Filter loan_status",
     options=df["loan_status"].unique(),
     default=df["loan_status"].unique()
)

# Apply custom CSS for multiselect, sidebar styling, and new elements
st.markdown(
    """
    <style>
    /* Main background color */
    .main {
        background-color: #00172B;
    }

    /* Sidebar background color */
    [data-testid="stSidebar"] {
        background-color: #063970;
        color: blue;
    }

    /* Sidebar content color */
    .sidebar-content {
        color: black;
    }

    /* Multiselect styling */
    .stMultiSelect .css-1wa3eu0 {
        background-color: #063970 !important; /* Background color for the multiselect */
        color: #063970 !important; /* Text color inside the multiselect */
    }

    /* Multiselect selected items color */
    .stMultiSelect .css-1n76uvr .css-1vbd788 {
        background-color: #063970 !important; /* Selected option background */
        color: black !important; /* Selected option text color */
    }

    /* Sidebar multiselect label color */
    .stSidebar .css-10trblm, .stSidebar .css-1d391kg {
        color: #063970 !important; /* Label color for multiselect options */
    }

    /* Box shadow for options */
    .stSidebar .css-2b097c-container {
        box-shadow: 0 0 2px #686664;
    }

    /* Metric container styling */
    [data-testid="metric-container"] {
        box-shadow: 0 0 2px #686664;
        padding: 5px;
    }

    /* Plot container styling */
    .plot-container > div {
        box-shadow: 0 0 2px #686664;
        padding: 5px;
    }

    /* Expander button styling */
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.3rem;
        color: #686664;
    }

    /* Metric container styling */
    [data-testid="metric-container"] {
        box-shadow: 0 0 2px #686664;
        padding: 5px;
    }

    /* Plot container styling */
    .plot-container > div {
        box-shadow: 0 0 2px #686664;
        padding: 5px;
    }

    </style>
    """,
    unsafe_allow_html=True
)




df_selection=df.query(
    "loan_intent==@loan_intent & person_home_ownership==@person_home_ownership & loan_status ==@loan_status"
)        

def table():
  with st.expander("Tabular"):
  #st.dataframe(df_selection,use_container_width=True)
   shwdata = st.multiselect('Filter :', df.columns, default=["person_age","person_income","person_home_ownership","person_emp_length","loan_intent","loan_grade","loan_amnt","loan_int_rate","loan_percent_income","cb_person_default_on_file","cb_person_cred_hist_length","loan_status"])
   st.dataframe(df_selection[shwdata],use_container_width=True)

if selected=="Dataset Overview":
# Dataset summary
    st.markdown("### Dataset Overview")
    st.markdown(""" The dataset contains **loan approval prediction data** with:
                        - **58,645 rows**
                - **13 columns**
                
                    """)
        # Detailed column descriptions
    st.markdown("### Column Descriptions")
    column_descriptions = {
            "id": "Unique identifier for each loan application.",
            "person_age": "Age of the loan applicant.",
            "person_income": "Annual income of the applicant.",
            "person_home_ownership": "Homeownership status (e.g., RENT, OWN).",
            "person_emp_length": "Number of years of employment experience.",
            "loan_intent": "Purpose of the loan (e.g., EDUCATION, MEDICAL, PERSONAL, VENTURE).",
            "loan_grade": "Loan grade representing the risk level.",
            "loan_amnt": "Requested loan amount.",
            "loan_int_rate": "Interest rate on the loan.",
            "loan_percent_income": "Ratio of the loan amount to the applicant’s income.",
            "cb_person_default_on_file": "Whether the person has a history of defaults (Y for yes, N for no).",
            "cb_person_cred_hist_length": "Length of the applicant’s credit history in years.",
            "loan_status": "Loan status (0 for rejected, 1 for approved)."
        }

    table()
    st.dataframe(df_selection.describe().T,use_container_width=True)
    
    for column, description in column_descriptions.items():
        st.markdown(
            f"<span style='color:green; font-weight:bold;'>{column}</span>: {description}",
            unsafe_allow_html=True,
        )
        
        

elif selected=="Data Aanlysis":
    st.title("EDA for loan data")        
    
    # Data Cleaning :
    
    # determine the row that has the person age is 123
    index_to_drop = df[df['person_age'] == 123].index
    
    # Delete the row who has age 123    
    df = df.drop(index_to_drop)
    
    
    # determine the row that has the person_emp_length is 123
    index_to_drop = df[df['person_emp_length'] == 123].index

    # Delete the row who has age 123 
    df = df.drop(index_to_drop)

    
    ########################## plot 1 ###################################### 
    def metrics():
        col1, col2= st.columns(2)

        col1.metric(label="Total person_income", value= f"{ df_selection.person_income.max()-df.person_income.min():,.0f}")
        
        col2.metric(label="Total loan_amnt", value= f"{df_selection.loan_amnt.sum():,.0f}")

        

        style_metric_cards(background_color="#121270",border_left_color="#f20045",box_shadow="3px")
    
    metrics()
    # Create the scatter plot to display the Age of the loan applicant 
    fig = px.scatter(df_selection['person_age'].value_counts().reset_index(),  x='person_age',y='count',  color='person_age',color_continuous_scale=px.colors.sequential.Plasma)
    
    # Update layout for title
    fig.update_layout( title="Age of the loan applicant",  title_x=0.4, title_font=dict(size=25,color="#EAC696"),title_xanchor='center')

    # Annotate mean, min, and max values
    fig.add_annotation( x=df_selection['person_age'].mean(), y=0, text=f"Mean: {df_selection['person_age'].mean():.2f}", showarrow=True, arrowhead=2, ax=0, ay=-50, font=dict(size=12, color="yellow") )
    fig.add_annotation(x=df_selection['person_age'].min(), y=0, text=f"Min: {df_selection['person_age'].min()}",showarrow=True, arrowhead=2, ax=0, ay=-50,font=dict(size=12, color="#A0E9FF"))
    fig.add_annotation(x=df_selection['person_age'].max(), text=f"Max: {df_selection['person_age'].max()}",showarrow=True, arrowhead=2, ax=0, ay=-50,font=dict(size=12, color="red"))
    
    st.plotly_chart(fig)
    
    st.divider()
    
    ########################## plot 2 ###################################### 
    
    # Create the scatter plot to display the Annual income of the applicant.
    fig = px.scatter(df_selection['person_income'].value_counts().reset_index(),     x='person_income', y='count',  color='person_income',color_continuous_scale=px.colors.sequential.Viridis)

    # Update layout for title
    fig.update_layout(title=" Annual income of the applicant." ,title_x=0.4, title_font=dict(size=25,color="#EAC696"),title_xanchor='center')

    # Annotate mean, min, and max values
    fig.add_annotation(x=df_selection['person_income'].mean(), y=0, text=f"Mean: {df_selection['person_income'].mean():.2f}",  showarrow=True, arrowhead=2, ax=0, ay=-50, font=dict(size=14, color="yellow") )
    fig.add_annotation(x=df_selection['person_income'].min(), y=0, text=f"Min: {df_selection['person_income'].min()}", showarrow=True, arrowhead=2, ax=0, ay=-75,  font=dict(size=12, color="#A0E9FF") )
    fig.add_annotation(x=df_selection['person_income'].max(), text=f"Max: {df_selection['person_income'].max()}",  showarrow=True, arrowhead=2, ax=0, ay=-50, font=dict(size=12, color="red")   )
    
    st.plotly_chart(fig)

    st.divider()

    ########################## plot 3 ###################################### 
    
    
    # Create the pie chart for Homeownership status
    fig = px.pie( df_selection['person_home_ownership'].value_counts().reset_index(), names='person_home_ownership',   values='count',  color='person_home_ownership'  )

    # Update layout for title
    fig.update_layout( title="Homeownership status", title_x=0.4, title_font=dict(size=25,color="#EAC696"),title_xanchor='center',legend_title="Homeownership status",legend_y=0.9)

    # Show name, percentage, and value on each slice
    fig.update_traces(textinfo='label+percent+value', hole=0.4,textposition="inside")

    st.plotly_chart(fig,use_container_width=True)

    st.divider()
    
########################## plot 4 ###################################### 
    # Create the scatter plot for Number of years of employment experience
    fig = px.scatter(df_selection['person_emp_length'].value_counts().reset_index(),   x='person_emp_length', y='count',  color='person_emp_length',color_continuous_scale=px.colors.sequential.Inferno)

    # Update layout for title
    fig.update_layout(   title="Number of years of employment experience.",title_x=0.4, title_font=dict(size=25,color="#EAC696"),title_xanchor='center' )

    # Annotate mean, min, and max values
    fig.add_annotation( x=df_selection['person_emp_length'].mean(), y=0, text=f"Mean: {df_selection['person_emp_length'].mean():.2f}",   showarrow=True, arrowhead=2, ax=20, ay=-40, font=dict(size=12, color="yellow")  )
    fig.add_annotation( x=df_selection['person_emp_length'].min(), y=0, text=f"Min: {df_selection['person_emp_length'].min()}",  showarrow=True, arrowhead=2, ax=20, ay=-40,  font=dict(size=12, color="#A0E9FF")  )
    fig.add_annotation( x=df_selection['person_emp_length'].max(), text=f"Max: {df_selection['person_emp_length'].max()}",  showarrow=True, arrowhead=2, ax=20, ay=-40,  font=dict(size=12, color="red")  )

    st.plotly_chart(fig)

    st.divider()

########################## plot 5 ###################################### 


    # Create the histogram plot for Purpose of the loan
    fig = px.histogram(df_selection['loan_intent'].value_counts().reset_index(),   x='loan_intent', y='count',   color='loan_intent', color_discrete_sequence=colors10)

    # Update layout for title
    fig.update_layout(title="The Purpose of the loan",title_x=0.4, title_font=dict(size=25,color="#EAC696"),title_xanchor='center' ) 
    fig.update_traces(textfont_size=18, textangle=0, textposition="outside", cliponaxis=False)
    
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

########################## plot 6 ###################################### 

# Create the pie chart for Loan grade representing the risk level
    fig = px.pie(df_selection['loan_grade'].value_counts().reset_index(), names='loan_grade', values='count',  color='loan_grade')

    # Update layout for title
    fig.update_layout(  title="Loan grade representing the risk level.",  title_x=0.4, title_font=dict(size=25,color="#EAC696"),title_xanchor='center')

    # Show name, percentage, and value on each slice
    fig.update_traces(textinfo='label+value', hole=0.4)
    
    st.plotly_chart(fig)

    st.divider()

########################## plot 7 ###################################### 

    # Create the scatter plot for Requested loan amount
    fig = px.scatter(df_selection['loan_amnt'].value_counts().reset_index(), x='loan_amnt', y='count',    color='loan_amnt',color_continuous_scale=px.colors.sequential.Cividis)

    # Update layout for title
    fig.update_layout(  title="Requested loan amount", title_x=0.4, title_font=dict(size=25,color="#EAC696"),title_xanchor='center')

    # Annotate mean, min, and max values
    fig.add_annotation(x=df_selection['loan_amnt'].mean(), y=0, text=f"Mean: {df_selection['loan_amnt'].mean():.2f}",  showarrow=True, arrowhead=2, ax=20, ay=-40, font=dict(size=12, color="yellow") )
    fig.add_annotation( x=df_selection['loan_amnt'].min(), y=0, text=f"Min: {df_selection['loan_amnt'].min()}", showarrow=True, arrowhead=2, ax=20, ay=-40, font=dict(size=12, color="#A0E9FF")  )
    fig.add_annotation( x=df_selection['loan_amnt'].max(), text=f"Max: {df_selection['loan_amnt'].max()}",  showarrow=True, arrowhead=2, ax=20, ay=-40,  font=dict(size=12, color="red"))

    st.plotly_chart(fig)

    st.divider()

########################## plot 8 ###################################### 

    # Create the scatter plot for Interest rate on the loan
    fig = px.scatter(df_selection['loan_int_rate'].value_counts().reset_index(),    x='loan_int_rate', y='count',  color='loan_int_rate',color_continuous_scale=px.colors.sequential.Reds)

    # Update layout for title
    fig.update_layout( title="Interest rate on the loan.", title_x=0.4, title_font=dict(size=25,color="#EAC696"),title_xanchor='center')

    # Annotate mean, min, and max values
    fig.add_annotation( x=df_selection['loan_int_rate'].mean(), y=0, text=f"Mean: {df_selection['loan_int_rate'].mean():.2f}",  showarrow=True, arrowhead=2, ax=0, ay=-30,  font=dict(size=12, color="yellow"))
    fig.add_annotation( x=df_selection['loan_int_rate'].min(), y=0, text=f"Min: {df_selection['loan_int_rate'].min()}",  showarrow=True, arrowhead=2, ax=0, ay=-30,  font=dict(size=12, color="#A0E9FF"))
    fig.add_annotation( x=df_selection['loan_int_rate'].max(), y=0, text=f"Max: {df_selection['loan_int_rate'].max()}",showarrow=True, arrowhead=2, ax=0, ay=-30,font=dict(size=12, color="red"))

    st.plotly_chart(fig)
    
    st.divider()
    
########################## plot 8 ###################################### 

    # Create the scatter plot for Ratio of the loan amount to the applicant’s income.
    fig = px.scatter(df_selection['loan_percent_income'].value_counts().reset_index(),   x='loan_percent_income', y='count',  color='loan_percent_income', color_continuous_scale=px.colors.sequential.Blues)

    # Update layout for title
    fig.update_layout( title="Ratio of the loan amount to the applicant’s income.",  title_x=0.4, title_font=dict(size=25,color="#EAC696"),title_xanchor='center') 

    # Annotate mean, min, and max values
    fig.add_annotation( x=df_selection['loan_percent_income'].mean(), y=0, text=f"Mean: {df_selection['loan_percent_income'].mean():.2f}",  showarrow=True, arrowhead=2, ax=10, ay=-40,  font=dict(size=12, color="yellow"))
    fig.add_annotation( x=df_selection['loan_percent_income'].min(), y=0, text=f"Min: {df_selection['loan_percent_income'].min()}",  showarrow=True, arrowhead=2, ax=10, ay=-40,  font=dict(size=12, color="#A0E9FF"))
    fig.add_annotation( x=df_selection['loan_percent_income'].max(), y=0, text=f"Max: {df_selection['loan_percent_income'].max()}", showarrow=True, arrowhead=2, ax=10, ay=-40, font=dict(size=12, color="red") )
    
    st.plotly_chart(fig)
    
    st.divider()

########################## plot 9 ###################################### 

    # Create the scatter plot for Length of the applicant’s credit history in years
    fig = px.scatter(df_selection['cb_person_cred_hist_length'].value_counts().reset_index(),   x='cb_person_cred_hist_length', y='count',   color='cb_person_cred_hist_length', color_continuous_scale=px.colors.sequential.Oranges)

    # Update layout for title
    fig.update_layout( title="Length of the applicant’s credit history in years.",  title_x=0.4, title_font=dict(size=25,color="#EAC696"),title_xanchor='center') 
    
    # Annotate mean, min, and max values
    fig.add_annotation(x=df_selection['cb_person_cred_hist_length'].mean(), y=0, text=f"Mean: {df_selection['cb_person_cred_hist_length'].mean():.2f}",  showarrow=True, arrowhead=2, ax=0, ay=-30,   font=dict(size=12, color="yellow")  )
    fig.add_annotation(x=df_selection['cb_person_cred_hist_length'].min(), y=0, text=f"Min: {df_selection['cb_person_cred_hist_length'].min()}",  showarrow=True, arrowhead=2, ax=0, ay=-30, font=dict(size=12, color="#A0E9FF") )
    fig.add_annotation(x=df_selection['cb_person_cred_hist_length'].max(), y=0, text=f"Max: {df_selection['cb_person_cred_hist_length'].max()}",   showarrow=True, arrowhead=2, ax=0, ay=-30,font=dict(size=12, color="red") )
    
    st.plotly_chart(fig)
    
    st.divider()

########################## plot 10 ###################################### 

    # Create the pie chart for Whether the person has a history of defaults.
    fig = px.pie(
        df_selection['cb_person_default_on_file'].value_counts().reset_index(),
        names='cb_person_default_on_file',  # Set regions as labels
        values='count',   # Set count as the value for each slice
        color='cb_person_default_on_file',
        color_discrete_sequence=grad_green
    )

    # Update layout for title
    fig.update_layout( title="Whether the person has a history of defaults.",title_x=0.4, title_font=dict(size=25,color="#EAC696"),title_xanchor='center' )


    # Show name, percentage, and value on each slice
    fig.update_traces(textinfo='label+percent+value', hole=0.4)
    
    st.plotly_chart(fig)
    
    st.divider()  # Horizontal rule for separation

    
    
########################## plot 11 ###################################### 

    # Create the pie chart for Loan status (0 for rejected, 1 for approved)
    fig = px.pie(
        df_selection['loan_status'].value_counts().reset_index(),
        names='loan_status',  # Set regions as labels
        values='count',   # Set count as the value for each slice
        color='loan_status',
        color_discrete_sequence=sky
    )

    # Update layout for title
    fig.update_layout(  title="Loan status (0 for rejected, 1 for approved).",  title_x=0.4, title_font=dict(size=25,color="#EAC696"),title_xanchor='center')
    # Show name, percentage, and value on each slice
    fig.update_traces(textinfo='label+percent+value', hole=0.4)
    
    st.plotly_chart(fig)
    
    
    st.divider()  # Horizontal rule for separation

########################## plot 12 ###################################### 


    # Create Scater plot between Applicant's Income vs Age with Loan Status as Bubble Siz
    fig = px.scatter(
        df_selection,
        x='person_age',
        y='person_income',
        size='person_income',  # Bubble size based on credit history length
        color='loan_status',
        color_continuous_scale='Viridis',
        title="Applicant's Income vs Age with Loan Status as Bubble Size"
    )

    # Update layout for title
    fig.update_layout(title_x=0.4, title_font=dict(size=25,color="#EAC696"),title_xanchor='center')

    st.plotly_chart(fig)


    # Use markdown with HTML to create headers and style text
    st.markdown(
        """
        <h4>The most common age range with higher income is between <span style="color:yellow;">22 and 50 years old</span>.</h4>
        <h4>Applicants aged between <span style="color:green;">22 and 35</span> tend to have incomes greater than <span style="color:red;">500k</span>.</h4>
        <h4>We notice that the accepted loans are mostly for ages between <span style="color:purple;">22 and 60</span> and it doesn't depend on the higher income at all. The maximum income with an accepted loan is <span style="color:orange;">379k</span>, however, the applicant with more than <span style="color:red;">1M income</span> hasn't been accepted.</h4>
        """,
        unsafe_allow_html=True
    )

########################## plot 13 ###################################### 

    st.divider()  # Horizontal rule for separation
    st.markdown("<br>", unsafe_allow_html=True)  # Optional extra spacing


    # Create a Pie chart between loan_status and loan_amount
    fig = px.pie(df_selection, names='loan_status', values='loan_amnt')

    # Update the layout with title and styling
    fig.update_layout(   title="Loan Status vs Loan Amount",  title_x=0.4, title_font=dict(size=25,color="#EAC696"),title_xanchor='center')
    # Show name, percentage, and value on each slice
    fig.update_traces(textinfo='label+percent+value', hole=0.4)

    # Show the plot
    st.plotly_chart(fig)

    st.markdown(
        """
        <h4>the loan amount of The Accepted Loan is <span style="color:yellow;">93M with 17.3% out of the total.</span>.</h4>
        <h4> the loan amount of The rejected Loan is <span style="color:green;">447M with 82.7% out of the total.</span> .</h4>
        """,
        unsafe_allow_html=True
    )
    
    st.divider()  # Horizontal rule for separation
    
    ########################## plot 14 ###################################### 

    # Create Scater plot between Applicant's Income vs loan_amount with loan_status 
    fig = px.scatter(
        df_selection,
        x='loan_amnt',
        y='person_income',
        size='loan_amnt',  # Bubble size based on credit history length
        color='loan_status',
        color_continuous_scale=px.colors.sequential.haline,  # Use a continuous color scale for better gradient
        title="Applicant's Income vs loan_amount with loan_status as Bubble Size"
    )

    # Update layout for title
    fig.update_layout(title_x=0.4, title_font=dict(size=25,color="#EAC696"),title_xanchor='center' )

    st.plotly_chart(fig)

    st.markdown("""<h4>  most of approval loan amount between <span style="color:#00A9FF;">  15k and 30k </span> and the applicant has less income </h4>""",unsafe_allow_html=True )
    
    st.divider() 
        
     ########################## plot 15 ###################################### 

    # Create Scater plot between Applicant's loan_amnt vs Age with Loan Status
    fig = px.scatter(
        df_selection,
        x='person_age',
        y='loan_amnt',
        size='loan_amnt',  # Bubble size based on credit history length
        color='loan_status',
        color_continuous_scale=px.colors.sequential.Plotly3,
        title="Applicant's loan_amount vs Age with Loan Status as Bubble Size"
    )

    # Update layout for title
    fig.update_layout(title_x=0.4, title_font=dict(size=25,color="#EAC696"),title_xanchor='center'  )

    st.plotly_chart(fig)

    st.markdown("""<h4> Most of approval loan amount between <span style="color:#00A9FF;"> 15k and 30k </span> and the applicant has ages between <span style="color:#00A9FF;"> 22 to 60. </span>  </h4>""",unsafe_allow_html=True )
    
    st.divider() 
    
    
    ########################## plot 16 ###################################### 

        
        # Create Scater plot between Applicant's person_emp_length vs Age with Loan Status
    fig = px.scatter(
        df_selection,
        x='person_age',
        y='person_emp_length',
        size='person_emp_length',  # Bubble size based on credit history length
        color='loan_status',
        color_continuous_scale=px.colors.sequential.RdBu,  # Use a continuous color scale for better gradient
        title="Applicant's person_emp_length vs Age with Loan Status as Bubble Size"
    )

    # Update layout for title
    fig.update_layout(title_x=0.4, title_font=dict(size=25,color="#EAC696"),title_xanchor='center')
    st.plotly_chart(fig)
    
    st.markdown("""<h4> <span style="color:#00A9FF;">  most of approval loan if the applicant has high Number of years of employment experience </span> </h4> """,unsafe_allow_html=True )  
    
    st.divider() 

    ########################## plot 16 ###################################### 

    # Create Scater plot between Applicant's cb_person_cred_hist_length vs Age with Loan Status
    fig = px.scatter(
        df_selection,
        x='person_age',
        y='cb_person_cred_hist_length',
        size='cb_person_cred_hist_length',  # Bubble size based on credit history length
        color='loan_status',
        color_continuous_scale=px.colors.sequential.Blackbody,  # Use a continuous color scale for better gradient
        title="Applicant's cb_person_cred_hist_length vs Age with Loan Status as Bubble Size"
    ) 
    
    # Update layout for title
    fig.update_layout(title_x=0.4, title_font=dict(size=25,color="#EAC696"),title_xanchor='center')
    st.plotly_chart(fig)
    
    st.markdown("""<h4> it seems that there is a <span style="color:#00A9FF;"> positive strong relationship </span> between age and the Length of the applicant’s credit history in years   </h4> """,unsafe_allow_html=True )   
    
    st.divider() 

    ########################## plot 17 ###################################### 
    
    
    fig = px.treemap(
        df_selection,
        path=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'],
        values='loan_amnt',
        color='loan_intent',  # Use loan amount for a gradient effect
        color_discrete_sequence=pastel,  # A lighter, easy-to-read gradient scale for dark themes
        title="Loan Amount Distribution by Home Ownership, Loan Intent, Loan Grade, and Default Status"
    )

    # Update layout for a centered title, background, and padding
    fig.update_layout(
        title={
            'text': "Loan Amount Distribution by Home Ownership, Loan Intent, Loan Grade, and Default Status",
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=25, color='#EAC696')  # Title color that contrasts with dark background
        },
        margin=dict(t=50, l=25, r=25, b=25),
        paper_bgcolor="#1E1E1E"  # Dark gray background for a modern dark theme
    )

    # Customize the hover info for clarity
    fig.update_traces(
        textinfo="label+value+percent root",  # Show label, value, and percentage of the root level
        hovertemplate="<b>%{label}</b><br>Loan Amount: %{value}<br>Percentage of Total: %{percentRoot:.2%}",
        marker=dict(colorscale='Mint')  # Ensure consistency in color tone
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)



    st.markdown(
    """
        <div style="font-family: Arial, sans-serif; font-size: 16px; line-height: 1.6;">
        <h4 style="margin-bottom: 10px;">It seems that in <span style="color:#00A9FF; font-weight:bold;">higher loan grades</span> across all homeownership types, the majority of applicants <span style="color:#00A9FF; font-weight:bold;">do not have a history of defaults</span>, indicating that most loans are paid on time.</h4>
        
        <h4 style="margin-bottom: 10px;">The most common loan purposes are for <span style="color:#00A9FF; font-weight:bold;">Education</span>, particularly among people with <span style="color:#00A9FF; font-weight:bold;">rent and mortgage homes</span>. However, those with <span style="color:#00A9FF; font-weight:bold;">owned homes or other living arrangements</span>
        are in the second phase. This suggests that individuals who need loans for <span style="color:#00A9FF; font-weight:bold;">education</span> often request higher loan amounts.</h4>
        
        <h4 style="margin-bottom: 10px;">For <span style="color:#00A9FF; font-weight:bold;">Medical loans</span>, the second-highest group is renters, while <span style="color:#00A9FF; font-weight:bold;">mortgage holders</span> and those with other living situations rank lower. Interestingly, those with <span style="color:#00A9FF; font-weight:bold;">owned homes</span> are in fourth place.</h4>
        
        <h4 style="margin-bottom: 10px;">For <span style="color:#00A9FF; font-weight:bold;">Venture loans</span>, individuals with <span style="color:#00A9FF; font-weight:bold;">owned homes or other living arrangements</span> take the first place, followed by renters and others. Mortgage holders rank third.</h4>
        </div>
        """, unsafe_allow_html=True)


    st.divider() 


    ########################## plot 18 ###################################### 

    # Create a Pie chart between loan_intent and loan_amnt
    fig = px.pie(df_selection, names='loan_intent', values='loan_amnt')

    # Update the layout with title and styling
    fig.update_layout(
        title="loan_intent with the loan amount", title_x=0.4, title_font=dict(size=25,color="#EAC696"),title_xanchor='center')
    # Show name, percentage, and value on each slice
    fig.update_traces(textinfo='label+percent+value', hole=0.4)

    # Show the plot
    st.plotly_chart(fig)


    st.divider() 
    
    ########################## plot 19 ###################################### 

    fig = px.box(
        data_frame=df,
        x='loan_grade',
        y='loan_int_rate',
        color='loan_status',
        color_discrete_sequence=['red', 'green'],
        title="Applicant's Interest Rate by loan_grade"
    )

    # Update layout for title
    fig.update_layout(
        title_x=0.5,
        title_font=dict(size=20),
        xaxis_title="Loan Status (0 = Rejected, 1 = Approved)",
        yaxis_title="Interest Rate (%)"
    )

    # Show the plot
    st.plotly_chart(fig)

    st.divider() 
    
    
    ########################## plot 20 ###################################### 


    fig = px.treemap(
        df_selection, 
        path=['person_home_ownership', 'loan_intent', 'loan_grade'], 
        values='person_income',
        color='loan_intent',  # Use loan_int_rate for a gradient color
        color_discrete_sequence=sky,  # Choose a visually pleasing continuous color scale
        title="person_income by Home Ownership, Loan Intent, and Grade"
    )

    # Update layout for a cleaner and more readable design
    fig.update_layout(title={ 'text': "person_income by Home Ownership, Loan Intent, and Grade",  'x': 0.5,'xanchor': 'center','yanchor': 'top','font': dict(size=22, color='#EAC696') } # Center and customize title font  
    ,  margin=dict(t=50, l=25, r=25, b=25),  # Add padding for clarity
        paper_bgcolor="#1E1E1E"  # Set background color to white
    )

    # Update trace for better hover information
    fig.update_traces(
        textinfo="label+percent entry+value",  # Display label, percentage, and value
        hovertemplate="<b>%{label}</b><br>person_income: %{value}<br>Percentage: %{percentEntry:.2%}"
    )

    st.plotly_chart(fig)

    st.divider() 
    
    ######################### plot 21 ###################################### 

    # Create a Pie chart between person_home_ownership and loan_amnt
    fig = px.pie(df_selection, names='person_home_ownership', values='person_income',color_discrete_sequence=happy)

    # Update the layout with title and styling
    fig.update_layout(
        title="person_home_ownership with the person_income",title_x=0.4, title_font=dict(size=25,color="#EAC696"),title_xanchor='center')

    # Show name, percentage, and value on each slice
    fig.update_traces(textinfo='label+percent+value', hole=0.3)
    
    st.plotly_chart(fig)
    
  
    st.markdown("""<h4> <span style="color:#608BC1; font-weight:bold;"> Mortgage </span> home Has higher Income ,<span style="color:#FFDC7F; font-weight:bold;"> Rent </span>
                home is the 2nd one and the <span style="color:#B8001F; font-weight:bold;"> Own </span> home is the 3rd </span> </h4> """, unsafe_allow_html=True)


    st.divider()
    
    
    ######################### plot 22 ###################################### 

    # Create the histogram plot for person_home_ownership and loan_amnt
    fig = px.histogram(df_selection, 
                    x='person_home_ownership', y='loan_amnt', 
                    color='person_home_ownership', color_discrete_sequence=colors10)

    # Update layout for title
    fig.update_layout(
        title="person_home_ownership and loan_amnt ",title_x=0.4, title_font=dict(size=25,color="#EAC696"),title_xanchor='center' )
    
    st.plotly_chart(fig)
    st.divider()

    
        ######################### plot 23 ###################################### 

    fig = px.box(
        data_frame=df_selection,
        x='loan_status',
        y='loan_percent_income',
        color='loan_status',
        color_discrete_sequence=['red', 'green'],
        title="Applicant's loan_percent_income by Loan Status"
    )

    # Update layout for title
    fig.update_layout(
        title_x=0.5,
        title_font=dict(size=20),
        title_xanchor='center' ,
        xaxis_title="Loan Status (0 = Rejected, 1 = Approved)",
        yaxis_title="Interest Rate (%)"
    )

    st.plotly_chart(fig)
    st.divider()
        
    ######################### plot 24 ###################################### 

    fig = px.box(
        data_frame=df_selection,
        x='loan_status',
        y='loan_int_rate',
        color='loan_status',
        color_discrete_sequence=['red', 'green'],
        title="Applicant's Interest Rate by Loan Status"
    )

    # Update layout for title
    fig.update_layout(
        title_x=0.5,
        title_font=dict(size=20),
        title_xanchor='center' ,
        xaxis_title="Loan Status (0 = Rejected, 1 = Approved)",
        yaxis_title="Interest Rate (%)"
    )

    st.plotly_chart(fig)
    st.divider()
        
    ######################### summary ###################################### 

    st.markdown(
    """
        <div style="background-color:#EAF2F8; padding:20px; border-radius:8px; border:1px solid #2980B9; font-size:20px;">
            <p align="center" style="color:#2C3E50; font-weight:bold; font-size:22px;">💡 Insights from the Visualization:</p>
                <!-- Insight 1 -->
                <div style="background-color:#D6EAF8; padding:15px; border-radius:8px; margin-bottom:15px;">
                    <p style="font-size:18px; color:#2C3E50;">
                        The most common age range with higher income is between <span style="color:#2980B9;">22 and 50 years old</span>.
                    </p>
                    <p style="font-size:18px; color:#2980B9;">
                        Most accepted loans are for applicants aged <span style="color:#2980B9;">22-60</span>, and high income does not guarantee loan acceptance.
                    </p>
                    <p style="font-size:18px; color:#2980B9;">
                        Applicants aged between <span style="color:#2980B9;">22 and 35</span> tend to have incomes greater than <span style="color:#E74C3C;">500k</span>.
                    </p>
                    <p style="font-size:18px; color:#2980B9;">
                        The accepted loans are mostly for ages <span style="color:#2980B9;">22-60</span>. The max income with an accepted loan is <span style="color:#E74C3C;">379k</span>, however, applicants with incomes over <span style="color:#E74C3C;">1M</span> were not accepted.
                    </p>
                </div>
                <!-- Loan Approval Insight -->
                <div style="background-color:#D6EAF8; padding:15px; border-radius:8px; margin-bottom:15px;">
                    <p style="font-size:18px; color:#2C3E50;">
                        <b>Accepted Loan Amount: <span style="color:#148F77;">93M (17.3% of total)</span></b><br>
                        <b>Rejected Loan Amount: <span style="color:#E74C3C;">447M (82.7% of total)</span></b>
                    </p>
                    <p style="font-size:20px; color:#E74C3C; background-color:#2C3E50; padding:10px; border-radius:8px; text-align:center;">
                        <b>Most Applicant Loans are Rejected</b>
                    </p>
                </div>
                <!-- Additional Insights -->
                <div style="background-color:#F9EBEA; padding:15px; border-radius:8px; margin-bottom:15px;">
                    <p style="font-size:18px; color:#2C3E50;">
                        Most approved loan amounts are between <span style="color:#E74C3C;">15k and 30k</span>, often for applicants with lower incomes or ages 22-60.
                    </p>
                    <p style="font-size:18px; color:#148F77;">
                        Higher loan approvals are correlated with applicants having longer employment experience.
                    </p>
                </div>
                <!-- Relationship between Age and Credit History -->
                <div style="background-color:#E8F8F5; padding:15px; border-radius:8px; margin-bottom:15px;">
                    <p style="font-size:18px; color:#1A5276; text-align:center;">
                        <b>Strong positive relationship observed between age and length of credit history.</b>
                    </p>
                </div>
                <!-- Loan Intent and Home Ownership Insights -->
                <div style="background-color:#FEF5E7; padding:15px; border-radius:8px; color:#7D6608; margin-bottom:15px;">
                    <p style="font-size:18px;">
                        Higher loan grades correlate with applicants who have no default history and often pay on time.
                    </p>
                    <p style="font-size:18px;">
                        Most loans are for <span style="color:#2980B9;">education</span>, especially for those renting or with mortgages, while homeowners often request loans for <span style="color:#2980B9;">ventures</span>.
                    </p>
                    <p style="font-size:18px;">
                        <span style="color:#E74C3C;">Medical loans</span> are primarily for renters, and <span style="color:#148F77;">venture loans</span> are commonly requested by homeowners.
                    </p>
                </div>
            <!-- Interest Rate and Loan Grade Insights -->
                <div style="background-color:#D1F2EB; padding:15px; border-radius:8px; margin-bottom:15px;">
                    <p style="font-size:18px; color:#117A65; text-align:center;">
                        Higher loan grades (higher risk) are associated with higher interest rates.
                    </p>
                </div>
                <!-- Income by Home Ownership -->
                <div style="background-color:#EBEDEF; padding:15px; border-radius:8px; text-align:center;">
                    <p style="font-size:18px; color:#5D6D7E;">
                        Mortgage homeowners have the highest income, followed by renters, and then owners.
                    </p>
                </div>
        </div>
    """, 
    unsafe_allow_html=True
    )




elif selected == "Modeling":
    st.title("ML Loan Prediction")
    st.text("In this app, we will classify the Loan Status (0 for rejected, 1 for approved).")
    st.text("Please enter the following values:")

    # Input for all required features as integers
    Person_age = st.slider("Enter the person's age", min_value=18, max_value=100, value=18, step=1) 
    Credit_hist_length = st.slider("Enter credit history length (in years)", min_value=0, max_value=50, value=10, step=1)
    Person_income = st.slider("Enter the person's income", min_value=1000, max_value=4000000, value=50000, step=1000)
    Loan_amnt = st.slider("Enter the loan amount (in thousands)", min_value=1, max_value=5000, value=100, step=10)
    formatted_loan_amnt = f"{Loan_amnt}k" if Loan_amnt < 1000 else f"{Loan_amnt / 1000:.1f}M"
    loan_int_rate = st.number_input("Enter the interest rate on the loan", min_value=0, max_value=40, value=1, step=1)
    Loan_percent_income = st.number_input("Enter the loan percent income", min_value=0, max_value=50, value=1, step=1)

    # Map categorical inputs to integers
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write("**Home Ownership**")
        Home_ownership = st.radio("Select ownership type", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'], key="home")
    with col2:
        st.write("**Loan Intent**")
        Loan_intent = st.radio("Purpose of loan", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'HOMEIMPROVEMENT'], key="intent")
    with col3:
        st.write("**Loan Grade**")
        Loan_grade = st.radio("Risk level", ['A', 'B', 'C', 'D', 'E', 'F', 'G'], key="grade")
    with col4:
        st.write("**Default on File**")
        Default_on_file = st.radio("History of defaults", ['Y', 'N'], key="default")

    # Display entered values
    st.write("### Summary of entered values:")
    st.write(f"**Age:** {Person_age} years")
    st.write(f"**Income:** ${Person_income}")
    st.write(f"**Interest Rate:** {loan_int_rate}%")
    st.write(f"**Loan Amount:** ${Loan_amnt}")
    st.write(f"**Loan Percent Income:** {Loan_percent_income}%")
    st.write(f"**Credit History Length:** {Credit_hist_length} years")
    st.write(f"**Homeownership status:** {Home_ownership}")
    st.write(f"**Purpose of the loan :** {Loan_intent}")
    st.write(f"**Loan grade representing the risk level:** {Loan_grade}")
    st.write(f"**Credit History of defaults (Y for yes, N for no):** {Default_on_file}")
    loan = df.copy()
    # Cached preprocessing and model training
    @st.cache_data
    def preprocess_data(df):
        label = LabelEncoder()
        object_column = loan[['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']]
        for i in object_column:
            loan[i] = label.fit_transform(loan[i])

        X = loan.drop(['id', 'loan_status', 'person_emp_length'], axis=1)
        y = loan['loan_status']
        smote_enn = SMOTEENN(random_state=42)
        X_combined, y_combined = smote_enn.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test, scaler

    @st.cache_resource
    def train_model(X_train, y_train):
        gb = xgb.XGBClassifier(n_estimators=600, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=10)
        gb.fit(X_train, y_train)
        return gb

    # Preprocess and train model
    loan = df.copy()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(loan)
    gb = train_model(X_train, y_train)

    # User input handling and prediction
    if st.button("Submit"):
        input_data = [Person_age, Person_income, Loan_amnt, Loan_percent_income, Credit_hist_length, loan_int_rate]

        # Encode categorical values
        category_mapping = {
            "Home_ownership": {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3},
            "Loan_intent": {'EDUCATION': 0, 'MEDICAL': 1, 'VENTURE': 2, 'PERSONAL': 3, 'HOMEIMPROVEMENT': 4},
            "Loan_grade": {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6},
            "Default_on_file": {'Y': 1, 'N': 0}
        }
        input_data.extend([
            category_mapping['Home_ownership'][Home_ownership],
            category_mapping['Loan_intent'][Loan_intent],
            category_mapping['Loan_grade'][Loan_grade],
            category_mapping['Default_on_file'][Default_on_file]
        ])

        # Scale input data and predict
        input_data = scaler.transform([input_data])
        result = gb.predict(input_data)
        if result == 1:
            st.success("Loan Approved ✅")
        else:
            st.error("Loan Rejected ❌")

