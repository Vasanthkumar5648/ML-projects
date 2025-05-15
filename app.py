import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')

st.info('This is app builds a machine learning model!')
with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  df

  st.write('**X**')
  X_raw = df.drop('species', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.species
  y_raw

with st.expander('Data visualization'):
                 st.scatter_chart(data=df,x='bill_length_mm', y='body_mass_g', color='species')
                   

  
