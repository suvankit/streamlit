#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pycaret.datasets import get_data
dataset = get_data('employee')


# In[2]:


data_seen = dataset.sample(frac=0.95, random_state=780).reset_index(drop=True)
data_unseen = dataset.drop(data_seen.index).reset_index(drop=True)
dataset=dataset.drop(['department','average_montly_hours'],axis=1)
print('Data for Modeling: ' + str(data_seen.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))


# In[3]:


from pycaret.classification import *
setting_up = setup(data = data_seen, target = 'left', session_id=123)


# In[4]:


compare_models()


# In[5]:


rf = create_model('rf')


# In[6]:


tuned_model = tune_model(rf)


# In[7]:


final = finalize_model(tuned_model)
unseen_predictions = predict_model(final, data=data_unseen)
unseen_predictions.head()


# In[8]:


save_model(final,'Final_model')


# In[9]:


from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
model = load_model('Final_model')


# In[10]:


def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions


# In[11]:


def run():
    from PIL import Image
    image = Image.open('Employee.png')
    image_hospital = Image.open('office.jpg')
    st.image(image,use_column_width=False)
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))
    st.sidebar.info('This app is created to predict if an employee will leave the company')
    st.sidebar.success('https://www.pycaret.org')
    st.sidebar.image(image_hospital)
    st.title("Predicting employee leaving")
    if add_selectbox == 'Online':
        satisfaction_level=st.number_input('satisfaction_level' , min_value=0.1, max_value=1.0, value=0.1)
        last_evaluation =st.number_input('last_evaluation',min_value=0.1, max_value=1.0, value=0.1)
        number_project = st.number_input('number_project', min_value=0, max_value=50, value=5)
        time_spend_company = st.number_input('time_spend_company', min_value=1, max_value=10, value=3)
        Work_accident = st.number_input('Work_accident',  min_value=0, max_value=50, value=0)
        promotion_last_5years = st.number_input('promotion_last_5years',  min_value=0, max_value=50, value=0)
        salary = st.selectbox('salary', ['low', 'high','medium'])
        output=""
        input_dict={'satisfaction_level':satisfaction_level,'last_evaluation':last_evaluation,'number_project':number_project,'time_spend_company':time_spend_company,'Work_accident': Work_accident,'promotion_last_5years':promotion_last_5years,'salary' : salary}
        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)
        st.success('The output is {}'.format(output))
        if add_selectbox == 'Batch':
            file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)


# In[ ]:





# In[ ]:




