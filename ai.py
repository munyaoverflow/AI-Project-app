# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# total_cases.csv
# load data

header = st.beta_container()
dataset = st.beta_container()
modeltraining = st.beta_container()
feature_block = st.beta_container()

with header:
    st.title('Covid Predictor')
    st.text("In this project the aim was to develop a python based covid predictor for the number of total world cases")

with dataset:
    st.header("Total World Covid Cases")
    st.text("The dataset was gotten online")

    # load data
    data = pd.read_csv('corona.csv', sep=',')
    africa = data[['id','World','Zimbabwe','Zambia','South Africa']]
    data = data[['id', 'World']]
    #print('-' * 30), print("HEAD"), print('-' * 30)
    #print(data.head())
    st.write(africa.head())
    world_cases = pd.DataFrame(africa['World'].value_counts()).head(50)
    st.bar_chart(world_cases)

   # print('-' * 30), print("PREPARE DATA"), print('-' * 30)
    x = np.array(data[['id']]).reshape(-1, 1)
    y = np.array(data['World']).reshape(-1, 1)
    plt.plot(y, '-m')  # plot with asterisk
# plt.show()
polyFeat = PolynomialFeatures(degree=4)
x = polyFeat.fit_transform(x)
with modeltraining:
    st.header("Predictions")
    st.text("You can pick days for the predictions now")
    sel_col, disp_col = st.beta_columns(2)
    days = sel_col.slider("How many days do you want to predict for?", min_value=1, max_value=10000, step = 10)
    #days = int(input('Enter number of days to predict to:'))
    #print('-' * 30), print("PREDICTION"), print('-' * 30)

    #print('-' * 30), print("TRAINING DATA"), print('-' * 30)
    model = linear_model.LinearRegression()
    xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2)
    model.fit(xtrain, ytrain)
    model.predict(xtest)
    accuracy = model.score(xtest, ytest)
    disp_col.subheader(f'Accuracy: {round(accuracy * 100, 3)} %')
    y0 = model.predict(xtest)
    y2 = model.predict(x)

    disp_col.subheader(f'Prediction - Cases in millions after {days} days:')
    disp_col.subheader(round(int(model.predict(polyFeat.fit_transform([[335 + days]])))/100000,2),)  #because data ends at 335 days
    disp_col.subheader(f'Mean Absolute Error: {round(mean_absolute_error(ytest,y0)/1000000,2)}')
    disp_col.subheader(f'Mean Squared Error: {round(mean_squared_error(ytest,y0)/1000000000000,2)}')
    disp_col.subheader(f'Root Mean Squared Error: {round(np.sqrt(mean_squared_error(ytest,y0)/10000000000000),2)}')
    

