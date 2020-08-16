import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import datetime
from sklearn.preprocessing import normalize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

@st.cache()
def get_and_process_data():
    data = pd.read_csv('data/household_power_consumption.txt', sep=';', 
                 parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True, 
                 low_memory=False, na_values=['nan','?'])
    data['Date'] = data.dt.dt.date
    df = data.groupby(['Date']).mean().reset_index()
    
    
    return df

def content():
    st.title('Regression on Time Series 📈')
    
    data = get_and_process_data()
    st.dataframe(data.head())

    col = st.selectbox('Select Column',data.columns.tolist())
    month = st.selectbox('Select Month',list(np.arange(1,12)))
    
    df = data[['Date',col]].fillna(0)
    
    #Split Val Set
    df['set'] = 'train'
    
    split_date = datetime.datetime.strptime(f'2010-{month}-01','%Y-%m-%d').date()
    
    df.set.loc[df.Date>split_date] = 'val'
    
    df['X'] = np.arange(1,len(df)+1)
    df['norm_X'] = normalize(df.X.values.reshape(-1,1),axis=0)
    
    if col != 'Date':
        st.altair_chart(alt.Chart(df[df.set=='train']).mark_circle().encode(x='Date:T',y=col)+
                       alt.Chart(df[df.set=='val']).mark_circle(color='red').encode(x='Date:T',y=col),use_container_width=True)
    

    
    
    st.header('Fit a Linear Regression')
    
    X = df.norm_X.values.reshape(-1,1)
    
    train_X = df.norm_X[df.set=='train'].values.reshape(-1,1)
    val_X = df.norm_X[df.set=='val'].values.reshape(-1,1)
    
    train_y = df[col][df.set == 'train'].values
    val_y = df[col][df.set == 'val'].values
    
    fit_lr = st.button('Fit LR Model')
    
    if fit_lr:
        lr = LinearRegression()
        pred_y = lr.fit(train_X,train_y).predict(val_X)
        pred_all = lr.predict(X)
        
        df['pred'] = pred_all
        
        
        
        st.altair_chart(alt.Chart(df[df.set=='train']).mark_circle().encode(x='Date:T',y=col)+
                       alt.Chart(df[df.set=='val']).mark_circle(color='red').encode(x='Date:T',y=col) +
            alt.Chart(df).mark_line(color='orange').encode(x='Date:T',y='pred'))
        
        err = np.sqrt(mean_squared_error(val_y,pred_y))
        st.write(f'Validation RMSE: {err}')
    
    st.header('Fit a Polynomial Regression')
    
    degree = st.selectbox('Select Degree',[2,3,4,5,6,7,8,9,10])
    fit_polyr = st.button('Fit PolyR Model')
    
    poly = PolynomialFeatures(degree=degree)
    
    poly_X = poly.fit_transform(X)
    poly_train_X = poly.fit_transform(train_X)
    poly_val_X = poly.fit_transform(val_X)
    
    if fit_polyr:
        lr = LinearRegression()
        pred_y = lr.fit(poly_train_X,train_y).predict(poly_val_X)
        pred_all = lr.predict(poly_X)
        
        df['pred'] = pred_all
        
        st.altair_chart(alt.Chart(df[df.set=='train']).mark_circle().encode(x='Date:T',y=col)+
                       alt.Chart(df[df.set=='val']).mark_circle(color='red').encode(x='Date:T',y=col) +
            alt.Chart(df).mark_line(color='orange').encode(x='Date:T',y='pred'))
        
        err = np.sqrt(mean_squared_error(val_y,pred_y))
        st.write(f'RMSE: {err}')


if __name__ == '__main__':

    content()