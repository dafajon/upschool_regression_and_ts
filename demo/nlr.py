import streamlit as st
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
import altair as alt

def content():

    st.title('Non-linear Components 🌊')
    
    st.markdown('Generalized Linear Form')
    st.latex(r'\mathbf{\hat{y}} = \alpha_1 \mathbf{x_1} + \alpha_1 \mathbf{x_2} + ... + \alpha_n \mathbf{x_n} + \beta')
    st.markdown('Single Feature')
    st.latex(r'\mathbf{\hat{y}} = \alpha_1 \mathbf{x} + \beta')
    st.markdown('Extend Feature Space With Non-linearity')
    st.latex(r'\hat{y} = \alpha \mathbf{x} + \alpha_2 \mathbf{x}^2 + ... \alpha_n \mathbf{x}^n')
    st.latex('or')
    st.latex(r'\hat{y} = \alpha \mathbf{x_1} + \alpha_2 sin(\mathbf{x}) + ... + \alpha_n cos(\mathbf{x})')
    
    
    
    st.markdown('## [Basis Functions](http://www.utstat.utoronto.ca/~radford/sta414.S11/week1b.pdf)')
    st.latex(r'\hat{y} = \sum_{i=1}^n \alpha_i \phi_i(\mathbf{x})')

        
    n_samples = 500
    n_val_samples = 50
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, 1)
    noise = rng.normal(loc=0.0, scale=2, size=n_samples)
    y = (5 * X[:, 0] + 3*np.sin(1 * np.pi * X[:, 0]) - noise)
    
    X_val = rng.randn(n_val_samples, 1)
    noise = rng.normal(loc=0.0, scale=2, size=n_val_samples)
    y_val = (5 * X_val[:, 0] + 3*np.sin(1 * np.pi * X_val[:, 0]) - noise)
    
    df = pd.DataFrame({'X':X.reshape(-1,),'y':y})
    val_df = pd.DataFrame({'X':X_val.reshape(-1,),'y':y_val})
    
    st.altair_chart(alt.Chart(df).mark_circle().encode(x='X',y='y') + alt.Chart(val_df).mark_circle(color = 'red').encode(x='X',y='y'))
    
    st.header('First fit linear regression...')
    
    with st.echo():
        from sklearn.linear_model import LinearRegression
        
        #Initialize model
        lr = LinearRegression()
        
        #Fit model
        lr.fit(X,y)
    
    st.write(lr)
    
    with st.echo():
        
        y_train_pred = lr.predict(X)
        y_valid_pred = lr.predict(X_val)
        
    show = st.button('Show Model')
    if show:
        df['pred'] = y_train_pred
        st.altair_chart(alt.Chart(df).mark_circle().encode(x='X',y='y') + alt.Chart(val_df).mark_circle(color = 'red').encode(x='X',y='y') + alt.Chart(df).mark_line(color = 'orange').encode(x='X',y='pred')) 
        
        with st.echo():
            from sklearn.metrics import mean_squared_error
            
            train_err = mean_squared_error(y,y_train_pred)
            valid_err = mean_squared_error(y_val,y_valid_pred)
            
        st.write(f'Training Error: {train_err}')
        st.write(f'Validation Error: {valid_err}')
        
        
    st.header('Adding Polynomial Basis')
        
    with st.echo():
        from sklearn.preprocessing import PolynomialFeatures
        
    degree = st.selectbox('Select Degree',[3,5,10,12,15,18,20])
    
    with st.echo():
        poly = PolynomialFeatures(degree=degree)
        
        X_poly = poly.fit_transform(X)
        X_val_poly = poly.fit_transform(X_val)
        
    st.write(X.shape,X_poly.shape)
           
        
    with st.echo():  
        #Initialize model
        lr_poly = LinearRegression()
        
        #Fit model
        lr_poly.fit(X_poly,y)
        
    
    with st.echo():
        
        y_train_poly_pred = lr_poly.predict(X_poly)
        y_valid_poly_pred = lr_poly.predict(X_val_poly)
        
        
    show_2 = st.button('Show Second Model')
    
    
    if show_2:
        df['pred_2'] = y_train_poly_pred
        st.altair_chart(alt.Chart(df).mark_circle().encode(x='X',y='y') + alt.Chart(val_df).mark_circle(color = 'red').encode(x='X',y='y') + alt.Chart(df).mark_line(color = 'orange').encode(x='X',y='pred_2')) 
        
        with st.echo():
            from sklearn.metrics import mean_squared_error
            
            train_err = mean_squared_error(y,y_train_poly_pred)
            valid_err = mean_squared_error(y_val,y_valid_poly_pred)
            
        st.write(f'Training Error: {train_err}')
        st.write(f'Validation Error: {valid_err}')

if __name__ == '__main__':
    content()