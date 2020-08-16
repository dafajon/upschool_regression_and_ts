import streamlit as st
import altair as alt
from sklearn.datasets import make_regression
import pandas as pd
import numpy as np

def content():

    st.title('Error and Loss(Objective Error) 🤬')


    st.header('Learning is optimization')

    X,y = make_regression(n_samples=100,n_features=1,n_informative=1,bias=38,noise=32,random_state=42)
    mock_data = pd.DataFrame({'input features (X)':X.reshape(-1,),'target output (y)':y})
    st.altair_chart(alt.Chart(mock_data).mark_circle().\
            encode(x='input features (X)',y='target output (y)'))

    st.markdown("""
    ## What is a linear model?
    """)
    st.markdown("""
    Relationship between input $X$ and output $y$ modeled as a function. $f(X) = y$

    **Goal:** Find an approriate $f(.)$

    **Assumption:** $f$ is a linear function.  
    * $f(X) = a_1x_1 + a_2x_2 + ... + a_nx_n + bias = y$ 
    * In above simple case $ax + bias = y$
    """)

    slope_min,slope_max =-100,100
    bias_min,bias_max = -100,100
    slope_default = int((slope_min + slope_max)/2)
    bias_default = int((bias_min + bias_max)/2)
    slope = st.slider('Slope (a)',min_value=slope_min,max_value=slope_max,value=slope_default,step=5)
    intercept = st.slider('Intercept (bias)',min_value=bias_min,max_value=bias_max,value=bias_default,step=5)

    mock_data['fx'] = mock_data['input features (X)']*slope + intercept

    st.altair_chart(alt.Chart(mock_data).mark_circle().encode(x='input features (X)',y='target output (y)')+
                    alt.Chart(mock_data).mark_line(color='red').encode(x='input features (X)',y='fx'))


    st.markdown("""
    ## Measuring What is Good
    """)
    st.markdown(r"""
    $f(x) = \hat{y}$

    $\hat{y}$ -> model output given $X$

    $y$ -> Actual target values in data

    $Objective Error = \frac{1}{N} \Sigma_{i=0}^{N}|y_i-\hat{y}_i|$

    """)
    y = mock_data['target output (y)']
    y_hat = mock_data['fx']
    st.write(f'Error = {np.mean(np.abs(y - y_hat))}')


    st.markdown("""
    #### Mean Absolute Error in parameter space
    """)
    def abs_err_given_slope(s):
        y_hat = mock_data['input features (X)']*s + intercept
        y = mock_data['target output (y)']
        err = np.sum(np.abs(y - y_hat))
        return err

    slopes = np.linspace(-100,100)
    errors = np.array([abs_err_given_slope(s) for s in slopes])

    err_df = pd.DataFrame()
    err_df['slope (a)'] = slopes
    err_df['mean absolute error'] = errors


    st.altair_chart(
        alt.Chart(err_df).mark_line().encode(x='slope (a)',y='mean absolute error'),
        use_container_width = True
    )

    st.markdown("""
    #### Mean Squared Error in parameter space
    """)

    st.markdown(r"""
        $f(x) = \hat{y}$

        $\hat{y}$ -> model output given $X$

        $y$ -> Actual target values in data

        $Objective Error = \frac{1}{N} \Sigma_{i=0}^{n}(y_i-\hat{y}_i)^2$

        """)

    st.write(f'Error = {np.mean((y - y_hat)**2)}')

    def sqr_err_given_slope(s):
        y_hat = mock_data['input features (X)']*s + intercept
        y = mock_data['target output (y)']
        err = np.sum(np.abs(y - y_hat)**2)
        return err

    errors = np.array([sqr_err_given_slope(s) for s in slopes])

    sq_err_df = pd.DataFrame()
    sq_err_df['slope (a)'] = slopes
    sq_err_df['mean squared error'] = errors
    
    st.altair_chart(
        alt.Chart(sq_err_df).mark_line().encode(x='slope (a)',y='mean squared error'),
        use_container_width = True
    )


    

if __name__ == '__main__':
    content()