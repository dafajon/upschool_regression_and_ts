import streamlit as st

import pandas as pd 
import numpy as np
from sklearn.datasets import make_regression


import altair as alt

def content():

    st.title('Linear Assumption 🤔')

    st.markdown('When life gives you lemons, you fit a model through them...')


    st.header('Generate Data')


    
    mu = st.slider('Mean',-100,100,5)
    std = st.slider('Std Dev',0.0,20.0,0.1)
    n_samples = int(st.text_input('Sample Size',value='1'))



    with st.echo():
        from sklearn.datasets import make_regression

        X,y = make_regression(n_informative=1,n_features=1,n_samples=n_samples,
                            bias=mu,noise=std,random_state=42)

        

    data = pd.DataFrame({'X':X.reshape(-1,),'y':y})
    plot = alt.Chart(data).mark_circle().encode(x='X',y='y')
    st.altair_chart(plot,use_container_width=True)
    print(data.head())



    #st.code(code,language='python')

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

    slope_min,slope_max =-100.,100.
    bias_min,bias_max = -100.,100.

    st.latex('\hat{y}_i = a x_i + b ')

    st.markdown("""
        * **Data:** observations 
        * **Model:** assumptions, inductive bias
    """)


    slope_default = (slope_min + slope_max)/2
    bias_default = (bias_min + bias_max)/2
    slope = st.slider('Slope (a)',min_value=slope_min,max_value=slope_max,value=slope_default,step=2.5)
    intercept = st.slider('Intercept (bias)',min_value=bias_min,max_value=bias_max,value=bias_default,step=2.5)

    data['fx'] = data['X']*slope + intercept

    line_fit_chart = alt.Chart(data).mark_circle().encode(x='X',y='y')+alt.Chart(data).mark_line(color='red').encode(x='X',y='fx')

   
    st.altair_chart(line_fit_chart,use_container_width=True)


    st.header('Problem of Over Determined System')
    st.markdown('* Adrressed by: ')
    st.image('resources/laplace.jpg',width=600)
    st.latex('y_i = a x_i + b + \epsilon')
    st.latex('\epsilon \sim \mathcal{N}(\mu,\,\sigma^{2})')

    st.header('How to approach to error term?')
    st.markdown('1 - Ordinary Least Squares')

    st.latex('\epsilon = y_i - \hat{y}_i')
    st.latex('= y_i - f(x_i)')
    st.latex('= y_i - ax_i - b')
    st.latex('argmin_{a^*,b^*}(y_i - ax_i - b)')

    st.markdown('2 - Probabilistic (Gaussian) Process')

    st.latex('y_i = a x_i + b + \mathcal{N}(\mu,\,\sigma^{2})')
    st.latex('\epsilon_i =  \mathcal{N}(y_i - ax_i -b,\,\sigma^{2})')
    st.latex(r'\frac{1}{\sqrt{2 \pi \sigma^2}} exp(-(y_i - ax_i -b)^2 / \sigma^2) = p(y_i | x_i, a,b)')

    

if __name__ == '__main__':
    content()