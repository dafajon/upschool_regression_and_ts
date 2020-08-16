import streamlit as st

def content():
    st.title('Regression for Time Series Forecasting 🔮')

    st.header('What is **"streamlit"** ?')
    st.markdown("Tool for designing interactive apps that host Data Science and Machine Learning work.")
    st.markdown(""" 
                * Pure python
                * Widgets
                * Markdown, Latex and code rendering
    """)
    
    st.title("Today's Content")
    st.markdown('* Introduction to Linear Models')
    st.markdown('* Metric and Loss')
    st.markdown('* Inducing Non-linearity')
    st.markdown('* Time Series Example')


if __name__ == '__main__':
    content()