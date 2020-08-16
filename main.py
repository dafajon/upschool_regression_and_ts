import streamlit as st
import datetime

from demo import *




titles = ("Welcome",
          "Linear Assumption",
          "Error Metric and Loss",
          "Non-linear Components",
          "Regression on Time Series")


callable_dict = {'Welcome': welcome,
                 'Linear Assumption': lr,
                 'Error Metric and Loss':err,
                 'Non-linear Components':nlr,
                 'Regression on Time Series':ts}

st.sidebar.title("Content")

st.sidebar.subheader("Today's Agenda")

part = st.sidebar.radio("", titles)

callable_dict.get(part, lambda: None)()