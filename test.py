import os
import streamlit as st
from streamlit_ace import st_ace
import fire
import logging
from pathlib import Path


st.write(f"top of the code {st.session_state}")


if st.button('Say hello', key="0"):
     st.write('Why hello there')
else:
     st.write('Goodbye')

if st.selectbox('Select',["A","B"],  key="1" ):
     st.write('Why nothing selected')
else:
  st.write(f"selectbox {st.session_state}")

st.text_input("text",key="3")
