import streamlit as st
options = [["a","a1"],["b","b1"],['config.yaml', 'configs/rick-configs-1/config.yaml']]
options_show_basename = lambda opt: opt[0]
st.selectbox("select",options,format_func=options_show_basename )