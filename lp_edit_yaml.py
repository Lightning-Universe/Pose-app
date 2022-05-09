import os
import streamlit as st
import fire
import logging

def file_selector(form, dir='.'):
  filenames = os.listdir(dir)
  selected_filename = form.selectbox('Select a file', filenames)
  return os.path.join(dir, selected_filename)

def run(dir='.'):
  """Display YAML file and return arguments and env variable fields

  :dir (str): dir name to pull the yaml file from
  :return (dict): {'script_arg':script_arg, 'script_env':script_env}
  """

  form = st.form(key='script_param')

  script_arg = form.text_input('Script Args',placeholder="training.max_epochs=11")
  script_env = form.text_input('Script Env Vars')

  filename = file_selector(form=form, dir=dir)
  form.write('You selected `%s`' % filename)
  submit_button = form.form_submit_button(label='Submit')

  try:
    with open(filename) as input:
      content_raw = input.read()
      form.code(content_raw, language="yaml")
  except FileNotFoundError:
    form.error('File not found.')
  except Exception:  
    form.error("can't process.")


  #if submit_button:
  #  return({'script_arg':script_arg, 'script_env':script_env})
  
if __name__ == '__main__':
  fire.Fire(run)
