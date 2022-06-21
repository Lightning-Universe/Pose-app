import streamlit as st
import args_utils
from datetime import datetime
import sh

def set_script_args(*args, **kwargs):
  if 'key' in kwargs and kwargs['key'] == 'train_args':
    script_args_dict = args_utils.args_to_dict(st.session_state['train_args'])

    # hydra.run.out added if not already exists 
    if not('hydra.run.dir' in script_args_dict):
      run_date_time=datetime.today().strftime('%Y-%m-%d/%H-%M-%S')
      script_args_dict['hydra.run.dir'] = f"outputs/{run_date_time}"

    # change back to array
    st.session_state['train_args'] = args_utils.dict_to_args(script_args_dict)

def run(root_dir=".", context=st):
    options = args_utils.get_dir_of_dir(root_dir=root_dir, include="tb_logs")
    context.selectbox("train outputs", options=options, key="train_outputs")
    
    # edit the script_args
    if 'train_args' in st.session_state:
      train_args = st.session_state
    else:
      train_args = None

    context.text_area("Train Args", value=train_args, placeholder='Use K=V format. Use a=1 --b=2 NOT a 1 --b 2', on_change = set_script_args, key="train_args", kwargs={'key':'train_args'})

    context.text_area("Train Env", placeholder='a=1 b=2', on_change = set_script_args, key="script_env", kwargs={'key':'train_env'})

if __name__ == "__main__":
  run(root_dir="../lightning_pose")
   
