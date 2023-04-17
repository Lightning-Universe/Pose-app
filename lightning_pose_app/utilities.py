from lightning.app.frontend import StreamlitFrontend as LitStreamlitFrontend
import shlex


def args_to_dict(script_args: str) -> dict:
    """convert str to dict A=1 B=2 to {'A':1, 'B':2}"""
    script_args_dict = {}
    for x in shlex.split(script_args, posix=False):
        try:
            k, v = x.split("=", 1)
        except:
            k = x
            v = None
        script_args_dict[k] = v
    return script_args_dict


def dict_to_args(script_args_dict: dict) -> str:
    """convert dict {'A':1, 'B':2} to str A=1 B=2 to """
    script_args_array = []
    for k,v in script_args_dict.items():
        script_args_array.append(f"{k}={v}")
    # return as a text
    return " \n".join(script_args_array)


class StreamlitFrontend(LitStreamlitFrontend):
    """Provide helpful print statements for where streamlit tabs are forwarded."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def start_server(self, *args, **kwargs):
        super().start_server(*args, **kwargs)
        try:
            print(f"Running streamlit on http://{kwargs['host']}:{kwargs['port']}")
        except:
            # on the cloud, args[0] = host, args[1] = port
            pass
