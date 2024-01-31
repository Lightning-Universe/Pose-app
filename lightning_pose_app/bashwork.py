import errno
from functools import partial
from lightning.app import LightningWork
from lightning.app.utilities.app_helpers import _collect_child_process_pids
import logging
import os
import shlex
import signal
import socket
from string import Template
import subprocess
import threading
import time

from lightning_pose_app.utilities import args_to_dict, WorkWithFileSystem


_logger = logging.getLogger('APP.BASHWORK')


def add_to_system_env(env_key='env', **kwargs) -> dict:
    """add env to the current system env"""
    new_env = None
    if env_key in kwargs:
        env = kwargs[env_key]
        if isinstance(env, str):
            env = args_to_dict(env)
        if not (env is None) and not (env == {}):
            new_env = os.environ.copy()
            new_env.update(env)
    return new_env


def is_port_in_use(host: str, port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        in_use = False
    except socket.error as e:
        in_use = True
        if e.errno == errno.EADDRINUSE:
            _logger.debug("Port is already in use")
        else:
            # something else raised the socket.error exception
            _logger.debug(e)

    s.close()
    return in_use


def work_calls_len(lwork: LightningWork):
    """get the number of call in state dict. state dict has current and past calls to work."""
    # reduce by 1 to remove latest_call_hash entry
    return len(lwork.state["calls"]) - 1


def work_is_free(lwork: LightningWork):
    """work is free to accept new calls.
    this is expensive when a lot of calls accumulate over time
    work is when there is there is no pending and running calls at the moment
    pending status is verified by examining each call history looking for anything call that
    is pending history status.stage is not reliable indicator as there is delay registering
    new calls status.stage shows SUCCEEDED even after 3 more calls are accepted in parallel mode
    """
    status = lwork.status
    state = lwork.state
    # more than one work can started this way
    # there is work assignment and status update
    # multiple works are queued but
    # count run that are in pending state
    if (
        status.stage == "not_started" or
        status.stage == "succeeded" or
        status.stage == "failed"
    ):
        # do not run if jobs are in pending state
        # not counting to reduce CPU load as looping thru all of the calls can get expensive
        for c in state["calls"]:
            if c == 'latest_call_hash':
                continue
            if len(state["calls"][c]['statuses']) == 1:
                return False
        return True
    # must in pending or running or stopped.
    else:
        return False


class LitBashWork(WorkWithFileSystem):

    def __init__(
        self,
        *args,
        name="bashwork",
        wait_seconds_after_run=10,
        wait_seconds_after_kill=10,
        **kwargs
    ):

        # required to to grab self.host and self.port in the cloud.
        # otherwise, the values flips from 127.0.0.1 to 0.0.0.0 causing two runs
        # host='0.0.0.0',
        super().__init__(*args, name=name, **kwargs)
        self.wait_seconds_after_run = wait_seconds_after_run
        self.wait_seconds_after_kill = wait_seconds_after_kill

        self.pid = None
        self.exit_code = None
        self.stdout = None
        self.inputs = None
        self.outputs = None
        self.args = ""

        self._wait_proc = None

    def reset_last_args(self) -> str:
        self.args = ""

    def reset_last_stdout(self) -> str:
        self.stdout = None

    def last_args(self) -> str:
        return self.args

    def last_stdout(self):
        return self.stdout

    def on_before_run(self):
        """Called before the python script is executed."""
        pass

    def on_after_run(self):
        """Called after python script executes. Wrap outputs in Path so they will be available"""
        pass

    # statistics on this work
    def work_is_free(self) -> bool:
        return work_is_free(self)

    def work_calls_len(self) -> int:
        return work_calls_len(self)

    def popen_wait(self, cmd, save_stdout, exception_on_error, **kwargs):
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            close_fds=True,
            shell=True,
            executable='/bin/bash',
            **kwargs
        ) as self._wait_proc:
            if self._wait_proc.stdout:
                with self._wait_proc.stdout:
                    for line in iter(self._wait_proc.stdout.readline, b""):
                        # logger.info("%s", line.decode().rstrip())
                        line = line.decode().rstrip()
                        if save_stdout:
                            if self.stdout is None:
                                self.stdout = []
                            self.stdout.append(line)

        if exception_on_error and self.exit_code != 0:
            raise Exception(self.exit_code)

    def popen_nowait(self, cmd, **kwargs):
        proc = subprocess.Popen(
            cmd,
            shell=True,
            executable='/bin/bash',
            close_fds=True,
            **kwargs
        )
        self.pid = proc.pid

    def subprocess_call(
        self,
        cmd,
        save_stdout=True,
        exception_on_error=False,
        venv_name="",
        wait_for_exit=True,
        timeout=0,
        **kwargs
    ):
        """run the command"""
        # replace host and port
        cmd = Template(cmd).substitute({'host': self.host, 'port': self.port})
        # convert multiline to a single line
        cmd = ' '.join(shlex.split(cmd))
        _logger.info(cmd, kwargs)
        kwargs['env'] = add_to_system_env(**kwargs)
        if venv_name:
            cmd = f"source ~/{venv_name}/bin/activate; which python; {cmd}; deactivate"

        if wait_for_exit:
            _logger.debug("wait popen")
            # start the thread
            target = partial(
                self.popen_wait, cmd, save_stdout=save_stdout,
                exception_on_error=exception_on_error, **kwargs)
            thread = threading.Thread(target=target)
            thread.start()
            # tr
            thread.join(timeout)
            if thread.is_alive() and timeout > 0:
                _logger.debug(f"terminating after waiting {timeout}")
                self._wait_proc.terminate()
            # should either wait, process is already done, or kill
            thread.join()
            # _logger.debug(self._wait_proc.returncode)
            _logger.debug("wait completed", cmd)
        else:
            _logger.debug("no wait popen")
            self.popen_nowait(cmd, **kwargs)
            _logger.debug("no wait completed", cmd)

    def run(
        self,
        args,
        venv_name="",
        save_stdout=False,
        wait_for_exit=True,
        input_output_only=False,
        kill_pid=False,
        inputs=[],
        outputs=[],
        run_after_run=[],
        timeout=0,
        timer=0,  # added for uniqueness and caching
        **kwargs
    ):

        _logger.debug(args, kwargs)

        # pre processing
        self.on_before_run()
        self.get_from_drive(inputs)
        self.args = args
        self.stdout = None

        # run the command
        if not input_output_only:

            # kill previous process
            if self.pid and kill_pid:
                _logger.debug(f"***killing {self.pid}")
                os.kill(self.pid, signal.SIGTERM)
                info = os.waitpid(self.pid, 0)
                while is_port_in_use(self.host, self.port):
                    _logger.debug(f"***killed. pid {self.pid} waiting to free port")
                    time.sleep(self.wait_seconds_after_kill)

            # start a new process
            self.subprocess_call(
                cmd=args, venv_name=venv_name, save_stdout=save_stdout,
                wait_for_exit=wait_for_exit, **kwargs)

        # Hack to get info after the run that can be passed to Flow
        for cmd in run_after_run:
            self.popen_wait(cmd, save_stdout=True, exception_on_error=False, **kwargs)

        # post processing
        self.put_to_drive(outputs)
        # give time for REDIS to catch up and propagate self.stdout back to flow
        if save_stdout:
            _logger.debug(f"waiting work to flow message sleeping {self.wait_seconds_after_run}")
            time.sleep(self.wait_seconds_after_run)

        # regular hook
        self.on_after_run()

    def on_exit(self):
        for child_pid in _collect_child_process_pids(os.getpid()):
            os.kill(child_pid, signal.SIGTERM)
