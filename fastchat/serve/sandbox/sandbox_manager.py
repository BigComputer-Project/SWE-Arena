'''
Facades for interacting with the e2b sandbox.
'''

import json
from typing import Literal
from e2b import Sandbox
from e2b.sandbox.commands.command_handle import CommandExitException
from e2b.exceptions import TimeoutException
import time
import threading
from httpcore import ReadTimeout
import queue

from .constants import E2B_API_KEY, SANDBOX_TEMPLATE_ID, SANDBOX_NGINX_PORT, SANDBOX_RETRY_COUNT, SANDBOX_TIMEOUT_SECONDS


def create_sandbox(template: str = SANDBOX_TEMPLATE_ID) -> Sandbox:
    '''
    Create a new sandbox.
    Will retry if the sandbox creation fails.
    '''
    print("Creating new sandbox...")
    for attempt in range(1, SANDBOX_RETRY_COUNT + 1):
        try:
            return Sandbox(
                api_key=E2B_API_KEY,
                template=template,
                timeout=SANDBOX_TIMEOUT_SECONDS,
            )
        except Exception as e:
            if attempt < SANDBOX_RETRY_COUNT:
                time.sleep(1 * attempt)
            else:
                raise e
    raise RuntimeError("Failed to create sandbox after maximum attempts")


def reuse_or_create_sandbox(sandbox_id: str | None, template: str = SANDBOX_TEMPLATE_ID) -> Sandbox:
    '''
    Reuse an existing sandbox if it is running, otherwise create a new sandbox.
    '''
    sandbox = None

    if sandbox_id is not None:
        try:
            sandbox = Sandbox.connect(
                sandbox_id=sandbox_id,
                api_key=E2B_API_KEY,
            )
            if not sandbox.is_running(request_timeout=5):
                sandbox = None
        except Exception as e:
            pass

    if sandbox is not None:
        sandbox.set_timeout(timeout=SANDBOX_TIMEOUT_SECONDS)
    else:
        sandbox = create_sandbox(template=template)

    return sandbox


def run_command_in_sandbox(
        sandbox: Sandbox,
        command: str,
        working_directory: str | None = None,
        timeout: int = 60,
        print_output: bool = True,
    ) -> tuple[bool, list[str], list[str]]:
    '''
    Run a command in the sandbox.
    Return whether the command was successful and the stdout and stderr output.
    '''
    is_run_success = False
    stdouts: list[str] = []
    stderrs: list[str] = []

    try:
        command_result = sandbox.commands.run(
            cmd=command,
            cwd=working_directory,
            timeout=timeout,
            request_timeout=timeout + 5,
            on_stdout=lambda message: print(message) if print_output else None and stdouts.append(message),
            on_stderr=lambda message: print(message) if print_output else None and stderrs.append(message),
        )
        if command_result and command_result.exit_code == 0:
            is_run_success = True
    except Exception as e:
        stderrs.append(str(e))
        is_run_success = False

    return is_run_success, stdouts, stderrs


def install_pip_dependencies(sandbox: Sandbox, dependencies: list[str]) -> list[str]:
    '''
    Install pip dependencies in the sandbox.

    Return errors if any.
    '''
    install_errors = []
    if not dependencies:
        return install_errors

    for dependency in dependencies:
        try:
            sandbox.commands.run(
                f"uv pip install --system {dependency}",
                timeout=60 * 3,
                on_stdout=lambda message: print(message),
                on_stderr=lambda message: print(message),
            )
        except Exception as e:
            install_errors.append(f"Error during installing pip package {dependency}: {str(e)}")
            continue

    return install_errors


def parse_npm_package_name(package) -> tuple[str, str | None]:
    '''abc@123 -> abc, 123'''
    return package.split("@")[0], package.split("@")[1] if "@" in package else None


def is_npm_package_installed(package: str, installed_packages: dict[str, str | None]) -> bool:
    package_name, package_version = parse_npm_package_name(package)
    return package_name in installed_packages and (package_version is None or installed_packages[package_name] == package_version)


def get_installed_npm_packages(sandbox: Sandbox, project_root: str) -> dict[str, str | None]:
    installed_packages_raw = []
    sandbox.commands.run(
        "npm list --depth=0 --json",
        cwd=project_root,
        timeout=30,
        on_stdout=lambda message: installed_packages_raw.append(message),
    )
    lines = [json.loads(line)
             for line in installed_packages_raw if line.strip()]
    if not lines:
        return {}
    package_data = lines[-1]
    dependencies = package_data.get("dependencies", {})
    return {
        dep_name: details.get("version")
        for dep_name, details in dependencies.items()
    }


def install_npm_dependencies(sandbox: Sandbox, dependencies: list[str], project_root: str = '~') -> list[str]:
    '''
    Install npm dependencies in the sandbox.

    Return errors if any.
    '''
    install_errors = []
    if not dependencies:
        return install_errors

    installed_packages: dict[str, str | None] = get_installed_npm_packages(
        sandbox, project_root)

    dependencies_to_install = [dependency for dependency in dependencies if not is_npm_package_installed(
        dependency, installed_packages)]

    for dependency in dependencies_to_install:
        try:
            sandbox.commands.run(
                f"npm install {dependency} --prefer-offline --no-audit --no-fund --legacy-peer-deps",
                cwd=project_root,
                timeout=60 * 3,
                on_stdout=lambda message: print(message),
                on_stderr=lambda message: print(message),
            )
        except Exception as e:
            install_errors.append(f"Error during installing npm package {dependency}:" + str(e))
            continue

    return install_errors


def run_background_command_with_timeout(
    sandbox: Sandbox,
    command: str,
    cwd: str = "~",
    timeout: int = 5,
) -> str:
    """
    Run a command in the background and wait for a short time to check for startup errors.

    Args:
        sandbox: The sandbox instance
        command: The command to run
        cwd: The working directory for the command
        timeout: How long to wait for startup errors (in seconds)

    Returns:
        str: Any error output collected during startup
    """
    stderr = ""

    cmd = sandbox.commands.run(
        command,
        timeout=60 * 3,  # Overall timeout for the command
        cwd=cwd,
        background=True,
    )

    def wait_for_command(result_queue):
        nonlocal stderr
        try:
            result = cmd.wait()
            if result.stderr:
                stderr += result.stderr
            result_queue.put(stderr)
        except ReadTimeout:
            result_queue.put(stderr)
        except CommandExitException as e:
            stderr += "".join(e.stderr)
            result_queue.put(stderr)
        except TimeoutException:
            return

    result_queue = queue.Queue()
    wait_thread = threading.Thread(
        target=wait_for_command, args=(result_queue,))
    wait_thread.daemon = True  # Make thread daemon so it won't prevent program exit
    wait_thread.start()

    try:
        return result_queue.get(timeout=timeout)
    except queue.Empty:
        return stderr


def get_sandbox_app_url(
    sandbox: Sandbox,
    app_type: Literal["react", "vue", "html", "pygame"]
) -> str:
    '''
    Get the URL for the app in the sandbox with container wrapper.
    '''
    return f"https://{sandbox.get_host(port=SANDBOX_NGINX_PORT)}/container/?app={app_type}"
