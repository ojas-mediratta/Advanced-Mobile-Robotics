import sys
from typing import Callable

def verify(function: Callable) -> str:
    """Will indicate with a print statement whether assertions passed or failed
    within function argument call.
    Args:
        function: Python function object
    Returns:
        string that is colored red or green when printed, indicating success
    """
    try:
        function()
        return '\x1b[32m"Correct"\x1b[0m'
    except AssertionError:
        return '\x1b[31m"Wrong"\x1b[0m'
    
    
def setup_path():
    if "./controllers/ROV_controller" not in sys.path:
        sys.path.append("./controllers/ROV_controller")

