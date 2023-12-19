import copy
import numpy as np
import ast
from typing import Any, Dict, Optional

def run_llm_code(code):
    """
    """	
    try:
        output = {}
        exec(code, output)
        this_component = output['this_component']
        print(this_component)

    except Exception as e:
        print("Code could not be executed", e)
        raise (e)

    return this_component

