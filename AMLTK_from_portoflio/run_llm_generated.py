from amltk.pipeline import Component

def run_llm_code(code):
  """
  Executing the new search space
  """
  try:
    globals_dict = {'Component': Component}
    output = {}
    exec(code, globals_dict, output)
    dict_components = output['dict_components']

  except Exception as e:
    dict_components = {}
    print("Code could not be executed", e)
    raise (e)

  return dict_components