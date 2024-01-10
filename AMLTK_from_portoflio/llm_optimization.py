import json
import time
import uuid
import posixpath
from .run_llm_generated import run_llm_code
from openai import OpenAI

# client = OpenAI(api_key='')
client = OpenAI()

def build_prompt_from_df(
        acc,
        description_model,
        task,
        metric,
        name_model,
        number_recommendations=3,
):
    return """
You are assisting me with automated machine learning. Consider I need a 'Component' object, the next it is an example of a Component to illustrate:

{
  Component(
            name = 'model_porfolio_1',
            item = ExtraTreesClassifier,
            config = {
            "bootstrap": False,
            "criterion": "entropy",
            "max_depth": None,
            "max_features": 0.9565902080710877,
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0,
            "min_samples_leaf": 4,
            "min_samples_split": 15,
            "min_weight_fraction_leaf": 0.0,
            "random_state": 0,
            },
            space={},
        ),
  }, """ + f""""

Now,  the Component I need to optimize is a {description_model} for a {task} task. The {task} performance is measured using {metric}. 
Please suggest a list with {number_recommendations} diverse yet effective Components to initiate a Bayesian Optimization process. 
The 'name' of this component must be {name_model} from 0 to {number_recommendations}. 
All Components should be in a dictionary 'dict_components', i.e. """ + " 'dict_components = {" + f"'{name_model}_0': Component(), '{name_model}_1': Component() " + """}'
Each codeblock ends with "```end" and starts with "```python".
"""


def generate_code(messages, llm_model):
    completion = client.chat.completions.create(
        model=llm_model,
        messages=messages,
        stop=["```end"],
        temperature=0.5,
        max_tokens=1500,
    )
    code = completion.choices[0].message.content
    code = code.replace("```python", "").replace("```", "").replace("<end>", "")
    return code


def improve_models(
        history,
        task='classification',
        display_method="markdown",
        size_search_space=5,
        real_metric=None,
        llm_model="gpt-3.5-turbo",
        search_space=None,
        pipeline_space = None
):
    if task == 'classification':
        metric = "accuracy"
        trace = (
            history.sortby(metric)
        )
    else:
        metric = "r2_score"
        trace = (
            history.sortby(metric)
        )

    natural_descriptions_LIST = []
    for element in trace:
        this_configured_pipeline = pipeline_space.configure(element.config)
        this_model = this_configured_pipeline.build(builder="sklearn")
        name_in_dictionary = list(this_model.named_steps.keys())[-1]
        this_model_name = type(this_model[name_in_dictionary])
        print('this_model_name', this_model_name)
        try:
            this_component = search_space[name_in_dictionary]
            this_hyperparameters = this_component.config
            params_without_none = {k: v for k, v in this_hyperparameters.items() if v is not None}
            natural_description_model = f"{str(this_component.item)} , with the next hyperparameters {params_without_none}"
            natural_descriptions_LIST.append(natural_description_model)
        except Exception as e:
            print("Error: ")
            print(e)

    def format_for_display(code):
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code

    def execute_and_evaluate_code_block(code):
        try:
            new_search_space_exec = run_llm_code(code)
        except Exception as e:
            new_search_space_exec = None
            display_method(f"Error in code execution. {type(e)} {e}")
            display_method(f"```python\n{format_for_display(code)}\n```\n")
            return e, None
        return None, new_search_space_exec

    if display_method == "markdown":
        from IPython.display import display, Markdown
        display_method = lambda x: display(Markdown(x))
    else:
        display_method = print

    # Get a list of accuracies by ordering the history (same as we did with the trace)
    history_df = history.df()
    history_df = history_df.sort_values(f"metric:{real_metric}", ascending=False)
    list_accuracies = list(history_df[f"metric:{real_metric}"])

    counter_name = 0
    final_search_space = dict()
    for representation, acc in zip(natural_descriptions_LIST, list_accuracies):
        name_models = str(uuid.uuid4())[:5]  # Let's consider only the first 5 characters to keep it simple
        prompt = build_prompt_from_df(acc, representation, task, metric, name_model=name_models)
        print(prompt)
        messages = [
            {
                "role": "system",
                "content": "You are an expert datascientist assistant creating a dictionary of Components. You answer only by generating code. Let’s think step by step.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        try:
            code = generate_code(messages, llm_model)
            print('Code generated successfully')
            print(code)
        except Exception as e:
            code = None
            display_method("Error in LLM API." + str(e))
            time.sleep(60)  # Wait 1 minute before next request
            continue
        e, new_search_space = execute_and_evaluate_code_block(code)
        if new_search_space is not None:
            # final_search_space = final_search_space.union(new_search_space)
            final_search_space = {**final_search_space, **new_search_space}
        counter_name += 1
        if counter_name >= size_search_space:
            break

    return final_search_space
