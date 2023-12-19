import pickle
import csv
import json
import os
from openai import OpenAI
# from run_llm_code import run_llm_code

os.environ["OPENAI_API_KEY"] = "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"

client = OpenAI()


def get_prompt(set_components, pipeline):
    return f""""
    An object "component" will be created. Take as reference the next set of "Components" :
    {set_components}
    
    Now, given the next dictionary use it to create a valid Component:
    {pipeline}

    Code formatting for each Component created:
    ```python
    from amltk.pipeline import Component
    (import all the sklearn packages necessaries)

    component = Component(...)

    ```end

    Each codeblock generates exactly one useful component. 
    Each codeblock ends with "```end" and starts with "```python".
"""

def generate_search_space(
        pipeline=None,
        model="gpt-4",
        just_print_prompt=False,
        display_method="markdown",
):
    global list_pipelines # To make it available to sklearn_wrapper in case the time out is reached
    def format_for_display(code):
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code

    if display_method == "markdown":
        from IPython.display import display, Markdown

        display_method = lambda x: display(Markdown(x))
    else:

        display_method = print

    set_components_str = """{
        Component(
            SVC,
            config={"probability": True, "random_state": request("seed")},
            space={"C": (0.1, 10.0)},
        ),
        Component(
            RandomForestClassifier,
            config={"random_state": request("seed")},
            space={"n_estimators": (10, 100), "criterion": ["gini", "log_loss"]},
        ), 
        Component(
            MLPClassifier,
            config={"random_state": request("seed")},
            space={
                "activation": ["identity", "logistic", "relu"],
                "alpha": (0.0001, 0.1),
                "learning_rate": ["constant", "invscaling", "adaptive"],
            },
        ), 
      }
      """
    prompt = get_prompt(set_components_str, pipeline)

    if just_print_prompt:
        code, prompt = None, prompt
        return code, prompt, None

    def generate_code(messages):
        if model == "skip":
            return ""

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            stop=["```end"],
            temperature=0.5,
            max_tokens=500,
        )
        code = completion.choices[0].message.content
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code

    messages = [
        {
            "role": "system",
            "content": "Your job is to convert a dictionary into a valid 'Component'. You answer only by generating code. Let’s think step by step.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    e = 1
    while e != None:
        try:
            e = None
            code = generate_code(messages)
        except Exception as e:
            display_method("Error in LLM API." + str(e))
    return code
        # try:
        #     this_component = run_llm_code(
        #         code,
        #     )
        # except Exception as e:
        #     display_method("Error in LLM API." + str(e))
        #
        # if e is None:
        #
        #     display_method("Code generated by LLM API:")
        #     display_method("```python\n" + format_for_display(code) + "\n```")
        #
        #     display_method("Component generated by LLM API:")
        #     display_method(this_component)
        #     return this_component

if __name__ == '__main__':
    with open('D:/PhD_third year/AutoML/LLM/LLM-AMLTK/RF_None_10CV_iterative_es_if.json', 'r') as json_file:
        loaded_dict = json.load(json_file)

    # list_components = []
    port = loaded_dict["portfolio"]
    counter_pipelines = 0
    for pipeline in port:
        this_component = generate_search_space(port[pipeline])
        with open(f'components_list{counter_pipelines}.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([this_component])
        counter_pipelines+=1
        # list_components.append(this_component)

    # with open('list_components.pickle', 'wb') as handle:
    #     pickle.dump(list_components, handle, protocol=pickle.HIGHEST_PROTOCOL)