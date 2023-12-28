
import openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from AMLTK_from_portoflio import AMLTK_v1

dataset = openml.datasets.get_dataset(40983) # 40983 is Wilt dataset: https://www.openml.org/search?type=data&status=active&id=40983
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute
)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

### Setup and Run LLM pipeline - This will be billed to your OpenAI Account!
automl = AMLTK_v1(
    N_WORKERS=32,
    partition="thin",
    cores=8,
    memory="32 GB",
    walltime=60,
    task="classification"
    )

automl.fit(X_train, y_train)

# This process is done only once
y_pred = automl.predict(X_test)
acc = accuracy_score(y_pred, y_test)
print(f'LLM Pipeline accuracy {acc}')
