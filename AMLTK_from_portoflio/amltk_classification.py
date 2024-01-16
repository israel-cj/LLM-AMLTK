from typing import Any

import openml
import uuid
import posixpath
import joblib

from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from amltk.optimization import History, Metric, Trial
from amltk.optimization.optimizers.smac import SMACOptimizer
from amltk.pipeline import Component, Choice, Node, Sequential, Split, request
from amltk.scheduling import Scheduler
from amltk.sklearn import split_data
from amltk.store import PathBucket
# Our module
from .search_spaces import classification_search_space, LLM_generated_classification_search_space


def run_amltk(
        N_WORKERS = 32,
        partition = "thin",
        cores = 8,
        memory = "32 GB",
        walltime = 60,
        seed = 0,
        X=None,
        y=None,
        search_space=None,
):
    N_WORKERS = N_WORKERS
    scheduler = Scheduler.with_slurm(
        n_workers=N_WORKERS,  # Number of workers to launch
        queue=partition,  # Name of the queue to submit to
        cores=cores,  # Number of cores per worker
        memory= memory,  # Memory per worker
        walltime= "20:00:00",  # Walltime per worker # I think this is independent from the timeout
        # submit_command="sbatch --extra-arguments",  # Sometimes you need extra arguments to the launch command
    )

    # # Running in colab test
    # N_WORKERS = 2
    # scheduler = Scheduler.with_processes(N_WORKERS)

    splits = {"train": 0.6, "val": 0.2, "test": 0.2}
    seed = seed
    _y = LabelEncoder().fit_transform(y)
    data = split_data(X, _y, splits=splits, seed=seed)  # type: ignore

    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    if search_space:
        search_space = search_space
    else:
        search_space = classification_search_space
        # search_space = {**LLM_generated_classification_search_space, **classification_search_space}

    items = set(search_space.values())
    pipeline = (
            Sequential(name="Pipeline")
            >> Split(
        {
            "categorical": [SimpleImputer(strategy="constant", fill_value="missing"), OneHotEncoder(drop="first", handle_unknown='ignore')],
            "numerical": Component(SimpleImputer, space={"strategy": ["mean", "median"]}),
        },
        name="feature_preprocessing",
        )
        >> items
    )

    def target_function(trial: Trial, _pipeline: Node) -> Trial.Report:
        trial.store({"config.json": trial.config})
        with trial.profile("data-loading"):
            X_train, X_val, X_test, y_train, y_val, y_test = (
                trial.bucket["X_train.csv"].load(),
                trial.bucket["X_val.csv"].load(),
                trial.bucket["X_test.csv"].load(),
                trial.bucket["y_train.npy"].load(),
                trial.bucket["y_val.npy"].load(),
                trial.bucket["y_test.npy"].load(),
            )

        sklearn_pipeline = _pipeline.configure(trial.config).build("sklearn")

        with trial.begin():
            sklearn_pipeline.fit(X_train, y_train)

        if trial.exception:
            trial.store({"exception.txt": f"{trial.exception}\n {trial.traceback}"})
            return trial.fail()

        with trial.profile("predictions"):
            train_predictions = sklearn_pipeline.predict(X_train)
            val_predictions = sklearn_pipeline.predict(X_val)
            test_predictions = sklearn_pipeline.predict(X_test)

        with trial.profile("probabilities"):
            val_probabilites = sklearn_pipeline.predict_proba(X_val)

        with trial.profile("scoring"):
            train_acc = float(accuracy_score(train_predictions, y_train))
            val_acc = float(accuracy_score(val_predictions, y_val))
            test_acc = float(accuracy_score(test_predictions, y_test))

        trial.summary["train/acc"] = train_acc
        trial.summary["val/acc"] = val_acc
        trial.summary["test/acc"] = test_acc

        trial.store(
            {
                "model.pkl": sklearn_pipeline,
                "val_probabilities.npy": val_probabilites,
                "val_predictions.npy": val_predictions,
                "test_predictions.npy": test_predictions,
            },
        )

        return trial.success(accuracy=val_acc)

    bucket = PathBucket("results_model_amltk", clean=True, create=True)
    bucket.store(
        {
            "X_train.csv": X_train,
            "X_val.csv": X_val,
            "X_test.csv": X_test,
            "y_train.npy": y_train,
            "y_val.npy": y_val,
            "y_test.npy": y_test,
        },
    )

    metric = Metric("accuracy", minimize=False, bounds=(0.0, 1.0))
    optimizer = SMACOptimizer.create(
        space=pipeline,  #  (1)!
        metrics=metric,
        bucket=bucket,
        # seed=seed,
    )
    task = scheduler.task(target_function)


    @scheduler.on_start(repeat=N_WORKERS)
    def launch_initial_tasks() -> None:
        """When we start, launch `n_workers` tasks."""
        trial = optimizer.ask()
        task.submit(trial, _pipeline=pipeline)


    trial_history = History()


    @task.on_result
    def process_result_and_launc(_, report: Trial.Report) -> None:
        """When we get a report, print it."""
        trial_history.add(report)
        optimizer.tell(report)
        if scheduler.running():
            trial = optimizer.ask()
            task.submit(trial, _pipeline=pipeline)


    @task.on_cancelled
    def stop_scheduler_on_cancelled(_: Any) -> None:
        raise RuntimeError("Scheduler cancelled a worker!")

    try:
        scheduler.run(timeout=walltime)
    except Exception as e:
        print(e)
    history_df = trial_history.df()
    print(" ... Report ...")
    print(history_df)
    print("Number of models evaluated")
    print(len(history_df))

    trace = (
        trial_history.sortby("accuracy")
    )

    # best_model_id = 0 # They are sorted the best is the 0
    # trace[best_model_id].config
    # configured_pipeline = pipeline.configure(trace[best_model_id].config)
    # best_model = configured_pipeline.build(builder="sklearn")
    name_models = str(uuid.uuid4())[:5]
    list_models = []
    counter = 0
    for this_trace in trace:
        if str(this_trace.status) == 'success':
            this_config = this_trace.config
            configured_pipeline = pipeline.configure(this_config)
            model = configured_pipeline.build(builder="sklearn")
            this_name = f"{name_models}_{counter}"
            list_models.append((this_name, model))
            counter += 1

    # Create the ensemble model
    best_model = VotingClassifier(
        estimators=list_models,
        voting='soft'  # 'hard' for hard voting, 'soft' for weighted voting
    )

    return best_model, history_df, trial_history, metric, search_space, pipeline

if __name__ == "__main__":
    dataset = openml.datasets.get_dataset(990)  # 990 = 'eucalyptus'
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    run_amltk(
        N_WORKERS=32,
        partition="thin",
        cores=8,
        memory="32 GB",
        walltime=60,
        X=X,
        y=y,
        search_space=None,
    )

