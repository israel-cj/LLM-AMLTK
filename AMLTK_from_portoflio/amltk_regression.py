from typing import Any

import openml
import posixpath
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from amltk.optimization import History, Metric, Trial
from amltk.optimization.optimizers.smac import SMACOptimizer
from amltk.pipeline import Component, Choice, Node, Sequential, Split, request
from amltk.scheduling import Scheduler
from amltk.sklearn import split_data
from amltk.store import PathBucket


def run_amltk_regressor(
        N_WORKERS = 32,
        partition = "thin",
        cores = 8,
        memory = "32 GB",
        walltime = 60,
        seed = 0,
        X=None,
        y=None,
):
    N_WORKERS = N_WORKERS
    scheduler = Scheduler.with_slurm(
        n_workers=N_WORKERS,  # Number of workers to launch
        queue=partition,  # Name of the queue to submit to
        cores=cores,  # Number of cores per worker
        memory= memory,  # Memory per worker
        walltime= "30:00:00",  # Walltime per worker # I think this is independen from the timeout
        # submit_command="sbatch --extra-arguments",  # Sometimes you need extra arguments to the launch command
    )

    splits = {"train": 0.6, "val": 0.2, "test": 0.2}
    seed = seed
    _y = LabelEncoder().fit_transform(y)
    data = split_data(X, _y, splits=splits, seed=seed)  # type: ignore

    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    pipeline = (
            Sequential(name="Pipeline")
            >> Split(
        {
            "categorical": [
                SimpleImputer(strategy="constant", fill_value="missing"),
                OneHotEncoder(drop="first", handle_unknown='ignore'),
            ],
            "numerical": Component(
                SimpleImputer,
                space={"strategy": ["mean", "median"]},
            ),
        },
        name="feature_preprocessing",
    )
            >> Choice(
        Component(
            name='model_regressor_porfolio_1',
            item=RandomForestRegressor,
            config={
                "max_depth": 15,
                "criterion": "squared_error",
            },
            space={},
        ),
        Component(
            name='model_regressor_porfolio_2',
            item=ExtraTreesRegressor,
            config={
                "min_samples_leaf": 15,
                "criterion": "squared_error",
            },
            space={},
        ),
        Component(
            name='model_regressor_porfolio_3',
            item=RandomForestRegressor,
            config={
                "min_samples_leaf": 5,
                "max_leaf_nodes": 50000,
                "max_features": 0.5,
            },
            space={},
        ),
        Component(
            name='model_regressor_porfolio_4',
            item=ExtraTreesRegressor,
            config={
                "min_samples_leaf": 1,
                "max_leaf_nodes": 15000,
                "max_features": 0.5,
            },
            space={},
        ),
        # The next components come from https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/configs/zeroshot/zeroshot_portfolio_2023.py
        # We assume that the criterion should be fixed and then we can play with the rest of the hyperparameters of the XT (extra trees) and RF (random forest)
        Component(
            name='model_regressor_porfolio_5',
            item=RandomForestRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 0.75,
                "max_leaf_nodes": 37308,
                "min_samples_leaf": 1,
            },
            space={},
        ),
        Component(
            name='model_regressor_porfolio_6',
            item=RandomForestRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 0.75,
                "max_leaf_nodes": 28310,
                "min_samples_leaf": 2,
            },
            space={},
        ),
        Component(
            name='model_regressor_porfolio_7',
            item=RandomForestRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 1.0,
                "max_leaf_nodes": 38572,
                "min_samples_leaf": 5,
            },
            space={},
        ),
        Component(
            name='model_regressor_porfolio_8',
            item=RandomForestRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 0.75,
                "max_leaf_nodes": 18242,
                "min_samples_leaf": 40,
            },
            space={},
        ),
        Component(
            name='model_regressor_porfolio_9',
            item=RandomForestRegressor,
            config={
                "criterion": "squared_error",
                "max_features": "log2",
                "max_leaf_nodes": 42644,
                "min_samples_leaf": 1,
            },
            space={},
        ),
        Component(
            name='model_regressor_porfolio_10',
            item=RandomForestRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 0.75,
                "max_leaf_nodes": 36230,
                "min_samples_leaf": 3,
            },
            space={},
        ),
        Component(
            name='model_regressor_porfolio_11',
            item=RandomForestRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 1.0,
                "max_leaf_nodes": 48136,
                "min_samples_leaf": 1,
            },
            space={},
        ),
        Component(
            name='model_regressor_porfolio_12',
            item=ExtraTreesRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 0.75,
                "max_leaf_nodes": 18392,
                "min_samples_leaf": 1,
            },
            space={},
        ),
        Component(
            name='model_regressor_porfolio_13',
            item=ExtraTreesRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 0.75,
                "max_leaf_nodes": 18392,
                "min_samples_leaf": 1,
            },
            space={},
        ),
        Component(
            name='model_regressor_porfolio_14',
            item=ExtraTreesRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 1.0,
                "max_leaf_nodes": 12845,
                "min_samples_leaf": 4,
            },
            space={},
        ),
        Component(
            name='model_regressor_porfolio_15',
            item=ExtraTreesRegressor,
            config={
                "criterion": "squared_error",
                "max_features": "sqrt",
                "max_leaf_nodes": 28532,
                "min_samples_leaf": 1,
            },
            space={},
        ),
        Component(
            name='model_regressor_porfolio_16',
            item=ExtraTreesRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 1.0,
                "max_leaf_nodes": 19935,
                "min_samples_leaf": 20,
            },
            space={},
        ),
        Component(
            name='model_regressor_porfolio_17',
            item=ExtraTreesRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 0.75,
                "max_leaf_nodes": 29813,
                "min_samples_leaf": 4,
            },
            space={},
        ),
        Component(
            name='model_regressor_porfolio_18',
            item=ExtraTreesRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 1.0,
                "max_leaf_nodes": 40459,
                "min_samples_leaf": 1,
            },
            space={},
        ),
        Component(
            name='model_regressor_porfolio_19',
            item=ExtraTreesRegressor,
            config={
                "criterion": "squared_error",
                "max_features": "sqrt",
                "max_leaf_nodes": 29702,
                "min_samples_leaf": 2,
            },
            space={},
        ),
        name="Regressor"
    )
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

        with trial.profile("scoring"):
            train_score = float(r2_score(train_predictions, y_train))
            val_score = float(r2_score(val_predictions, y_val))
            test_score = float(r2_score(test_predictions, y_test))

        trial.summary["train/score"] = train_score
        trial.summary["val/score"] = val_score
        trial.summary["test/score"] = test_score

        trial.store(
            {
                "model.pkl": sklearn_pipeline,
                "val_predictions.npy": val_predictions,
                "test_predictions.npy": test_predictions,
            },
        )

        return trial.success(r2_score=val_score)

    bucket = PathBucket("results_model_regression_amltk", clean=True, create=True)
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


    optimizer = SMACOptimizer.create(
        space=pipeline,  #  (1)!
        metrics=Metric("r2_score", minimize=False, bounds=(0.0, 1.0)),
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
        trial_history.sortby("r2_score")
    )
    best_trace = trace[0]
    best_bucket = best_trace.bucket
    new_path = posixpath.join(best_bucket.path, best_trace.name, 'model.pkl')
    print(new_path)
    best_model = joblib.load(str(new_path))

    return best_model, history_df

if __name__ == "__main__":
    dataset = openml.datasets.get_dataset(41021)  # 990 = 'eucalyptus'
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    run_amltk_regressor(
        N_WORKERS=32,
        partition="thin",
        cores=8,
        memory="32 GB",
        walltime=60,
        X=X,
        y=y,
    )

