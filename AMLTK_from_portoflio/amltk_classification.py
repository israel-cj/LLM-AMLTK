from typing import Any

import openml
import posixpath
import joblib

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from amltk.optimization import History, Metric, Trial
from amltk.optimization.optimizers.smac import SMACOptimizer
from amltk.pipeline import Component, Choice, Node, Sequential, Split, request
from amltk.scheduling import Scheduler
from amltk.sklearn import split_data
from amltk.store import PathBucket


def run_amltk(
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
        walltime= "05:00:00",  # Walltime per worker # I think this is independen from the timeout
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
            "categorical": [SimpleImputer(strategy="constant", fill_value="missing"), OneHotEncoder(drop="first")],
            "numerical": Component(SimpleImputer, space={"strategy": ["mean", "median"]}),
        },
        name="feature_preprocessing",
    )
            >> Choice(
        Component(
            SVC,
            config={"probability": True, "random_state": 0},
            space={"C": (0.1, 10.0)},
        ),
        Component(
            RandomForestClassifier,
            config={"random_state": 0},
            space={"n_estimators": (10, 100), "criterion": ["gini", "log_loss"]},
        ),
        Component(
            MLPClassifier,
            config={"random_state": 0},
            space={
                "activation": ["identity", "logistic", "relu"],
                "alpha": (0.0001, 0.1),
                "learning_rate": ["constant", "invscaling", "adaptive"],
            },
        ),
        Component(
            name='model_porfolio_1',
            item=ExtraTreesClassifier,
            config={
                "bootstrap": False,
                "criterion": "entropy",
                "max_depth": None,
                "max_features": 0.9565902080710877,
                "max_leaf_nodes": None,
                "min_impurity_decrease": 0.0,
                "min_samples_leaf": 4,
                "min_samples_split": 15,
                "min_weight_fraction_leaf": 0.0,
                "random_state": 0
            },
            space={},
        ),
        Component(
            name='model_porfolio_2',
            item=GradientBoostingClassifier,
            config={
                "learning_rate": 0.19595673731599184,
                "min_samples_leaf": 2,
                "max_depth": None,  # 'None' in the dictionary is a string, it should be None
                "max_leaf_nodes": 10,
            },
            space={},
        ),
        Component(
            name='model_porfolio_3',
            item=SGDClassifier,
            config={
                "alpha": 8.445149920482102e-05,
                "average": True,  # 'True' in the dictionary is a string, it should be a boolean
                "fit_intercept": True,  # 'True' in the dictionary is a string, it should be a boolean
                "learning_rate": "optimal",
                "loss": "log_loss",  # change from  hinge
                "penalty": "l2",
                "tol": 0.017561035108251574,
            },
            space={},
        ),
        Component(
            name='model_porfolio_4',
            item=GradientBoostingClassifier,
            config={
                "learning_rate": 0.19298903469131437,
                "min_samples_leaf": 20,
                "max_depth": None,  # 'None' in the dictionary is a string, it should be None
                "max_leaf_nodes": 10,
                "tol": 1e-07,
                "n_iter_no_change": 6,
                "validation_fraction": 0.11549557161401015,
            },
            space={},
        ),
        Component(
            name='model_porfolio_5',
            item=GradientBoostingClassifier,
            config={
                "learning_rate": 0.07043555880113304,
                "min_samples_leaf": 7,
                "max_depth": None,  # 'None' in the dictionary is a string, it should be None
                "max_leaf_nodes": 698,
                "tol": 1e-07,
                "n_iter_no_change": 1,
                "validation_fraction": 0.16202792486532844,
            },
            space={},
        ),
        Component(
            name='model_porfolio_6',
            item=MLPClassifier,
            config={
                "activation": "relu",
                "alpha": 0.062143149732377403,
                "batch_size": "auto",
                "beta_1": 0.9,
                "beta_2": 0.999,
                "early_stopping": False,  # 'train' in the dictionary is not a valid value, it should be a boolean
                "epsilon": 1e-08,
                "hidden_layer_sizes": (59, 59),  # assuming 'hidden_layer_depth' means 2 layers of 59 nodes each
                "learning_rate_init": 0.03451304647704975,
                "n_iter_no_change": 32,
                "shuffle": True,  # 'True' in the dictionary is a string, it should be a boolean
                "solver": "adam",
                "tol": 0.0001,
            },
            space={},
        ),
        Component(
            name='model_porfolio_7',
            item=GradientBoostingClassifier,
            config={
                "learning_rate": 0.013391089474609338,
                "min_samples_leaf": 138,
                "max_depth": None,  # 'None' in the dictionary is a string, it should be None
                "max_leaf_nodes": 3,
                "tol": 1e-07,
            },
            space={},
        ),
        Component(
            name='model_porfolio_8',
            item=SGDClassifier,
            config={
                "alpha": 3.7007715928095603e-06,
                "average": False,  # 'False' in the dictionary is a string, it should be a boolean
                "fit_intercept": True,  # 'True' in the dictionary is a string, it should be a boolean
                "learning_rate": "constant",
                "loss": "log_loss",  # changed from perceptron
                "penalty": "elasticnet",
                "tol": 0.04024454652952893,
                "eta0": 2.0514805804594057e-06,
                "l1_ratio": 7.163773086996508e-09,
            },
            space={},
        ),
        Component(
            name='model_porfolio_9',
            item=GradientBoostingClassifier,
            config={
                "learning_rate": 0.7615407561275519,
                "min_samples_leaf": 136,
                "max_depth": None,  # 'None' in the dictionary is a string, it should be None
                "max_leaf_nodes": 34,
                "tol": 1e-07,
                "n_iter_no_change": 11,
            },
            space={},
        ),
        Component(
            name='model_porfolio_10',
            item=SGDClassifier,
            config={
                "alpha": 2.781693052845358e-05,
                "average": False,  # 'False' in the dictionary is a string, it should be a boolean
                "fit_intercept": True,  # 'True' in the dictionary is a string, it should be a boolean
                "learning_rate": "constant",
                "loss": "modified_huber",
                "penalty": "elasticnet",
                "tol": 3.467206367066193e-05,
                "epsilon": 0.009309372110873482,
                "eta0": 9.309855728417528e-05,
                "l1_ratio": 0.6620356750678396,
            },
            space={},
        ),
        Component(
            name='model_porfolio_11',
            item=GradientBoostingClassifier,
            config={
                "learning_rate": 0.7962804610609661,
                "min_samples_leaf": 21,
                "max_depth": None,  # 'None' in the dictionary is a string, it should be None
                "max_leaf_nodes": 183,
                "tol": 1e-07,
                "n_iter_no_change": 3,
            },
            space={},
        ),
        Component(
            name='model_porfolio_12',
            item=SGDClassifier,
            config={
                "alpha": 0.03522482415097184,
                "average": True,  # 'True' in the dictionary is a string, it should be a boolean
                "fit_intercept": True,  # 'True' in the dictionary is a string, it should be a boolean
                "learning_rate": "optimal",
                "loss": "log_loss",  # changed from log_loss
                "penalty": "l1",
                "tol": 0.0001237547963958395,
            },
            space={},
        ),
        Component(
            name='model_porfolio_13',
            item=ExtraTreesClassifier,
            config={
                "bootstrap": False,  # 'False' in the dictionary is a string, it should be a boolean
                "criterion": "entropy",
                "max_depth": None,  # 'None' in the dictionary is a string, it should be None
                "max_features": 0.9589411099523568,
                "max_leaf_nodes": None,  # 'None' in the dictionary is a string, it should be None
                "min_impurity_decrease": 0.0,
                "min_samples_leaf": 1,
                "min_samples_split": 5,
                "min_weight_fraction_leaf": 0.0,
            },
            space={},
        ),
        Component(
            name='model_porfolio_14',
            item=ExtraTreesClassifier,
            config={
                "bootstrap": True,  # 'True' in the dictionary is a string, it should be a boolean
                "criterion": "entropy",
                "max_depth": None,  # 'None' in the dictionary is a string, it should be None
                "max_features": 0.2786066993244293,
                "max_leaf_nodes": None,  # 'None' in the dictionary is a string, it should be None
                "min_impurity_decrease": 0.0,
                "min_samples_leaf": 16,
                "min_samples_split": 16,
                "min_weight_fraction_leaf": 0.0,
            },
            space={},
        ),
        Component(
            name='model_porfolio_15',
            item=SGDClassifier,
            config={
                "alpha": 0.0003013049465842118,
                "average": False,  # 'False' in the dictionary is a string, it should be a boolean
                "fit_intercept": True,  # 'True' in the dictionary is a string, it should be a boolean
                "learning_rate": "optimal",
                "loss": "log_loss",  # changed from log_loss
                "penalty": "l2",
                "tol": 1.4130607731928e-05,
            },
            space={},
        ),
        Component(
            name='model_porfolio_16',
            item=GradientBoostingClassifier,
            config={
                "learning_rate": 0.19661821109657385,
                "max_depth": None,  # 'None' in the dictionary is a string, it should be None
                "max_leaf_nodes": 103,
                "min_samples_leaf": 1,
                "tol": 1e-07,
            },
            space={},
        ),
        Component(
            name='model_porfolio_17',
            item=RandomForestClassifier,
            config={
                "bootstrap": False,  # 'False' in the dictionary is a string, it should be False
                "criterion": "gini",
                "max_depth": None,  # 'None' in the dictionary is a string, it should be None
                "max_features": 0.8727662516250878,
                "min_samples_leaf": 11,
                "min_samples_split": 15,
                "min_weight_fraction_leaf": 0.0,
            },
            space={},
        ),
        Component(
            name='model_porfolio_18',
            item=GradientBoostingClassifier,
            config={
                "learning_rate": 0.08365360491111613,
                "min_samples_leaf": 41,
                "max_leaf_nodes": 10,
                "validation_fraction": 0.14053446064492747,
                "n_iter_no_change": 3,
                "tol": 1e-07,
            },
            space={},
        ),
        Component(
            name='model_porfolio_20',
            item=GradientBoostingClassifier,
            config={
                "learning_rate": 0.6047210939446936,
                "min_samples_leaf": 4,
                "max_depth": None,  # 'None' in the dictionary is a string, it should be None
                "max_leaf_nodes": 297,
                "validation_fraction": 0.35540373091502847,
                "n_iter_no_change": 7,
                "tol": 1e-07,
            },
            space={},
        ),
        Component(
            name='model_porfolio_21',
            item=GradientBoostingClassifier,
            config={
                "learning_rate": 0.01595430363700077,
                "min_samples_leaf": 161,
                "max_depth": None,  # 'None' in the dictionary is a string, it should be None
                "max_leaf_nodes": 4,
                "n_iter_no_change": 12,
                "tol": 1e-07,
            },
            space={},
        ),
        Component(
            name='model_porfolio_22',
            item=GradientBoostingClassifier,
            config={
                "learning_rate": 0.19121190092267595,
                "min_samples_leaf": 5,
                "max_depth": None,  # 'None' in the dictionary is a string, it should be None
                "max_leaf_nodes": 6,
                "tol": 1e-07,
            },
            space={},
        ),
        Component(
            name='model_porfolio_24',
            item=RandomForestClassifier,
            config={
                "bootstrap": True,  # 'True' in the dictionary is a string, it should be True
                "criterion": "entropy",
                "max_depth": None,  # 'None' in the dictionary is a string, it should be None
                "max_features": 0.8786824478611839,
                "min_samples_leaf": 5,
                "min_samples_split": 9,
                "min_weight_fraction_leaf": 0.0,
            },
            space={},
        ),
        Component(
            name='model_porfolio_25',
            item=ExtraTreesClassifier,
            config={
                "bootstrap": False,  # 'False' in the dictionary is a string, it should be False
                "criterion": "entropy",
                "max_depth": None,  # 'None' in the dictionary is a string, it should be None
                "max_features": 0.8547968025148605,
                "min_samples_leaf": 1,
                "min_samples_split": 14,
                "min_weight_fraction_leaf": 0.0,
            },
            space={},
        ),
        Component(
            name='model_porfolio_26',
            item=SGDClassifier,
            config={
                "alpha": 0.0003503796227984789,
                "average": False,  # 'False' in the dictionary is a string, it should be False
                "fit_intercept": True,  # 'True' in the dictionary is a string, it should be True
                "learning_rate": "optimal",
                "loss": "log_loss",  # changed from squared_hinge'
                "penalty": "l2",
                "tol": 0.030501813434465796,
            },
            space={},
        ),
        Component(
            name='model_porfolio_28',
            item=MLPClassifier,
            config={
                "activation": "relu",
                "alpha": 4.77181795127223e-05,
                "batch_size": "auto",
                "beta_1": 0.9,
                "beta_2": 0.999,
                "early_stopping": True,  # 'valid' in the dictionary is a string, it should be True
                "epsilon": 1e-08,
                "hidden_layer_sizes": (129,),
                # Assuming 'hidden_layer_depth' means number of layers and 'num_nodes_per_layer' means nodes per layer
                "learning_rate_init": 0.010457046844512303,
                "n_iter_no_change": 32,
                "shuffle": True,  # 'True' in the dictionary is a string, it should be True
                "solver": "adam",
                "tol": 0.0001,
                "validation_fraction": 0.1,
            },
            space={},
        ),
        Component(
            name='model_porfolio_29',
            item=MLPClassifier,
            config={
                "activation": "tanh",
                "alpha": 0.039008613885296826,
                "batch_size": "auto",
                "beta_1": 0.9,
                "beta_2": 0.999,
                "early_stopping": False,
                # 'train' in the dictionary is a string, it should be False as early stopping is not desired during training
                "epsilon": 1e-08,
                "hidden_layer_sizes": (140, 140, 140),
                # Assuming 'hidden_layer_depth' means number of layers and 'num_nodes_per_layer' means nodes per layer
                "learning_rate_init": 0.0023110285365222803,
                "n_iter_no_change": 32,
                "shuffle": True,  # 'True' in the dictionary is a string, it should be True
                "solver": "adam",
                "tol": 0.0001,
            },
            space={},
        ),
        Component(
            name='model_porfolio_30',
            item=ExtraTreesClassifier,
            config={
                "bootstrap": False,  # 'False' in the dictionary is a string, it should be False
                "criterion": "entropy",
                "max_depth": None,  # 'None' in the dictionary is a string, it should be None
                "max_features": 0.35771994991556005,
                "max_leaf_nodes": None,  # 'None' in the dictionary is a string, it should be None
                "min_impurity_decrease": 0.0,
                "min_samples_leaf": 2,
                "min_samples_split": 15,
                "min_weight_fraction_leaf": 0.0,
            },
            space={},
        ),
        Component(
            name='model_porfolio_31',
            item=GradientBoostingClassifier,
            config={
                "learning_rate": 0.06202602561543075,
                "min_samples_leaf": 16,
                "max_depth": None,  # 'None' in the dictionary is a string, it should be None
                "max_leaf_nodes": 3,
                "tol": 1e-07,
            },
            space={},
        ),
        Component(
            name='model_porfolio_32',
            item=GradientBoostingClassifier,
            config={
                "learning_rate": 0.02352567307613563,
                "min_samples_leaf": 2,
                "max_depth": None,  # 'None' in the dictionary is a string, it should be None
                "max_leaf_nodes": 654,
                "tol": 1e-07,
                "n_iter_no_change": 19,
                "validation_fraction": 0.29656195578950684,
            },
            space={},
        ),

        name="estimator"
    )
    )

    # # This pipeline generates 3 out 10 folds
    # pipeline = (
    #     Sequential(name="Pipeline")
    #     >> Split(
    #         {
    #             "categorical": [
    #                 SimpleImputer(strategy="constant", fill_value="missing"),
    #                 OneHotEncoder(drop="first"),
    #             ],
    #             "numerical": Component(
    #                 SimpleImputer,
    #                 space={"strategy": ["mean", "median"]},
    #             ),
    #         },
    #         name="feature_preprocessing",
    #     )
    #     >> Choice(
    #     Component(
    #         SVC,
    #         config={"probability": True, "random_state": 0},
    #         space={"C": (0.1, 10.0)},
    #     ),
    #     Component(
    #         RandomForestClassifier,
    #         config={"random_state": 0},
    #         space={"n_estimators": (10, 100), "criterion": ["gini", "log_loss"]},
    #     ),
    #     Component(
    #         MLPClassifier,
    #         config={"random_state": 0},
    #         space={
    #             "activation": ["identity", "logistic", "relu"],
    #             "alpha": (0.0001, 0.1),
    #             "learning_rate": ["constant", "invscaling", "adaptive"],
    #         },
    #     ),
    #     name="estimator"
    #     )
    # )


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


    optimizer = SMACOptimizer.create(
        space=pipeline,  #  (1)!
        metrics=Metric("accuracy", minimize=False, bounds=(0.0, 1.0)),
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
    best_trace = trace[0]
    best_bucket = best_trace.bucket
    new_path = posixpath.join(best_bucket.path, best_trace.name, 'model.pkl')
    print(new_path)
    best_model = joblib.load(str(new_path))

    return best_model, history_df

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
    )

