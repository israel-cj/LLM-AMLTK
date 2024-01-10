from .amltk_classification import run_amltk
from .amltk_regression import run_amltk_regressor
from .llm_optimization import improve_models
import uuid

class AMLTK_llm():
    """
    Parameters:
    """
    def __init__(
            self,
            N_WORKERS = 32,
            partition="genoa",
            cores=8,
            memory="32 GB",
            walltime=60,
            task = "classification",
            enhance = True,
            search_space = None
    ) -> None:
        self.N_WORKERS = N_WORKERS
        self.partition = partition
        self.cores = cores
        self.memory = memory
        self.walltime = walltime
        self.task = task
        self.uid = str(uuid.uuid4())
        self.report = None
        self.model = None
        self.real_history = None
        self.real_metric = None
        self.pipeline = None
        self.enhance = enhance
        self.search_space = search_space
    def fit(
            self, X, y, disable_caafe=False
    ):
        """
        Fit the model to the training data.

        Parameters:
        -----------
        X : pd.DataFrame
            The training data features.
        y : pd.Serie
            The training data target values.

        """
        print('uid', self.uid)
        self.fit_inner(X, y, int(self.walltime/2))
        print('A model has been optimized with SMACOptimizer')
        self.model.fit(X, y)
        # optimize the search method with LLM
        if self.enhance:
            print('Looking for a better search space with LLM')
            new_search_space = improve_models(
                history = self.real_history,
                task = self.task,
                real_metric=self.real_metric,
                llm_model="gpt-3.5-turbo",
                search_space=self.search_space,
                pipeline_space = self.pipeline,
            )
            if len(new_search_space)>0:
                print('Search space created')
                self.fit_inner(X, y, int(self.walltime/2), search_space=new_search_space)
                print('A new model was found with SMACOptimizer')
                self.model.fit(X, y)
                print('The model generated with LLM was trained')

    def fit_inner(self, X, y, walltime, search_space = None):
        if self.task == "classification":
            self.model, self.report, self.real_history, self.real_metric, self.search_space, self.pipeline = run_amltk(
                N_WORKERS=self.N_WORKERS,
                partition=self.partition,
                cores=self.cores,
                memory=self.memory,
                walltime=walltime,
                X=X,
                y=y,
                search_space=search_space,
            )

        if self.task == "regression":
            self.model, self.report, self.real_history, self.real_metric, self.search_space, self.pipeline = run_amltk_regressor(
                N_WORKERS=self.N_WORKERS,
                partition=self.partition,
                cores=self.cores,
                memory=self.memory,
                walltime=walltime,
                X=X,
                y=y,
                search_space=search_space,
            )

    def predict(self, X):
        return self.model.predict(X)  # type: ignore

    def predict_log_proba(self, X):
        return self.model.predict_log_proba(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

