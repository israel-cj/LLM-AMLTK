from .amltk_classification import run_amltk
import uuid

class AMLTK_v1():
    """
    Parameters:
    """
    def __init__(
            self,
            N_WORKERS = 32,
            partition="thin",
            cores=8,
            memory="32 GB",
            walltime="01:00:00",
            task = "classification"
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

        if self.task == "classification":
            self.model, self.report = run_amltk(
                N_WORKERS=32,
                partition="thin",
                cores=8,
                memory="32 GB",
                walltime="01:00:00",
                X=X,
                y=y,
            )

        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)  # type: ignore

    def predict_log_proba(self, X):
        return self.model.predict_log_proba(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

