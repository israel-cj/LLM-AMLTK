"Classification: "
from amltk.pipeline import Component
from sklearn.ensemble import ExtraTreesClassifier

component = Component(
    ExtraTreesClassifier,
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
    "random_state": request("seed")
    },
    space={
        
    },
)

from amltk.pipeline import Component
from sklearn.ensemble import GradientBoostingClassifier

component = Component(
    GradientBoostingClassifier,
    config={
        "loss": "auto",
        "learning_rate": 0.19595673731599184,
        "min_samples_leaf": 2,
        "max_depth": None,  # 'None' in the dictionary is a string, it should be None
        "max_leaf_nodes": 10,
    },
    space={},
)

from amltk.pipeline import Component
from sklearn.linear_model import SGDClassifier

component = Component(
    SGDClassifier,
    config={
        "alpha": 8.445149920482102e-05,
        "average": True,  # 'True' in the dictionary is a string, it should be a boolean
        "fit_intercept": True,  # 'True' in the dictionary is a string, it should be a boolean
        "learning_rate": "optimal",
        "loss": "hinge",
        "penalty": "l2",
        "tol": 0.017561035108251574,
    },
    space={},
)

from amltk.pipeline import Component
from sklearn.ensemble import GradientBoostingClassifier

component = Component(
    GradientBoostingClassifier,
    config={
        "loss": "auto",
        "learning_rate": 0.19298903469131437,
        "min_samples_leaf": 20,
        "max_depth": None,  # 'None' in the dictionary is a string, it should be None
        "max_leaf_nodes": 10,
        "tol": 1e-07,
        "n_iter_no_change": 6,
        "validation_fraction": 0.11549557161401015,
    },
    space={},
)

from amltk.pipeline import Component
from sklearn.ensemble import GradientBoostingClassifier

component = Component(
    GradientBoostingClassifier,
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
)


from amltk.pipeline import Component
from sklearn.neural_network import MLPClassifier

component = Component(
    MLPClassifier,
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
)


from amltk.pipeline import Component
from sklearn.ensemble import GradientBoostingClassifier

component = Component(
    GradientBoostingClassifier,
    config={
        "loss": "auto",
        "learning_rate": 0.013391089474609338,
        "min_samples_leaf": 138,
        "max_depth": None,  # 'None' in the dictionary is a string, it should be None
        "max_leaf_nodes": 3,
        "tol": 1e-07,
    },
    space={},
)

from amltk.pipeline import Component
from sklearn.linear_model import SGDClassifier

component = Component(
    SGDClassifier,
    config={
        "alpha": 3.7007715928095603e-06,
        "average": False,  # 'False' in the dictionary is a string, it should be a boolean
        "fit_intercept": True,  # 'True' in the dictionary is a string, it should be a boolean
        "learning_rate": "constant",
        "loss": "perceptron",
        "penalty": "elasticnet",
        "tol": 0.04024454652952893,
        "eta0": 2.0514805804594057e-06,
        "l1_ratio": 7.163773086996508e-09,
    },
    space={},
)

from amltk.pipeline import Component
from sklearn.ensemble import GradientBoostingClassifier

component = Component(
    GradientBoostingClassifier,
    config={
        "loss": "auto",
        "learning_rate": 0.7615407561275519,
        "min_samples_leaf": 136,
        "max_depth": None,  # 'None' in the dictionary is a string, it should be None
        "max_leaf_nodes": 34,
        "tol": 1e-07,
        "n_iter_no_change": 11,
    },
    space={},
)


from amltk.pipeline import Component
from sklearn.linear_model import SGDClassifier

component = Component(
    SGDClassifier,
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
)

from amltk.pipeline import Component
from sklearn.ensemble import GradientBoostingClassifier

component = Component(
    GradientBoostingClassifier,
    config={
        "loss": "auto",
        "learning_rate": 0.7962804610609661,
        "min_samples_leaf": 21,
        "max_depth": None,  # 'None' in the dictionary is a string, it should be None
        "max_leaf_nodes": 183,
        "tol": 1e-07,
        "n_iter_no_change": 3,
    },
    space={},
)

from amltk.pipeline import Component
from sklearn.linear_model import SGDClassifier

component = Component(
    SGDClassifier,
    config={
        "alpha": 0.03522482415097184,
        "average": True,  # 'True' in the dictionary is a string, it should be a boolean
        "fit_intercept": True,  # 'True' in the dictionary is a string, it should be a boolean
        "learning_rate": "optimal",
        "loss": "hinge",
        "penalty": "l1",
        "tol": 0.0001237547963958395,
    },
    space={},
)

from amltk.pipeline import Component
from sklearn.ensemble import ExtraTreesClassifier

component = Component(
    ExtraTreesClassifier,
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
)

from amltk.pipeline import Component
from sklearn.ensemble import ExtraTreesClassifier

component = Component(
    ExtraTreesClassifier,
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
)

from amltk.pipeline import Component
from sklearn.linear_model import SGDClassifier

component = Component(
    SGDClassifier,
    config={
        "alpha": 0.0003013049465842118,
        "average": False,  # 'False' in the dictionary is a string, it should be a boolean
        "fit_intercept": True,  # 'True' in the dictionary is a string, it should be a boolean
        "learning_rate": "optimal",
        "loss": "hinge",
        "penalty": "l2",
        "tol": 1.4130607731928e-05,
    },
    space={},
)

from amltk.pipeline import Component
from sklearn.ensemble import GradientBoostingClassifier

component = Component(
    GradientBoostingClassifier,
    config={
        "early_stop": "off",
        "l2_regularization": 0.06413435439100516,
        "learning_rate": 0.19661821109657385,
        "loss": "auto",
        "max_bins": 255,
        "max_depth": None,  # 'None' in the dictionary is a string, it should be None
        "max_leaf_nodes": 103,
        "min_samples_leaf": 1,
        "scoring": "loss",
        "tol": 1e-07,
    },
    space={},
)

from amltk.pipeline import Component
from sklearn.ensemble import GradientBoostingClassifier

component = Component(
    GradientBoostingClassifier,
    config={
        "learning_rate": 0.08365360491111613,
        "n_estimators": 100,  # default value
        "subsample": 1.0,  # default value
        "criterion": "friedman_mse",  # default value
        "min_samples_split": 2,  # default value
        "min_samples_leaf": 41,
        "min_weight_fraction_leaf": 0.0,  # default value
        "max_depth": None,  # 'None' in the dictionary is a string, it should be None
        "min_impurity_decrease": 0.0,  # default value
        "min_impurity_split": None,  # default value
        "init": None,  # default value
        "random_state": None,  # default value
        "max_features": None,  # default value
        "verbose": 0,  # default value
        "max_leaf_nodes": 10,
        "warm_start": False,  # default value
        "validation_fraction": 0.14053446064492747,
        "n_iter_no_change": 3,
        "tol": 1e-07,
        "ccp_alpha": 0.0,  # default value
    },
    space={},
)

from amltk.pipeline import Component
from sklearn.linear_model import PassiveAggressiveClassifier

component = Component(
    PassiveAggressiveClassifier,
    config={
        "C": 0.023921642444558362,
        "fit_intercept": True,  # 'True' in the dictionary is a string, it should be a boolean
        "loss": "hinge",
        "tol": 6.434802101116561e-05,
        "average": False,  # 'False' in the dictionary is a string, it should be a boolean
    },
    space={},
)

from amltk.pipeline import Component
from sklearn.ensemble import GradientBoostingClassifier

component = Component(
    GradientBoostingClassifier,
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
)


from amltk.pipeline import Component
from sklearn.ensemble import GradientBoostingClassifier

component = Component(
    GradientBoostingClassifier,
    config={
        "learning_rate": 0.01595430363700077,
        "min_samples_leaf": 161,
        "max_depth": None,  # 'None' in the dictionary is a string, it should be None
        "max_leaf_nodes": 4,
        "n_iter_no_change": 12,
        "tol": 1e-07,
    },
    space={},
)

from amltk.pipeline import Component
from sklearn.ensemble import GradientBoostingClassifier

component = Component(
    GradientBoostingClassifier,
    config={
        "learning_rate": 0.19121190092267595,
        "min_samples_leaf": 5,
        "max_depth": None,  # 'None' in the dictionary is a string, it should be None
        "max_leaf_nodes": 6,
        "tol": 1e-07,
    },
    space={},
)


from amltk.pipeline import Component
from sklearn.ensemble import ExtraTreesClassifier

component = Component(
    ExtraTreesClassifier,
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
)


from amltk.pipeline import Component
from sklearn.linear_model import SGDClassifier

component = Component(
    SGDClassifier,
    config={
        "alpha": 0.0003503796227984789,
        "average": False,  # 'False' in the dictionary is a string, it should be False
        "fit_intercept": True,  # 'True' in the dictionary is a string, it should be True
        "learning_rate": "optimal",
        "loss": "squared_hinge",
        "penalty": "l2",
        "tol": 0.030501813434465796,
    },
    space={},
)

from amltk.pipeline import Component
from sklearn.linear_model import PassiveAggressiveClassifier

component = Component(
    PassiveAggressiveClassifier,
    config={
        "C": 0.0015705840539998888,
        "average": False,  # 'False' in the dictionary is a string, it should be False
        "fit_intercept": True,  # 'True' in the dictionary is a string, it should be True
        "loss": "squared_hinge",
        "tol": 0.009455868366528827,
    },
    space={},
)


from amltk.pipeline import Component
from sklearn.neural_network import MLPClassifier

component = Component(
    MLPClassifier,
    config={
        "activation": "relu",
        "alpha": 4.77181795127223e-05,
        "batch_size": "auto",
        "beta_1": 0.9,
        "beta_2": 0.999,
        "early_stopping": True,  # 'valid' in the dictionary is a string, it should be True
        "epsilon": 1e-08,
        "hidden_layer_sizes": (129,),  # Assuming 'hidden_layer_depth' means number of layers and 'num_nodes_per_layer' means nodes per layer
        "learning_rate_init": 0.010457046844512303,
        "n_iter_no_change": 32,
        "shuffle": True,  # 'True' in the dictionary is a string, it should be True
        "solver": "adam",
        "tol": 0.0001,
        "validation_fraction": 0.1,
    },
    space={},
)


from amltk.pipeline import Component
from sklearn.neural_network import MLPClassifier

component = Component(
    MLPClassifier,
    config={
        "activation": "tanh",
        "alpha": 0.039008613885296826,
        "batch_size": "auto",
        "beta_1": 0.9,
        "beta_2": 0.999,
        "early_stopping": False,  # 'train' in the dictionary is a string, it should be False as early stopping is not desired during training
        "epsilon": 1e-08,
        "hidden_layer_sizes": (140, 140, 140),  # Assuming 'hidden_layer_depth' means number of layers and 'num_nodes_per_layer' means nodes per layer
        "learning_rate_init": 0.0023110285365222803,
        "n_iter_no_change": 32,
        "shuffle": True,  # 'True' in the dictionary is a string, it should be True
        "solver": "adam",
        "tol": 0.0001,
    },
    space={},
)


from amltk.pipeline import Component
from sklearn.ensemble import ExtraTreesClassifier

component = Component(
    ExtraTreesClassifier,
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
)

from amltk.pipeline import Component
from sklearn.ensemble import GradientBoostingClassifier

component = Component(
    GradientBoostingClassifier,
    config={
        "learning_rate": 0.06202602561543075,
        "min_samples_leaf": 16,
        "max_depth": None,  # 'None' in the dictionary is a string, it should be None
        "max_leaf_nodes": 3,
        "tol": 1e-07,
    },
    space={},
)

from amltk.pipeline import Component
from sklearn.ensemble import GradientBoostingClassifier

component = Component(
    GradientBoostingClassifier,
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
)

from amltk.pipeline import Component
from sklearn.ensemble import GradientBoostingClassifier

component = Component(
    GradientBoostingClassifier,
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
)

from amltk.pipeline import Component
from sklearn.ensemble import RandomForestClassifier

component = Component(
    RandomForestClassifier,
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
)