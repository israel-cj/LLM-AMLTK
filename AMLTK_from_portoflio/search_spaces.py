# Sklearn classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
# Sklearn regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

# AMLTK
from amltk.pipeline import Component


classification_search_space = {
    'model_portfolio_1': Component(
            name = 'model_portfolio_1',
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
      'model_portfolio_2': Component(
            name = 'model_portfolio_2',
            item = GradientBoostingClassifier,
            config={
                "learning_rate": 0.19595673731599184,
                "min_samples_leaf": 2,
                "max_depth": None,  # 'None' in the dictionary is a string, it should be None
                "max_leaf_nodes": 10,
            },
            space={},
          ),
      'model_porftolio_3': Component(
            name = 'model_portfolio_3',
            item = SGDClassifier,
            config={
                "alpha": 8.445149920482102e-05,
                "average": True,  # 'True' in the dictionary is a string, it should be a boolean
                "fit_intercept": True,  # 'True' in the dictionary is a string, it should be a boolean
                "learning_rate": "optimal",
                "loss": "log_loss", # change from  hinge
                "penalty": "l2",
                "tol": 0.017561035108251574,
            },
            space={},
        ),
      'model_portfolio_4': Component(
              name = 'model_portfolio_4',
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
      'model_portfolio_5': Component(
            name = 'model_portfolio_5',
            item = GradientBoostingClassifier,
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
      'model_portfolio_6': Component(
            name = 'model_portfolio_6',
            item = MLPClassifier,
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
      'model_portfolio_7': Component(
            name = 'model_portfolio_7',
            item = GradientBoostingClassifier,
            config={
                "learning_rate": 0.013391089474609338,
                "min_samples_leaf": 138,
                "max_depth": None,  # 'None' in the dictionary is a string, it should be None
                "max_leaf_nodes": 3,
                "tol": 1e-07,
            },
            space={},
        ),
      'model_portfolio_8': Component(
            name = 'model_portfolio_8',
            item = SGDClassifier,
            config={
                "alpha": 3.7007715928095603e-06,
                "average": False,  # 'False' in the dictionary is a string, it should be a boolean
                "fit_intercept": True,  # 'True' in the dictionary is a string, it should be a boolean
                "learning_rate": "constant",
                "loss": "log_loss", #changed from perceptron
                "penalty": "elasticnet",
                "tol": 0.04024454652952893,
                "eta0": 2.0514805804594057e-06,
                "l1_ratio": 7.163773086996508e-09,
            },
            space={},
        ),
      'model_portfolio_9': Component(
            name = 'model_portfolio_9',
            item = GradientBoostingClassifier,
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
      'model_portfolio_10': Component(
            name = 'model_portfolio_10',
            item = SGDClassifier,
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
      'model_portfolio_11': Component(
            name = 'model_portfolio_11',
            item = GradientBoostingClassifier,
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
      'model_portfolio_12': Component(
            name = 'model_portfolio_12',
            item = SGDClassifier,
            config={
                "alpha": 0.03522482415097184,
                "average": True,  # 'True' in the dictionary is a string, it should be a boolean
                "fit_intercept": True,  # 'True' in the dictionary is a string, it should be a boolean
                "learning_rate": "optimal",
                "loss": "log_loss", #changed from log_loss
                "penalty": "l1",
                "tol": 0.0001237547963958395,
            },
            space={},
        ),
       'model_portfolio_13': Component(
            name = 'model_portfolio_13',
            item = ExtraTreesClassifier,
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
      'model_portfolio_14': Component(
            name = 'model_portfolio_14',
            item = ExtraTreesClassifier,
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
      'model_portfolio_15': Component(
            name = 'model_portfolio_15',
            item = SGDClassifier,
            config={
                "alpha": 0.0003013049465842118,
                "average": False,  # 'False' in the dictionary is a string, it should be a boolean
                "fit_intercept": True,  # 'True' in the dictionary is a string, it should be a boolean
                "learning_rate": "optimal",
                "loss": "log_loss", #changed from log_loss
                "penalty": "l2",
                "tol": 1.4130607731928e-05,
            },
            space={},
        ),
      'model_portfolio_16': Component(
            name = 'model_portfolio_16',
            item = GradientBoostingClassifier,
            config={
                "learning_rate": 0.19661821109657385,
                "max_depth": None,  # 'None' in the dictionary is a string, it should be None
                "max_leaf_nodes": 103,
                "min_samples_leaf": 1,
                "tol": 1e-07,
            },
            space={},
        ),
      'model_portfolio_17': Component(
            name = 'model_portfolio_17',
            item = RandomForestClassifier,
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
      'model_portfolio_18': Component(
            name = 'model_portfolio_18',
            item = GradientBoostingClassifier,
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
      'model_portfolio_20': Component(
            name = 'model_portfolio_20',
            item = GradientBoostingClassifier,
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
      'model_portfolio_21': Component(
            name = 'model_portfolio_21',
            item = GradientBoostingClassifier,
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
      'model_portfolio_22': Component(
            name = 'model_portfolio_22',
            item = GradientBoostingClassifier,
            config={
                "learning_rate": 0.19121190092267595,
                "min_samples_leaf": 5,
                "max_depth": None,  # 'None' in the dictionary is a string, it should be None
                "max_leaf_nodes": 6,
                "tol": 1e-07,
            },
            space={},
        ),
      'model_portfolio_24': Component(
            name = 'model_portfolio_24',
            item = RandomForestClassifier,
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
      'model_portfolio_25': Component(
            name = 'model_portfolio_25',
            item = ExtraTreesClassifier,
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
       'model_portfolio_26': Component(
            name = 'model_portfolio_26',
            item = SGDClassifier,
            config={
                "alpha": 0.0003503796227984789,
                "average": False,  # 'False' in the dictionary is a string, it should be False
                "fit_intercept": True,  # 'True' in the dictionary is a string, it should be True
                "learning_rate": "optimal",
                "loss": "log_loss", #changed from squared_hinge'
                "penalty": "l2",
                "tol": 0.030501813434465796,
            },
            space={},
        ),
       'model_portfolio_28': Component(
            name = 'model_portfolio_28',
            item = MLPClassifier,
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
        ),
      'model_portfolio_29': Component(
            name = 'model_portfolio_29',
            item = MLPClassifier,
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
        ),
       'model_portfolio_30': Component(
            name = 'model_portfolio_30',
            item = ExtraTreesClassifier,
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
      'model_portfolio_31': Component(
            name = 'model_portfolio_31',
            item = GradientBoostingClassifier,
            config={
                "learning_rate": 0.06202602561543075,
                "min_samples_leaf": 16,
                "max_depth": None,  # 'None' in the dictionary is a string, it should be None
                "max_leaf_nodes": 3,
                "tol": 1e-07,
            },
            space={},
        ),
      'model_portfolio_32': Component(
            name = 'model_portfolio_32',
            item = GradientBoostingClassifier,
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
}

regression_search_space = {
    'model_regressor_porfolio_1': Component(
            name = 'model_regressor_porfolio_1',
            item = RandomForestRegressor,
            config={
                "max_depth": 15,
                "criterion": "squared_error",
            },
            space={},
        ),
    'model_regressor_porfolio_2': Component(
            name = 'model_regressor_porfolio_2',
            item = ExtraTreesRegressor,
            config={
                "min_samples_leaf": 15,
                "criterion": "squared_error",
            },
            space={},
        ),
    'model_regressor_porfolio_3': Component(
            name = 'model_regressor_porfolio_3',
            item = RandomForestRegressor,
            config={
                "min_samples_leaf": 5,
                "max_leaf_nodes": 50000,
                "max_features": 0.5,
            },
            space={},
        ),
    'model_regressor_porfolio_4': Component(
            name = 'model_regressor_porfolio_4',
            item = ExtraTreesRegressor,
            config={
                "min_samples_leaf": 1,
                "max_leaf_nodes": 15000,
                "max_features":0.5,
            },
            space={},
        ),
      # The next components come from https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/configs/zeroshot/zeroshot_portfolio_2023.py
      # We assume that the criterion should be fixed and then we can play with the rest of the hyperparameters of the XT (extra trees) and RF (random forest)
    'model_regressor_porfolio_5': Component(
            name = 'model_regressor_porfolio_5',
            item = RandomForestRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 0.75,
                "max_leaf_nodes": 37308,
                "min_samples_leaf": 1,
            },
            space={},
        ),
    'model_regressor_porfolio_6': Component(
            name = 'model_regressor_porfolio_6',
            item = RandomForestRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 0.75,
                "max_leaf_nodes": 28310,
                "min_samples_leaf": 2,
            },
            space={},
        ),
    'model_regressor_porfolio_7': Component(
            name = 'model_regressor_porfolio_7',
            item = RandomForestRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 1.0,
                "max_leaf_nodes": 38572,
                "min_samples_leaf": 5,
            },
            space={},
        ),
    'model_regressor_porfolio_8': Component(
            name = 'model_regressor_porfolio_8',
            item = RandomForestRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 0.75,
                "max_leaf_nodes": 18242,
                "min_samples_leaf": 40,
            },
            space={},
        ),
    'model_regressor_porfolio_9': Component(
            name = 'model_regressor_porfolio_9',
            item = RandomForestRegressor,
            config={
                "criterion": "squared_error",
                "max_features": "log2",
                "max_leaf_nodes": 42644,
                "min_samples_leaf": 1,
            },
            space={},
        ),
    'model_regressor_porfolio_10': Component(
            name = 'model_regressor_porfolio_10',
            item = RandomForestRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 0.75,
                "max_leaf_nodes": 36230,
                "min_samples_leaf": 3,
            },
            space={},
        ),
    'model_regressor_porfolio_11': Component(
            name = 'model_regressor_porfolio_11',
            item = RandomForestRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 1.0,
                "max_leaf_nodes": 48136,
                "min_samples_leaf": 1,
            },
            space={},
        ),
    'model_regressor_porfolio_12': Component(
            name = 'model_regressor_porfolio_12',
            item = ExtraTreesRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 0.75,
                "max_leaf_nodes": 18392,
                "min_samples_leaf": 1,
            },
            space={},
        ),
    'model_regressor_porfolio_13': Component(
            name = 'model_regressor_porfolio_13',
            item = ExtraTreesRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 0.75,
                "max_leaf_nodes": 18392,
                "min_samples_leaf": 1,
            },
            space={},
        ),
    'model_regressor_porfolio_14': Component(
            name = 'model_regressor_porfolio_14',
            item = ExtraTreesRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 1.0,
                "max_leaf_nodes": 12845,
                "min_samples_leaf": 4,
            },
            space={},
        ),
    'model_regressor_porfolio_15': Component(
            name = 'model_regressor_porfolio_15',
            item = ExtraTreesRegressor,
            config={
                "criterion": "squared_error",
                "max_features": "sqrt",
                "max_leaf_nodes": 28532,
                "min_samples_leaf": 1,
            },
            space={},
        ),
    'model_regressor_porfolio_16': Component(
            name = 'model_regressor_porfolio_16',
            item = ExtraTreesRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 1.0,
                "max_leaf_nodes": 19935,
                "min_samples_leaf": 20,
            },
            space={},
        ),
    'model_regressor_porfolio_17': Component(
            name = 'model_regressor_porfolio_17',
            item = ExtraTreesRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 0.75,
                "max_leaf_nodes": 29813,
                "min_samples_leaf": 4,
            },
            space={},
        ),
    'model_regressor_porfolio_18': Component(
            name = 'model_regressor_porfolio_18',
            item = ExtraTreesRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 1.0,
                "max_leaf_nodes": 40459,
                "min_samples_leaf": 1,
            },
            space={},
        ),
    'model_regressor_porfolio_19': Component(
            name = 'model_regressor_porfolio_19',
            item = ExtraTreesRegressor,
            config={
                "criterion": "squared_error",
                "max_features": "sqrt",
                "max_leaf_nodes": 29702,
                "min_samples_leaf": 2,
            },
            space={},
        ),
}