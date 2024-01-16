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


LLM_generated_classification_search_space = {
      'LLM_model_portfolio_1': Component(
          name = 'LLM_model_portfolio_1',
          item = ExtraTreesClassifier,
          config = {
              "random_state": 0,
          },
          space={
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "max_depth": list(range(1, 21)),  # depths from 1 to 20 and None
              "max_features": [i/10.0 for i in range(1, 11)],  # fractions from 0.1 to 1.0
              "min_samples_leaf": list(range(1, 21)),  # from 1 to 20
              "min_samples_split": list(range(2, 21)),  # from 2 to 20
              "min_weight_fraction_leaf": [i/10.0 for i in range(0, 11)],  # fractions from 0.0 to 1.0
          },
      ),
      'LLM_model_portfolio_2': Component(
          name = 'LLM_model_portfolio_2',
          item = GradientBoostingClassifier,
          config={
                "learning_rate": 0.19595673731599184,
                "min_samples_leaf": 2,
                "max_depth": None,
                "max_leaf_nodes": 10,
            },
            space={
                "n_estimators": list(range(50, 200, 10)),  # number of boosting stages to perform
                "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # fraction of samples to be used for fitting the individual base learners
            },
        ),
      'LLM_model_portfolio_3': Component(
          name = 'LLM_model_portfolio_3',
          item = SGDClassifier,
          config={
              "learning_rate": "optimal",
              "loss": "log_loss", # change from  hinge
              "penalty": "l2",
          },
          space={
              "alpha": [10**-i for i in range(1, 7)],  # values from 0.1 to 1e-6
              "average": [True, False],
              "fit_intercept": [True, False],
              "tol": [10**-i for i in range(1, 5)],  # values from 0.1 to 1e-4
          },
      ),
      'LLM_model_portfolio_4': Component(
          name = 'LLM_model_portfolio_4',
          item = GradientBoostingClassifier,
          config={
              "learning_rate": 0.19298903469131437,
              "min_samples_leaf": 20,
              "max_depth": None,
              "max_leaf_nodes": 10,
              "tol": 1e-07,
              "n_iter_no_change": 6,
              "validation_fraction": 0.11549557161401015,
          },
          space={
              "n_estimators": list(range(50, 200, 10)),  # number of boosting stages to perform
              "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # fraction of samples to be used for fitting the individual base learners
          },
      ),
      'LLM_model_portfolio_5': Component(
          name = 'LLM_model_portfolio_5',
          item = GradientBoostingClassifier,
          config={
              "learning_rate": 0.07043555880113304,
              "min_samples_leaf": 7,
              "max_depth": None,
              "max_leaf_nodes": 698,
              "tol": 1e-07,
              "n_iter_no_change": 1,
              "validation_fraction": 0.16202792486532844,
          },
          space={
              "n_estimators": list(range(50, 200, 10)),  # number of boosting stages to perform
              "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # fraction of samples to be used for fitting the individual base learners
          },
      ),
      'LLM_model_portfolio_6': Component(
          name = 'LLM_model_portfolio_6',
          item = MLPClassifier,
          config={
              "activation": "relu",
              "batch_size": "auto",
              "beta_1": 0.9,
              "beta_2": 0.999,
              "epsilon": 1e-08,
              "solver": "adam",
          },
          space={
              "alpha": [10**-i for i in range(1, 7)],  # values from 0.1 to 1e-6
              "early_stopping": [True, False],
              "hidden_layer_sizes": [(i, i) for i in range(10, 61, 10)],  # pairs of equal values from 10 to 60
              "learning_rate_init": [i/10.0 for i in range(1, 11)],  # fractions from 0.1 to 1.0
              "n_iter_no_change": list(range(10, 41, 5)),  # values from 10 to 40 with a step of 5
              "shuffle": [True, False],
              "tol": [10**-i for i in range(4, 5)],  # values from 1e-4
          },
      ),
      'LLM_model_portfolio_7': Component(
          name = 'LLM_model_portfolio_7',
          item = GradientBoostingClassifier,
          config={
              "learning_rate": 0.013391089474609338,
              "min_samples_leaf": 138,
              "max_depth": None,
              "max_leaf_nodes": 3,
              "tol": 1e-07,
          },
          space={
              "n_estimators": list(range(50, 200, 10)),  # number of boosting stages to perform
              "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # fraction of samples to be used for fitting the individual base learners
              "n_iter_no_change": list(range(1, 11)),  # number of iterations with no improvement to wait before stopping training
              "validation_fraction": [i/10.0 for i in range(1, 11)],  # the proportion of training data to set aside as validation set for early stopping
          },
      ),
      'LLM_model_portfolio_8': Component(
          name = 'LLM_model_portfolio_8',
          item = SGDClassifier,
          config={
              "average": False,
              "fit_intercept": True,
              "learning_rate": "constant",
              "loss": "log_loss",
              "penalty": "elasticnet",
          },
          space={
              "alpha": [10**-i for i in range(6, 7)],  # values from 1e-6
              "tol": [i/100.0 for i in range(1, 5)],  # fractions from 0.01 to 0.04
              "eta0": [10**-i for i in range(6, 7)],  # values from 1e-6
              "l1_ratio": [10**-i for i in range(8, 9)],  # values from 1e-8
          },
      ),
      'LLM_model_portfolio_9': Component(
          name = 'LLM_model_portfolio_9',
          item = GradientBoostingClassifier,
          config={
              "learning_rate": 0.7615407561275519,
              "min_samples_leaf": 136,
              "max_depth": None,
              "max_leaf_nodes": 34,
              "tol": 1e-07,
              "n_iter_no_change": 11,
          },
          space={
              "n_estimators": list(range(50, 200, 10)),  # number of boosting stages to perform
              "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # fraction of samples to be used for fitting the individual base learners
              "validation_fraction": [i/10.0 for i in range(1, 11)],  # the proportion of training data to set aside as validation set for early stopping
          },
      ),
      'LLM_model_portfolio_10': Component(
          name = 'LLM_model_portfolio_10',
          item = SGDClassifier,
          config={
              "average": False,
              "fit_intercept": True,
              "learning_rate": "constant",
              "loss": "modified_huber",
              "penalty": "elasticnet",
          },
          space={
              "alpha": [10**-i for i in range(5, 6)],  # values from 1e-5
              "tol": [10**-i for i in range(5, 6)],  # values from 1e-5
              "epsilon": [i/100.0 for i in range(1, 11)],  # fractions from 0.01 to 0.1
              "eta0": [10**-i for i in range(4, 5)],  # values from 1e-4
              "l1_ratio": [i/10.0 for i in range(1, 11)],  # fractions from 0.1 to 1.0
          },
      ),
      'LLM_model_portfolio_11': Component(
          name = 'LLM_model_portfolio_11',
          item = GradientBoostingClassifier,
          config={
              "learning_rate": 0.7962804610609661,
              "min_samples_leaf": 21,
              "max_depth": None,
              "max_leaf_nodes": 183,
              "tol": 1e-07,
              "n_iter_no_change": 3,
          },
          space={
              "n_estimators": list(range(50, 200, 10)),  # number of boosting stages to perform
              "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # fraction of samples to be used for fitting the individual base learners
              "validation_fraction": [i/10.0 for i in range(1, 11)],  # the proportion of training data to set aside as validation set for early stopping
          },
      ),
      'LLM_model_portfolio_12': Component(
            name = 'LLM_model_portfolio_12',
            item = SGDClassifier,
            config={
                "average": True,
                "fit_intercept": True,
                "learning_rate": "optimal",
                "loss": "log_loss",
                "penalty": "l1",
            },
            space={
                "alpha": [i/100.0 for i in range(1, 4)],  # fractions from 0.01 to 0.03
                "tol": [10**-i for i in range(4, 5)],  # values from 1e-4
            },
        ),
       'LLM_model_portfolio_13': Component(
          name = 'LLM_model_portfolio_13',
          item = ExtraTreesClassifier,
          config={
              "bootstrap": False,
              "criterion": "entropy",
          },
          space={
              "max_depth": list(range(1, 21)),  # depths from 1 to 20 and None
              "max_features": [i/10.0 for i in range(1, 11)],  # fractions from 0.1 to 1.0
              "max_leaf_nodes": list(range(2, 21)),  # from 2 to 20 and None
              "min_impurity_decrease": [0.0],
              "min_samples_leaf": list(range(1, 22)),  # from 1 to 21
              "min_samples_split": list(range(2, 6)),  # from 2 to 5
              "min_weight_fraction_leaf": [0.0],
          },
      ),
      'LLM_model_portfolio_14': Component(
          name = 'LLM_model_portfolio_14',
          item = ExtraTreesClassifier,
          config={
              "bootstrap": True,
              "criterion": "entropy",
          },
          space={
              "max_depth": list(range(1, 21)),  # depths from 1 to 20 and None
              "max_features": [i/10.0 for i in range(1, 11)],  # fractions from 0.1 to 1.0
              "max_leaf_nodes": list(range(2, 21)),  # from 2 to 20 and None
              "min_impurity_decrease": [0.0],
              "min_samples_leaf": list(range(1, 17)),  # from 1 to 16
              "min_samples_split": list(range(2, 17)),  # from 2 to 16
              "min_weight_fraction_leaf": [0.0],
          },
      ),
      'LLM_model_portfolio_15': Component(
          name = 'LLM_model_portfolio_15',
          item = SGDClassifier,
          config={
              "average": False,
              "fit_intercept": True,
              "learning_rate": "optimal",
              "loss": "log_loss",
              "penalty": "l2",
          },
          space={
              "alpha": [i/10000.0 for i in range(1, 4)],  # fractions from 0.0001 to 0.0003
              "tol": [10**-i for i in range(4, 6)],  # values from 1e-4 to 1e-5
          },
      ),
      'LLM_model_portfolio_16': Component(
          name = 'LLM_model_portfolio_16',
          item = GradientBoostingClassifier,
          config={
              "learning_rate": 0.19661821109657385,
              "max_leaf_nodes": 103,
              "min_samples_leaf": 1,
              "tol": 1e-07,
          },
          space={
              "max_depth": list(range(1, 21)),  # depths from 1 to 20 and None
          },
      ),
      'LLM_model_portfolio_17': Component(
          name = 'LLM_model_portfolio_17',
          item = RandomForestClassifier,
          config={
              "bootstrap": False,
              "criterion": "gini",
              "min_samples_leaf": 11,
              "min_samples_split": 15,
              "min_weight_fraction_leaf": 0.0,
          },
          space={
              "max_depth": list(range(1, 21)),  # depths from 1 to 20 and None
              "max_features": [i/10.0 for i in range(1, 10)],  # fractions from 0.1 to 0.9
          },
      ),
      'LLM_model_portfolio_18': Component(
          name = 'LLM_model_portfolio_18',
          item = GradientBoostingClassifier,
          config={
              "learning_rate": 0.08365360491111613,
              "min_samples_leaf": 41,
              "max_leaf_nodes": 10,
              "validation_fraction": 0.14053446064492747,
              "n_iter_no_change": 3,
              "tol": 1e-07,
          },
          space={
              "max_depth": list(range(1, 21)),  # depths from 1 to 20 and None
          },
      ),
      'LLM_model_portfolio_20': Component(
          name = 'LLM_model_portfolio_20',
          item = GradientBoostingClassifier,
          config={
              "learning_rate": 0.6047210939446936,
              "min_samples_leaf": 4,
              "max_leaf_nodes": 297,
              "validation_fraction": 0.35540373091502847,
              "n_iter_no_change": 7,
              "tol": 1e-07,
          },
          space={
              "max_depth": list(range(1, 21)),  # depths from 1 to 20 and None
          },
      ),
      'LLM_model_portfolio_21': Component(
          name = 'LLM_model_portfolio_21',
          item = GradientBoostingClassifier,
          config={
              "learning_rate": 0.01595430363700077,
              "min_samples_leaf": 161,
              "max_leaf_nodes": 4,
              "n_iter_no_change": 12,
              "tol": 1e-07,
          },
          space={
              "max_depth": list(range(1, 21)),  # depths from 1 to 20 and None
          },
      ),
      'LLM_model_portfolio_22': Component(
          name = 'LLM_model_portfolio_22',
          item = GradientBoostingClassifier,
          config={
              "learning_rate": 0.19121190092267595,
              "min_samples_leaf": 5,
              "max_leaf_nodes": 6,
              "tol": 1e-07,
          },
          space={
              "max_depth": list(range(1, 21)),  # depths from 1 to 20 and None
          },
      ),
      'LLM_model_portfolio_24': Component(
          name = 'LLM_model_portfolio_24',
          item = RandomForestClassifier,
          config={
              "bootstrap": True,
              "criterion": "entropy",
              "min_samples_leaf": 5,
              "min_samples_split": 9,
              "min_weight_fraction_leaf": 0.0,
          },
          space={
              "max_depth": list(range(1, 21)),  # depths from 1 to 20 and None
              "max_features": [i/10.0 for i in range(1, 10)],  # fractions from 0.1 to 0.9
          },
      ),
      'LLM_model_portfolio_25': Component(
            name = 'LLM_model_portfolio_25',
            item = ExtraTreesClassifier,
            config={
                "bootstrap": False,
                "criterion": "entropy",
                "min_samples_leaf": 1,
                "min_samples_split": 14,
                "min_weight_fraction_leaf": 0.0,
            },
            space={
                "max_depth": list(range(1, 21)),  # depths from 1 to 20 and None
                "max_features": [i/10.0 for i in range(1, 10)],  # fractions from 0.1 to 0.9
            },
        ),
       'LLM_model_portfolio_26': Component(
            name = 'LLM_model_portfolio_26',
            item = SGDClassifier,
            config={
                "average": False,
                "fit_intercept": True,
                "learning_rate": "optimal",
                "loss": "log_loss",
                "penalty": "l2",
            },
            space={
                "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],  # logarithmically spaced values
                "tol": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],  # logarithmically spaced values
            },
        ),
       'LLM_model_portfolio_28': Component(
          name = 'LLM_model_portfolio_28',
          item = MLPClassifier,
          config={
              "activation": "relu",
              "alpha": 4.77181795127223e-05,
              "batch_size": "auto",
              "beta_1": 0.9,
              "beta_2": 0.999,
              "early_stopping": True,
              "epsilon": 1e-08,
              "hidden_layer_sizes": (129,),
              "learning_rate_init": 0.010457046844512303,
              "n_iter_no_change": 32,
              "shuffle": True,
              "solver": "adam",
              "tol": 0.0001,
              "validation_fraction": 0.1,
          },
          space={
              "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],  # logarithmically spaced values
          },
      ),
      'LLM_model_portfolio_29': Component(
            name = 'LLM_model_portfolio_29',
            item = MLPClassifier,
            config={
                "activation": "tanh",
                "alpha": 0.039008613885296826,
                "batch_size": "auto",
                "beta_1": 0.9,
                "beta_2": 0.999,
                "early_stopping": False,
                "epsilon": 1e-08,
                "hidden_layer_sizes": (140, 140, 140),
                "learning_rate_init": 0.0023110285365222803,
                "n_iter_no_change": 32,
                "shuffle": True,
                "solver": "adam",
                "tol": 0.0001,
            },
            space={
                "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],  # logarithmically spaced values
            },
        ),
       'LLM_model_portfolio_30': Component(
          name = 'LLM_model_portfolio_30',
          item = ExtraTreesClassifier,
          config={
              "bootstrap": False,
              "criterion": "entropy",
              "max_features": 0.35771994991556005,
              "min_impurity_decrease": 0.0,
              "min_samples_leaf": 2,
              "min_samples_split": 15,
              "min_weight_fraction_leaf": 0.0,
          },
          space={
              "max_depth": list(range(1, 21)),  # depths from 1 to 20 and None
              "max_leaf_nodes": list(range(2, 21)),  # leaf nodes from 2 to 20 and None
          },
      ),
      'LLM_model_portfolio_31': Component(
          name = 'LLM_model_portfolio_31',
          item = GradientBoostingClassifier,
          config={
              "learning_rate": 0.06202602561543075,
              "min_samples_leaf": 16,
              "max_leaf_nodes": 3,
              "tol": 1e-07,
          },
          space={
              "max_depth": list(range(1, 21)),  # depths from 1 to 20 and None
          },
      ),
      'LLM_model_portfolio_32': Component(
          name = 'LLM_model_portfolio_32',
          item = GradientBoostingClassifier,
          config={
              "learning_rate": 0.02352567307613563,
              "min_samples_leaf": 2,
              "max_leaf_nodes": 654,
              "tol": 1e-07,
              "n_iter_no_change": 19,
              "validation_fraction": 0.29656195578950684,
          },
          space={
              "max_depth": list(range(1, 21)),  # depths from 1 to 20 and None
          },
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

LLM_generated_regression_search_space = {
    'LLM_model_regressor_porfolio_1': Component(
        name = 'LLM_model_regressor_porfolio_1',
        item = RandomForestRegressor,
        config={
            "max_depth": 15,
            "criterion": "squared_error",
        },
        space={
            "n_estimators": list(range(50, 200, 10)),  # number of trees in the forest
            "max_features": ['auto', 'sqrt', 'log2'],  # number of features to consider when looking for the best split
        },
    ),
    'LLM_model_regressor_porfolio_2': Component(
        name = 'LLM_model_regressor_porfolio_2',
        item = ExtraTreesRegressor,
        config={
            "min_samples_leaf": 15,
            "criterion": "squared_error",
        },
        space={
            "n_estimators": list(range(50, 200, 10)),  # number of trees in the forest
            "max_features": ['auto', 'sqrt', 'log2'],  # number of features to consider when looking for the best split
            "max_depth": list(range(1, 21)),  # depths from 1 to 20 and None
        },
    ),
    'LLM_model_regressor_porfolio_3': Component(
        name = 'LLM_model_regressor_porfolio_3',
        item = RandomForestRegressor,
        config={
            "min_samples_leaf": 5,
            "max_leaf_nodes": 50000,
            "max_features": 0.5,
        },
        space={
            "n_estimators": list(range(50, 200, 10)),  # number of trees in the forest
            "max_depth": list(range(1, 21)),  # depths from 1 to 20
        },
    ),
    'LLM_model_regressor_porfolio_4': Component(
          name = 'LLM_model_regressor_porfolio_4',
          item = ExtraTreesRegressor,
          config={
              "min_samples_leaf": 1,
              "max_leaf_nodes": 15000,
              "max_features": 0.5,
          },
          space={
              "n_estimators": list(range(50, 200, 10)),  # number of trees in the forest
              "max_depth": list(range(1, 21)),  # depths from 1 to 20
          },
      ),
      # The next components come from https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/configs/zeroshot/zeroshot_portfolio_2023.py
      # We assume that the criterion should be fixed and then we can play with the rest of the hyperparameters of the XT (extra trees) and RF (random forest)
    'LLM_model_regressor_porfolio_5': Component(
          name = 'LLM_model_regressor_porfolio_5',
          item = RandomForestRegressor,
          config={
              "criterion": "squared_error",
              "max_features": 0.75,
              "max_leaf_nodes": 37308,
              "min_samples_leaf": 1,
          },
          space={
              "n_estimators": list(range(50, 200, 10)),  # number of trees in the forest
              "max_depth": list(range(1, 21)),  # depths from 1 to 20
          },
      ),
    'LLM_model_regressor_porfolio_6': Component(
          name = 'LLM_model_regressor_porfolio_6',
          item = RandomForestRegressor,
          config={
              "criterion": "squared_error",
              "max_features": 0.75,
              "max_leaf_nodes": 28310,
              "min_samples_leaf": 2,
          },
          space={
              "n_estimators": list(range(50, 200, 10)),  # number of trees in the forest
              "max_depth": list(range(1, 21)),  # depths from 1 to 20
          },
      ),
    'LLM_model_regressor_porfolio_7': Component(
          name = 'LLM_model_regressor_porfolio_7',
          item = RandomForestRegressor,
          config={
              "criterion": "squared_error",
              "max_features": 1.0,
              "max_leaf_nodes": 38572,
              "min_samples_leaf": 5,
          },
          space={
              "n_estimators": list(range(50, 200, 10)),  # number of trees in the forest
              "max_depth": list(range(1, 21)),  # depths from 1 to 20
          },
      ),
    'LLM_model_regressor_porfolio_8': Component(
        name = 'LLM_model_regressor_porfolio_8',
        item = RandomForestRegressor,
        config={
            "criterion": "squared_error",
            "max_features": 0.75,
            "max_leaf_nodes": 18242,
            "min_samples_leaf": 40,
        },
        space={
            "n_estimators": list(range(50, 200, 10)),  # number of trees in the forest
            "max_depth": list(range(1, 21)),  # depths from 1 to 20
        },
    ),
    'LLM_model_regressor_porfolio_9': Component(
        name = 'LLM_model_regressor_porfolio_9',
        item = RandomForestRegressor,
        config={
            "criterion": "squared_error",
            "max_features": "log2",
            "max_leaf_nodes": 42644,
            "min_samples_leaf": 1,
        },
        space={
            "n_estimators": list(range(50, 200, 10)),  # number of trees in the forest
            "max_depth": list(range(1, 21)),  # depths from 1 to 20
        },
    ),
    'LLM_model_regressor_porfolio_10': Component(
          name = 'LLM_model_regressor_porfolio_10',
          item = RandomForestRegressor,
          config={
              "criterion": "squared_error",
              "max_features": 0.75,
              "max_leaf_nodes": 36230,
              "min_samples_leaf": 3,
          },
          space={
              "n_estimators": list(range(50, 200, 10)),  # number of trees in the forest
              "max_depth": list(range(1, 21)),  # depths from 1 to 20
          },
      ),
    'LLM_model_regressor_porfolio_11': Component(
            name = 'LLM_model_regressor_porfolio_11',
            item = RandomForestRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 1.0,
                "max_leaf_nodes": 48136,
                "min_samples_leaf": 1,
            },
            space={
                "n_estimators": list(range(50, 200, 10)),  # number of trees in the forest
                "max_depth": list(range(1, 21)),  # depths from 1 to 20
            },
        ),
    'LLM_model_regressor_porfolio_12': Component(
            name = 'LLM_model_regressor_porfolio_12',
            item = ExtraTreesRegressor,
            config={
                "criterion": "squared_error",
                "max_features": 0.75,
                "max_leaf_nodes": 18392,
                "min_samples_leaf": 1,
            },
            space={
                "n_estimators": list(range(50, 200, 10)),  # number of trees in the forest
                "max_depth": list(range(1, 21)),  # depths from 1 to 20
            },
        ),
    'LLM_model_regressor_porfolio_13': Component(
          name = 'LLM_model_regressor_porfolio_13',
          item = ExtraTreesRegressor,
          config={
              "criterion": "squared_error",
              "max_features": 0.75,
              "max_leaf_nodes": 18392,
              "min_samples_leaf": 1,
          },
          space={
              "n_estimators": list(range(50, 200, 10)),  # number of trees in the forest
              "max_depth": list(range(1, 21)),  # depths from 1 to 20
          },
      ),
    'LLM_model_regressor_porfolio_14': Component(
            name = 'LLM_model_regressor_porfolio_14',
            item = ExtraTreesRegressor,
            config={
              "criterion": "squared_error",
              "max_features": 1.0,
              "max_leaf_nodes": 12845,
              "min_samples_leaf": 4,
          },
          space={
              "n_estimators": list(range(50, 200, 10)),  # number of trees in the forest
              "max_depth": list(range(1, 21)),  # depths from 1 to 20
          },
      ),
    'LLM_model_regressor_porfolio_15': Component(
            name = 'LLM_model_regressor_porfolio_15',
            item = ExtraTreesRegressor,
            config={
              "criterion": "squared_error",
              "max_features": "sqrt",
              "max_leaf_nodes": 28532,
              "min_samples_leaf": 1,
          },
          space={
              "n_estimators": list(range(50, 200, 10)),  # number of trees in the forest
              "max_depth": list(range(1, 21)),  # depths from 1 to 20
          },
      ),
    'LLM_model_regressor_porfolio_16': Component(
          name = 'LLM_model_regressor_porfolio_16',
          item = ExtraTreesRegressor,
          config={
              "criterion": "squared_error",
              "max_features": 1.0,
              "max_leaf_nodes": 19935,
              "min_samples_leaf": 20,
          },
          space={
              "n_estimators": list(range(50, 200, 10)),  # number of trees in the forest
              "max_depth": list(range(1, 21)),  # depths from 1 to 20
          },
      ),
    'LLM_model_regressor_porfolio_17': Component(
          name = 'LLM_model_regressor_porfolio_17',
          item = ExtraTreesRegressor,
          config={
              "criterion": "squared_error",
              "max_features": 0.75,
              "max_leaf_nodes": 29813,
              "min_samples_leaf": 4,
          },
          space={
              "n_estimators": list(range(50, 200, 10)),  # number of trees in the forest
              "max_depth": list(range(1, 21)),  # depths from 1 to 20
          },
      ),
    'LLM_model_regressor_porfolio_18': Component(
          name = 'LLM_model_regressor_porfolio_18',
          item = ExtraTreesRegressor,
          config={
              "criterion": "squared_error",
              "max_features": 1.0,
              "max_leaf_nodes": 40459,
              "min_samples_leaf": 1,
          },
          space={
              "n_estimators": list(range(50, 200, 10)),  # number of trees in the forest
              "max_depth": list(range(1, 21)),  # depths from 1 to 20
          },
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
        space={
            "n_estimators": list(range(50, 200, 10)),  # number of trees in the forest
            "max_depth": list(range(1, 21)),  # depths from 1 to 20
        },
    ),
}