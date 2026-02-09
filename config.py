with open('features_tags.txt') as f:
    feats = f.readlines()

feats = [i.replace('\n', '') for i in feats]

# Define the XGBoost model with the best parameters
# best_params = {
#     'n_estimators': 151,
#     'max_depth': 4,
#     'learning_rate': 0.065,
#     'subsample': 0.727,
#     'colsample_bytree': 0.7,
#     'gamma': 8.618273879856856e-06,
#     'min_child_weight': 8,
#     'reg_alpha': 0.9708432445596958,
#     'reg_lambda': 0.0514046525837813,
#     'random_state': 42,
#     'n_jobs': 8,
#     'objective': 'reg:squarederror',
#     'tree_method': 'hist'
# }