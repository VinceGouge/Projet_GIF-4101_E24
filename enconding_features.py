
import h2o
from h2o.estimators import H2OTargetEncoderEstimator
import pandas as pd
import numpy as np
from sklearn import preprocessing
h2o.init()
import sklearn

features = h2o.import_file('features_te.csv')
session_query_count = features['description_data'].asfactor()
response = 'description_data'

seed =1234
train, test = features.split_frame(ratios = [.8], seed = 1234)
encoded_columns = ['category_hash', 'product_sku_hash', 'hashed_url']
fold_column = "kfold_column"
train[fold_column] = train.kfold_column(n_folds=5, seed=seed)
product_te = H2OTargetEncoderEstimator(fold_column=fold_column,
                                       data_leakage_handling="k_fold",
                                       blending=True,
                                       inflection_point=3,
                                       smoothing=10,
                                       noise=0.15,     # In general, the less data you have the more regularization you need
                                       seed=seed)
product_te.train(x=encoded_columns,
                 y=response,
                 training_frame=train)

feature_te = product_te.transform(frame=features)
feature_df = h2o.as_list(feature_te)
feature_df = feature_df.drop(['session_id_hash', 'product_id', 'product_sku_hash', 'hashed_url', 'interaction_id', 'action_id', 'category_hash'], axis=1)
min_max_scaler = preprocessing.MinMaxScaler()

feature_name = feature_df.columns.values.tolist()
for name in feature_name:
    feature_df[name] = min_max_scaler.fit_transform(np.array(feature_df[name].tolist()).reshape(-1,1))

'''
feature_df['category_hash_te'] = min_max_scaler.fit_transform(np.array(feature_df['category_hash_te'].tolist()).reshape(-1, 1))
feature_df['product_sku_hash_te'] = min_max_scaler.fit_transform(np.array(feature_df['product_sku_hash_te'].tolist()).reshape(-1, 1))
feature_df['hashed_url_te'] = min_max_scaler.fit_transform(np.array(feature_df['hashed_url_te'].tolist()).reshape(-1, 1))
feature_df['description_data'] = min_max_scaler.fit_transform(np.array(feature_df['description_data'].tolist()).reshape(-1, 1))
feature_df['image_data'] = min_max_scaler.fit_transform(np.array(feature_df['image_data'].tolist()).reshape(-1, 1))
'''



feature_df.to_csv(r'features_te.csv', header=True, index=False) 

