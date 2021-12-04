
import h2o
import category_encoders as ce
from h2o.estimators import H2OTargetEncoderEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import pandas as pd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
#import tensorflow as tf
#from tensorflow import keras
from sklearn.metrics.pairwise import cosine_similarity
h2o.init()
import sklearn
#sku_file = pd.read_csv('features.csv')
#autos["make_encoded"] = autos.groupby("make")["price"].transform("mean")

#autos[["make", "price", "make_encoded"]].head(10)
#
#Create the Dataframe
"""
# Ajout de la description du produit pour le Target Encoding
featureDataframe = featureDataframe.merge(sku_file[['product_sku_hash','description_vector']], on='product_sku_hash', how='left', sort=False)

# Ajout de la description du produit pour le Target Encoding
featureDataframe = featureDataframe.merge(sku_file[['product_sku_hash','description_vector']], on='product_sku_hash', how='left', sort=False)
#modifier le description_vector pour le rendre numérique 
featureDataframe['description_vector'] = featureDataframe['description_vector'].apply(lambda s: [float(x.strip(' []')) for x in str(s).split(',')])

description_vector = []
product_description = featureDataframe['description_vector'].tolist()
for vector in product_description :
    try:
      x = vector / np.linalg.norm(vector)
      y = np.ones(len(vector))  
      if type(x) is np.ndarray: x = x.reshape(1, -1) # get rid of the warning
      if type(y) is np.ndarray: y = y.reshape(1, -1)
      d = cosine_similarity(x, y)
      d = d[0][0]
    except:
      d = 0.0
    similarity = d * 10000
    description_vector.append(similarity)
featureDataframe['description_vector'] = np.array(description_vector , dtype=np.int32)

# Ajout de la description du produit pour le Target Encoding
featureDataframe = featureDataframe.merge(sku_file[['product_sku_hash','description_vector']], on='product_sku_hash', how='left', sort=False)
#modifier le description_vector pour le rendre numérique 
featureDataframe['description_vector'] = featureDataframe['description_vector'].apply(lambda s: [float(x.strip(' []')) for x in str(s).split(',')])

description_vector = []
product_description = featureDataframe['description_vector'].tolist()
for vector in product_description :
    try:
      x = vector / np.linalg.norm(vector)
      y = np.ones(len(vector))  
      if type(x) is np.ndarray: x = x.reshape(1, -1) # get rid of the warning
      if type(y) is np.ndarray: y = y.reshape(1, -1)
      d = cosine_similarity(x, y)
      d = d[0][0]
    except:
      d = 0.0
    similarity = d * 10000
    description_vector.append(similarity)
featureDataframe['description_vector'] = np.array(description_vector , dtype=np.int32)


features = pd.read_csv('features.csv')


features["description_vector"] = features["description_vector"].apply(lambda s: [float(x.strip(' []')) for x in s.split(',')])
product_id = features["product_sku_hash"].tolist()
price_bucket = features["price_bucket"].tolist()
hash_url = features["hashed_url"].tolist()
product_description = features["description_vector"].tolist()


data_label_string = 'product_id'
data_label = product_id
#selon les data
data_X = [price_bucket, hash_url, product_description]
data_X_label = ['price_bucket', 'hash_url', 'product_description']

product_id_list=[]
k_list = [0,2]

description_num = []
for element in product_description :
    x = element / np.linalg.norm(element)
    y = np.ones(len(element))
    try:
        if type(x) is np.ndarray: x = x.reshape(1, -1) # get rid of the warning
        if type(y) is np.ndarray: y = y.reshape(1, -1)
        d = cosine_similarity(x, y)
        d = d[0][0]
    except:
        print (x)
        print (y)
        d = 0.0
    similarity = d * 10000
    description_num.append(similarity)
description_vector = np.array(description_num , dtype=np.int32)

data_X[2] = description_num
for k in k_list:
    data = pd.DataFrame({data_label_string:data_label,data_X_label[k]:data_X[k]})

    #data=pd.DataFrame({'class':['A,','B','C','B','C','A','A','A'],'Marks':[50,30,70,80,45,97,80,68]})

    #Create target encoding object
    encoder=ce.TargetEncoder(cols=data_label_string, return_df=True) 
    data2 = encoder.fit_transform(data[data_label_string],data[data_X_label[k]])
    product_id_mean = data2[data_label_string].tolist()
    product_id_list.append(product_id_mean)

data_product = pd.DataFrame({'price': product_id_list[0], 'description': product_id_list[1], 'list': product_id_list[2]})
data_product.to_csv(r'product_id_list.csv', header=True, index=False) 
features["description_vector"]  = description_vector
features.to_csv(r'features.csv', header=True, index=False)

"""


#sku_file = pd.read_csv('../Dataset_FILTERED/sku_filtered.csv')

features = h2o.import_file('features_te.csv')
#sku_file = h2o.import_file('../Dataset_FILTERED/sku_filtered.csv')
session_query_count = features['description_data'].asfactor()
response = 'description_data'

seed =1234
#data= pd.DataFrame({'sessions_querry_count': session_query_count, 'product_id':product_id, 'price_bucket' : price_bucket, 'hash_url':hash_url, 'description_num':description_num})

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

# New target encoded train and test sets
#train_te = product_te.transform(frame=train, as_training=True)
#test_te = product_te.transform(frame=test, noise=0)

feature_te = product_te.transform(frame=features)
#data_all = h2o.as_list(feacture_total)
min_max_scaler = preprocessing.MinMaxScaler()
feature_df = h2o.as_list(feature_te)
feature_df['category_hash_te'] = min_max_scaler.fit_transform(np.array(feature_df['category_hash_te'].tolist()).reshape(-1, 1))
feature_df['product_sku_hash_te'] = min_max_scaler.fit_transform(np.array(feature_df['product_sku_hash_te'].tolist()).reshape(-1, 1))
feature_df['hashed_url_te'] = min_max_scaler.fit_transform(np.array(feature_df['hashed_url_te'].tolist()).reshape(-1, 1))
feature_df['image_data'] = min_max_scaler.fit_transform(np.array(feature_df['description_data'].tolist()).reshape(-1, 1))
feature_df['image_data'] = min_max_scaler.fit_transform(np.array(feature_df['description_data'].tolist()).reshape(-1, 1))

feature_df = feature_df.drop(['session_id_hash', 'product_sku_hash', 'hashed_url', 'interaction_id', 'action_id', 'category_hash'], axis=1)
feature_df_sub = feature_df.head(10000)
feature_df_sub.to_csv(r'features_te_sub.csv', header=True, index=False) 
feature_df.to_csv(r'features_te.csv', header=True, index=False) 

h = 2