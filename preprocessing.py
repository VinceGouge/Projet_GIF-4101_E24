import pandas as pd
import numpy as np

#pd.set_option('display.max_columns', None)

root = '../Dataset_FILTERED/'

# Lecture des fichiers csv en tableau pandas
#browsing_file = pd.read_csv(root + 'browsing_filtered.csv')
browsing_file = pd.read_csv(root + 'sub_browsing_filtered.csv')
#search_file = pd.read_csv(root + 'search_filtered.csv')
#sku_file = pd.read_csv(root + 'sku_filtered.csv')

linesWithAdd = np.where(browsing_file['product_action'] == 'add')[0]    # index de toutes les intéractions AC
linesWithPurchase = np.where(browsing_file['product_action'] == 'purchase')[0]    # index de tous les purchase

# Création d'un dataframe contenant les différents features nécessaires
#       session_id_hash (peut-être pertinant de remplacer par un id non-encrypté)
#       product_sku_hash (peut-être pertinant de remplacer par un id non-encrypté)
#       add_count_during_session
#       add_has_been_detailed
#       session_length
#       add_price

featureDataframe = browsing_file.loc[linesWithAdd]
featureDataframe = featureDataframe.drop(['event_type', 'product_action', 'server_timestamp_epoch_ms', 'hashed_url'], axis=1)
featureDataframe['add_count_during_session']=featureDataframe.groupby(by='session_id_hash')['session_id_hash'].transform('count')


print(featureDataframe)

