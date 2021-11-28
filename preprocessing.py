import pandas as pd
import numpy as np

#pd.set_option('display.max_columns', None)

root = '../Dataset_FILTERED/'

# Lecture des fichiers csv en tableau pandas
#browsing_file = pd.read_csv(root + 'browsing_filtered.csv')
browsing_file = pd.read_csv(root + 'sub_browsing_filtered.csv')
search_file = pd.read_csv(root + 'search_filtered.csv')
sku_file = pd.read_csv(root + 'sku_filtered.csv')


"""
Création d'un dataframe contenant les différents features nécessaires
      session_id_hash (peut-être pertinant de remplacer par un id non-encrypté)
      product_sku_hash (peut-être pertinant de remplacer par un id non-encrypté)
      is_purchased
      add_count_during_session
      add_has_been_detailed
      session_length
      add_price
"""

# Dataframe ou une ligne = un AC
featureDataframe = browsing_file.query("product_action == 'add'")
featureDataframe = featureDataframe.drop(['event_type', 'product_action', 'server_timestamp_epoch_ms', 'hashed_url'], axis=1)

sessions = featureDataframe['session_id_hash'].unique().tolist() # Liste des sessions distinctes ayant un AC


# Création du vecteur de réponse
target = browsing_file.query("product_action == 'purchase'")[['session_id_hash','product_sku_hash']].drop_duplicates()
target['has_been_purchased'] = np.ones(len(target))
target = featureDataframe.merge(target, how='left', on=['session_id_hash','product_sku_hash'], sort=False)['has_been_purchased']
target = target.fillna(value=0)


# Ajout du feature indiquant le nombre de AC dans la session
featureDataframe['add_count_during_session'] = featureDataframe.groupby(by='session_id_hash', sort=False)['session_id_hash'].transform('count')

# Ajout du feature indiquant si le produit a été détaillé
#featureDataframe['has_been_detailed'] = has_been_detailed
has_been_detailed = browsing_file.query("product_action == 'detail'")[['session_id_hash','product_sku_hash']].drop_duplicates()
has_been_detailed['has_been_detailed'] = np.ones(len(has_been_detailed))
featureDataframe = featureDataframe.merge(has_been_detailed, how='left', on=['session_id_hash','product_sku_hash'], sort=False)

# *** Feature retiré car étrangement chaque remove est fait sur un add n'ayant pas eu lieu durant la session ***
# Ajout du feature indiquant si le produit a été removed
#has_been_removed = browsing_file.query("product_action == 'remove'")[['session_id_hash', 'product_sku_hash']]
#has_been_removed['has_been_removed'] = np.ones(len(has_been_removed))
#featureDataframe = featureDataframe.merge(has_been_removed, how='left', left_on=['session_id_hash', 'product_sku_hash'], right_on=['session_id_hash', 'product_sku_hash'], sort=False)
#featureDataframe = featureDataframe.fillna(value={'has_been_removed':0})

# Ajout du feature indiquant le prix du produit
featureDataframe = featureDataframe.merge(sku_file[['product_sku_hash', 'price_bucket']], on='product_sku_hash', how='left')

# Ajout du feature indiquant la durée de la session
minTimeStamp = np.array(browsing_file.groupby(['session_id_hash'], sort=False)['server_timestamp_epoch_ms'].min().tolist())
maxTimeStamp = np.array(browsing_file.groupby(['session_id_hash'], sort=False)['server_timestamp_epoch_ms'].max().tolist())
minMaxTimeStamp = list(zip(sessions, (maxTimeStamp-minTimeStamp)))
session_length = pd.DataFrame(data=minMaxTimeStamp, columns=['session_id_hash', 'session_length'])
featureDataframe = featureDataframe.merge(session_length, on='session_id_hash', how='left')

# Ajout du feature indiquant le nombre d'intéraction durant la session
browsing_file['session_interaction_count'] = browsing_file.groupby(by='session_id_hash')['session_id_hash'].transform('count')
session_interaction = browsing_file.drop_duplicates(subset='session_id_hash')
featureDataframe = featureDataframe.merge(session_interaction[['session_id_hash', 'session_interaction_count']], on='session_id_hash', how='left')

# Ajout du feature indiquant le nombre de recherche durant la session
search_file['session_query_count'] = search_file.groupby(by='session_id_hash')['session_id_hash'].transform('count')
session_query_count = search_file.drop_duplicates(subset='session_id_hash')
featureDataframe = featureDataframe.merge(session_query_count[['session_id_hash', 'session_query_count']], on='session_id_hash', how='left')


featureDataframe = featureDataframe.fillna(value=0)
featureDataframe.to_csv('features.csv', header=True, index=False)