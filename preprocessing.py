import pandas as pd
import numpy as np
from sklearn import preprocessing

pd.options.mode.chained_assignment = None
#pd.set_option('display.max_columns', None)

root = '../Dataset_FILTERED/'

# Lecture des fichiers csv en tableau pandas
browsing_file = pd.read_csv(root + 'browsing_filtered.csv')
#browsing_file = pd.read_csv(root + 'sub_browsing_filtered.csv')
search_file = pd.read_csv(root + 'search_filtered.csv')
sku_file = pd.read_csv(root + 'sku_filtered.csv')


"""
Création d'un dataframe contenant les différents features nécessaires
    product_id
    add_count_during_session
    has_been_detailed
    price_bucket
    session_length
    session_interaction_count
    session_detail_count
    session_pageview_count
    session_query_count
    nb_click_before
    nb_click_after
    nb_add_before
    nb_add_after
"""

# Ajout du id de chaque ligne dans les informations complète d'une session
browsing_file['interaction_id'] = browsing_file.groupby('session_id_hash').cumcount()


# Ajout du id de chaque action par session
browsing_file['action_id'] = browsing_file.groupby(['session_id_hash', 'product_action']).cumcount()


# Ajout d'un id de produit dans le fichier sku
sku_file['product_id'] = np.arange(len(sku_file))


# Dataframe ou une ligne = un AC
featureDataframe = browsing_file.query("product_action == 'add'")
featureDataframe = featureDataframe.drop(['event_type', 'product_action', 'server_timestamp_epoch_ms'], axis=1)


# Liste des sessions distinctes ayant un AC
sessions_list = featureDataframe['session_id_hash'].drop_duplicates().tolist()


# Création du vecteur de réponse
target = browsing_file.query("product_action == 'purchase'")[['session_id_hash','product_sku_hash']].drop_duplicates()
target['has_been_purchased'] = np.ones(len(target))
target = featureDataframe.merge(target, how='left', on=['session_id_hash','product_sku_hash'], sort=False)['has_been_purchased']
target = target.fillna(value=0)


# Ajout du id du produit
featureDataframe = featureDataframe.merge(sku_file[['product_sku_hash','product_id']], on='product_sku_hash', how='left', sort=False)


# Ajout du feature indiquant le nombre de AC dans la session
featureDataframe['add_count_during_session'] = featureDataframe.groupby(by='session_id_hash', sort=False)['session_id_hash'].transform('count')


# Ajout du feature indiquant si le produit a été détaillé
has_been_detailed = browsing_file.query("product_action == 'detail'")[['session_id_hash','product_sku_hash']].drop_duplicates()
has_been_detailed['has_been_detailed'] = np.ones(len(has_been_detailed))
featureDataframe = featureDataframe.merge(has_been_detailed, how='left', on=['session_id_hash','product_sku_hash'], sort=False)
featureDataframe['has_been_detailed'] = featureDataframe['has_been_detailed'].fillna(value=0)


# *** Feature retiré car étrangement chaque remove est fait sur un add n'ayant pas eu lieu durant la session ***
# Ajout du feature indiquant si le produit a été removed
#has_been_removed = browsing_file.query("product_action == 'remove'")[['session_id_hash', 'product_sku_hash']]
#has_been_removed['has_been_removed'] = np.ones(len(has_been_removed))
#featureDataframe = featureDataframe.merge(has_been_removed, how='left', left_on=['session_id_hash', 'product_sku_hash'], right_on=['session_id_hash', 'product_sku_hash'], sort=False)


# Ajout du feature indiquant le prix du produit
featureDataframe = featureDataframe.merge(sku_file[['product_sku_hash', 'price_bucket']], on='product_sku_hash', how='left')
featureDataframe['price_bucket'] = featureDataframe['price_bucket'].fillna(value=7)


# Ajout du feature indiquant la durée de la session
minTimeStamp = np.array(browsing_file.groupby(['session_id_hash'], sort=False)['server_timestamp_epoch_ms'].min().tolist())
maxTimeStamp = np.array(browsing_file.groupby(['session_id_hash'], sort=False)['server_timestamp_epoch_ms'].max().tolist())
minMaxTimeStamp = list(zip(sessions_list, (maxTimeStamp-minTimeStamp)))
session_length = pd.DataFrame(data=minMaxTimeStamp, columns=['session_id_hash', 'session_length'])
featureDataframe = featureDataframe.merge(session_length, on='session_id_hash', how='left', sort=False)


# Ajout du feature indiquant le nombre d'intéraction durant la session
browsing_file['session_interaction_count'] = browsing_file.groupby(by='session_id_hash')['session_id_hash'].transform('count')
session_interaction = browsing_file.drop_duplicates(subset='session_id_hash')
featureDataframe = featureDataframe.merge(session_interaction[['session_id_hash', 'session_interaction_count']], on='session_id_hash', how='left', sort=False)


# Ajout du feature indiquant le nombre de detail durant la session
detail_interaction = browsing_file.query("product_action == 'detail'")
detail_interaction['session_detail_count'] = detail_interaction.groupby(['session_id_hash'])['session_id_hash'].transform('count')
detail_interaction = detail_interaction.drop_duplicates(subset='session_id_hash')
featureDataframe = featureDataframe.merge(detail_interaction[['session_id_hash', 'session_detail_count']], on='session_id_hash', how='left', sort=False)
featureDataframe['session_detail_count'] = featureDataframe['session_detail_count'].fillna(value=0)


# Ajout du feature indiquant le nombre de pageview durant la session
pageview_interaction = browsing_file.query("event_type == 'pageview'")
pageview_interaction['session_pageview_count'] = pageview_interaction.groupby(['session_id_hash'])['session_id_hash'].transform('count')
pageview_interaction = pageview_interaction.drop_duplicates(subset='session_id_hash')
featureDataframe = featureDataframe.merge(pageview_interaction[['session_id_hash','session_pageview_count']], on='session_id_hash', how='left', sort=False)
featureDataframe['session_pageview_count'] = featureDataframe['session_pageview_count'].fillna(value=0)


# Ajout du feature indiquant le nombre de recherche durant la session
search_file['session_query_count'] = search_file.groupby(by='session_id_hash')['session_id_hash'].transform('count')
session_query_count = search_file.drop_duplicates(subset='session_id_hash')
featureDataframe = featureDataframe.merge(session_query_count[['session_id_hash', 'session_query_count']], on='session_id_hash', how='left', sort=False)
featureDataframe['session_query_count'] = featureDataframe['session_query_count'].fillna(value=0)


# Ajout du feature indiquant le nombre d'intéraction avant le add
featureDataframe['nb_click_before'] = featureDataframe['interaction_id']


# Ajout du feature indiquant le nombre d'intéraction après le add
featureDataframe['nb_click_after'] = featureDataframe['session_interaction_count'] - featureDataframe['interaction_id'] - 1


# Ajout du feature indiquant le nombre de AC avant le add
featureDataframe['nb_add_before'] = featureDataframe['action_id']


# Ajout du feature indiquant le nombre de AC après le add
featureDataframe['nb_add_after'] = featureDataframe['add_count_during_session'] - featureDataframe['action_id'] - 1


# Ajout de la description du produit pour le Target Encoding
featureDataframe = featureDataframe.merge(sku_file[['product_sku_hash','description_data', 'image_data', 'category_hash']], on='product_sku_hash', how='left', sort=False)
featureDataframe.to_csv('features_te.csv', header=True, index=False)

featureDataframe = featureDataframe.drop(['session_id_hash', 'product_sku_hash', 'hashed_url', 'interaction_id', 'action_id', 'category_hash'], axis=1)

min_max_scaler = preprocessing.MinMaxScaler()
feature_name = featureDataframe.columns.values.tolist()
for name in feature_name:
    featureDataframe[name] = min_max_scaler.fit_transform(np.array(featureDataframe[name].tolist()).reshape(-1,1))


featureDataframe.to_csv('features.csv', header=True, index=False)
target.to_csv('target.csv', header=True, index=False)