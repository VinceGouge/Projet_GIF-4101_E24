import pandas as pd
import numpy as np

#pd.set_option('display.max_columns', None)

root = '../Dataset_FILTERED/'

# Lecture des fichiers csv en tableau pandas
#browsing_file = pd.read_csv(root + 'browsing_filtered.csv')
#search_file = pd.read_csv(root + 'search_filtered.csv')
sku_file = pd.read_csv(root + 'sku_filtered.csv')
browsing_file = pd.read_csv(root + 'sub_browsing_filtered.csv')
search_file = pd.read_csv(root + 'sub_search_filtered.csv')
#sku_file = pd.read_csv(root + 'sub_sku_filtered.csv')

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
reindexedDataframe = featureDataframe.reset_index()

is_purchased = np.zeros(len(featureDataframe))
has_been_detailed = np.zeros(len(featureDataframe))

sessions = featureDataframe['session_id_hash'].unique() # Liste des sessions distinctes ayant un AC
for session in sessions:
   session_AC = reindexedDataframe.query("session_id_hash == @session")
   N = len(session_AC)
   index = session_AC.index

   purchased_product = browsing_file.query("session_id_hash == @session & product_action == 'purchase'")["product_sku_hash"]
   if not purchased_product.empty:
       purchased_product = purchased_product.tolist()
       for prod in purchased_product:
           index_purchased = index[session_AC["product_sku_hash"] == prod]
           is_purchased[index_purchased] = 1

   detailed_product = browsing_file.query("session_id_hash == @session & product_action == 'detail'")["product_sku_hash"]
   if not detailed_product.empty:
       detailed_product = detailed_product.tolist()
       for detail in detailed_product:
           index_detailed = index[session_AC["product_sku_hash"] == detail]
           has_been_detailed[index_detailed] = 1


# Ajout du feature indiquant si le produit a été acheté
featureDataframe['is_purchased'] = is_purchased

# Ajout du feature indiquant le nombre de AC dans la session
featureDataframe['add_count_during_session'] = featureDataframe.groupby(by='session_id_hash')['session_id_hash'].transform('count')

# Ajout du feature indiquant si le produit a été détaillé
featureDataframe['has_been_detailed'] = has_been_detailed

# Ajout du feature indiquant le prix du produit
added_product_price = pd.merge(featureDataframe, sku_file, on='product_sku_hash')['price_bucket'].tolist()
#featureDataframe = featureDataframe.reset_index()
featureDataframe['price'] = added_product_price


featureDataframe.to_csv('features.csv', header=True, index=False)