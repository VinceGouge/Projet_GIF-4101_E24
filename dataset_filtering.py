import pandas as pd
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity

# Lecture des fichiers csv en tableau pandas
browsing_file = pd.read_csv('../Dataset_RAW/train/browsing_train.csv')
search_file = pd.read_csv('../Dataset_RAW/train/search_train.csv')
sku_file = pd.read_csv('../Dataset_RAW/train/sku_to_content.csv')

sessions_id_with_AC = browsing_file.query("product_action == 'add'")["session_id_hash"].unique().tolist()   # liste des sessions ayant un AC
added_product = browsing_file.query("product_action == 'add'")["product_sku_hash"].unique().tolist()    # liste des produits ayant été AC
browsingFiltered = browsing_file.query("session_id_hash == @sessions_id_with_AC")   # filtrage de browsing pour session avec AC
searchFiltered = search_file.query("session_id_hash == @sessions_id_with_AC")   # filtrage de search pour garder uniquement session avec AC
sku_filtered = sku_file.query("product_sku_hash == @added_product") # filtrage de sku pour garder uniqument produit ayant été AC
print('read_done')

sku_filtered['description_vector'] = sku_filtered['description_vector'].apply(lambda s: [float(x.strip(' []')) for x in str(s).split(',')])

description_data = []
product_description = sku_filtered['description_vector'].tolist()
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
    description_data.append(similarity)
sku_filtered['description_data'] = np.array(description_data , dtype=np.int32)
print('description_data done')
sku_filtered['image_vector'] = sku_filtered['image_vector'].apply(lambda s: [float(x.strip(' []')) for x in str(s).split(',')])
image_data = []
image_description = sku_filtered['image_vector'].tolist()
for vector in image_description :
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
    image_data.append(similarity)
sku_filtered['image_data'] = np.array(image_data , dtype=np.int32)
print('image_data done')

# création des fichiers csv filtrés
browsingFiltered.to_csv(r'../Dataset_FILTERED/browsing_filtered.csv', header=True, index=False)
searchFiltered.to_csv(r'../Dataset_FILTERED/search_filtered.csv', header=True, index=False)
sku_filtered.to_csv(r'../Dataset_FILTERED/sku_filtered.csv', header=True, index=False)

# 100000 premières lignes des fichiers filtré
subset_browsingFiltered = browsingFiltered.head(10000)
subset_browsingFiltered.to_csv(r'../Dataset_FILTERED/sub_browsing_filtered.csv', header=True, index=False)
subset_searchFiltered = searchFiltered.head(10000)
subset_searchFiltered.to_csv(r'../Dataset_FILTERED/sub_search_filtered.csv', header=True, index=False)
subset_skuFiltered = sku_filtered.head(10000)
subset_skuFiltered.to_csv(r'../Dataset_FILTERED/sub_sku_filtered.csv', header=True, index=False)
