import pandas as pd

# Lecture des fichiers csv en tableau pandas
browsing_file = pd.read_csv('../Dataset_RAW/train/browsing_train.csv')
search_file = pd.read_csv('../Dataset_RAW/train/search_train.csv')
sku_file = pd.read_csv('../Dataset_RAW/train/sku_to_content.csv')

session_id_with_AC = browsing_file.query("product_action == 'add'")["session_id_hash"]          # session ayant un AC
added_product = browsing_file.query("product_action == 'add'")["product_sku_hash"]                # produit ayant été AC
browsingFiltered = pd.merge(browsing_file, session_id_with_AC, how='inner', on='session_id_hash')     # filtrage de browsing pour session avec AC
searchFiltered = pd.merge(search_file, session_id_with_AC, how='inner', on='session_id_hash')         # filtrage de search pour session avec AC
skuFiltered = pd.merge(sku_file, added_product, how="inner", on="product_sku_hash")                   # fitrage de sku pour produit AC

# création des fichiers csv filtrés
browsingFiltered.to_csv(r'../Dataset_FILTERED/browsing_filtered.csv', header=True)
searchFiltered.to_csv(r'../Dataset_FILTERED/search_filtered.csv', header=True)
skuFiltered.to_csv(r'../Dataset_FILTERED/sku_filtered.csv', header=True)

# 1000 premières lignes des fichiers filtré
subset_browsingFiltered = browsingFiltered.head(1000)
subset_browsingFiltered.to_csv(r'../Dataset_FILTERED/sub_browsing_filtered.csv', header=True)
subset_searchFiltered = searchFiltered.head(1000)
subset_searchFiltered.to_csv(r'../Dataset_FILTERED/sub_search_filtered.csv', header=True)
subset_skuFiltered = skuFiltered.head(1000)
subset_skuFiltered.to_csv(r'../Dataset_FILTERED/sub_sku_filtered.csv', header=True)
