import pandas as pd

# Lecture des fichiers csv en tableau pandas
browsing_file = pd.read_csv('../Dataset_RAW/train/browsing_train.csv')
search_file = pd.read_csv('../Dataset_RAW/train/search_train.csv')
sku_file = pd.read_csv('../Dataset_RAW/train/sku_to_content.csv')

sessions_id_with_AC = browsing_file.query("product_action == 'add'")["session_id_hash"].unique().tolist()   # liste des sessions ayant un AC
added_product = browsing_file.query("product_action == 'add'")["product_sku_hash"].unique().tolist()    # liste des produits ayant été AC
browsingFiltered = browsing_file.query("session_id_hash == @sessions_id_with_AC")   # filtrage de browsing pour session avec AC
searchFiltered = search_file.query("session_id_hash == @sessions_id_with_AC")   # filtrage de search pour garder uniquement session avec AC
sku_filtered = sku_file.query("product_sku_hash == @added_product") # filtrage de sku pour garder uniqument produit ayant été AC

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
