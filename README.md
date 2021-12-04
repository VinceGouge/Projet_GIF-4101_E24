# Projet_GIF-4101_E24
Projet final équipe 24 du cours GIF-4101 Introduction à l'apprentissage automatique A21

# Browsing Events
session_id_hash = id_session
event_type = {pageview, event}
if event_type == event :
  product_action = {detail, add, purchase, remove}
  product_sku_hash = id_product
else :
  product_action = empty
  product_sku_hash = empty
server_timestamp_epoch_ms = temps action
hashed_url = url of the page

# Search Events
session_id_hash = id_session
server_timestamp_epoch_ms
query_vector = vector de recherche
product_skus_hash = products in the search response
clicked_skus_hash = products clicked after issuing the search query

# Sku_to_content
product_sku_hash = id_product
category_hash = category_product
price_bucket = 	The product price, provided as a 10-quantile integer
description_vector = description of the product
image_vector = image of the product

# Features importants
add_has_been_detailed
nb_detail_after
add_relative_price
add_price
mean_sim_desc_before 





# Other Features 
# Session
  session_lenght  : calculer le temps de la session 
  nb_unique_interactions : calculer la longueur du event_type
  nb_queries : calculer le nombre de query_vector
# Add-to-cart
  add_product_id : get the product_sku_hash when product_action == add_to_cart
  add_nb_interactions 
  add_has_been_detailed = get 
  add_has_been_removed
  add_has_been_viewed
  add_has_been_searched
  add_has_been_clicked
  add_category_hash
  add_main_category
  add_price
  add_relative_price
  add_relative_price_main
# Interaction statistics
  nb_add_before
  nb_add_after
  nb_detail_before
  nb_detail_after
  nb_remove_before
  nb_remove_after
  nb_view_before
  nb_view_after
  nb_click_before
  nb_click_after
# First and last Interactions features	
  product_url_id_list_after
  event_type_list_after
  product_action_list_after
  category_list_after
  price_list_after
  relative_price_list_after
  product_url_id_list_before
  event_type_list_before
  product_action_list_before
  category_list_before
  price_list_before
  add_category_hash
  add_main_category
  add_price
  add_relative_price
  add_relative_price_main
# Similarity features	
  mean_sim_desc 
  std_sim_desc
  mean_sim_img
  std_sim_img
  mean_sim_desc_before
  std_sim_desc_before
  mean_sim_img_before
  std_sim_img_before
  mean_sim_desc_after
  std_sim_desc_after
  mean_sim_img_after
  std_sim_img_after
  main_category_similarity_general
  main_category_similarity_add'
  





