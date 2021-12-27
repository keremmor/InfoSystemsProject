import ProjectLibrary

peak_time = 6.80
energy = 85

path = "output-tables/dfAfterKNN/DataFrameKNN_block_11.csv"

df = ProjectLibrary.read_dataframe(path)

# predicion_of_hhold = ProjectLibrary.predict_class_of_new_household(df, peak_time, energy)

# df_new_hh_added = ProjectLibrary.append_new_item_to_lists(df,
#                                                           predicion_of_hhold, peak_time,
#                                                           energy)

# ProjectLibrary.write_out_the_given_dataframe(df_new_hh_added, path)

# print(df_new_hh_added)
print(df)
# ProjectLibrary.knn_classification(df_new_hh_added)
ProjectLibrary.knn_classification(df)
