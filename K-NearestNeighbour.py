import matplotlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import pandas as pd
import ProjectLibrary

transformed_data = [[20, 11.49], [21.5, 21.81], [16.5, 102.44], [18.5, 26.47], [9, 20.84], [19, 10.25], [18.5, 12.6],
                    [6.5, 11.92], [20, 14.45], [19.5, 28.39], [18, 31.53], [22.5, 12.51], [22.5, 0.93], [20.5, 17.06],
                    [19, 28.99], [19, 62.9], [18, 15.54], [18.5, 13.91], [6, 17.74], [19.5, 9.19], [18.5, 29.28],
                    [22.5, 11.1], [18.5, 31.81], [20, 32.43], [21.5, 13.43], [8.5, 19.2], [20, 38.07], [17, 17.71],
                    [22, 12.46]]
labels_ = [3, 2, 4, 2, 1, 3, 3, 1, 3, 2, 2, 3, 3, 3, 2, 0, 3, 3, 1, 3, 2, 3, 2, 2, 3, 1, 2, 3, 3]

peak_time = 23
energy = 50


# ProjectLibrary.plot_labeled_data(transformed_data,labels_)  #first run the code with line 21 , comment the line 41
# then comment the line 21 , run with line 41
# so , you can see the difference


predicion_of_hhold = ProjectLibrary.predict_class_of_new_household(transformed_data, labels_, peak_time, energy)

new_label_list, new_transformed_data_list = ProjectLibrary.append_new_item_to_lists(transformed_data, labels_,
                                                                                    predicion_of_hhold, peak_time,
                                                                                    energy)

df1 = pd.DataFrame(new_label_list,
                   columns=['label'])

df2 = pd.DataFrame(new_transformed_data_list,
                   columns=['peak_hour', 'average_energy'])

df_result = pd.concat([df2, df1], axis=1, join='inner')

merged_df = ProjectLibrary.df_merge_knn(df1, df2, labels_, transformed_data)

ProjectLibrary.knn_classification(merged_df)
