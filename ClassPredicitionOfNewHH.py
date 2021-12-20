import numpy as np
from sklearn import neighbors, datasets
import pandas as pd
from sklearn import preprocessing
import ProjectLibrary

transformed_data = [[20, 11.49], [21.5, 21.81], [16.5, 102.44], [18.5, 26.47], [9, 20.84], [19, 10.25], [18.5, 12.6],
                    [6.5, 11.92], [20, 14.45], [19.5, 28.39], [18, 31.53], [22.5, 12.51], [22.5, 0.93], [20.5, 17.06],
                    [19, 28.99], [19, 62.9], [18, 15.54], [18.5, 13.91], [6, 17.74], [19.5, 9.19], [18.5, 29.28],
                    [22.5, 11.1], [18.5, 31.81], [20, 32.43], [21.5, 13.43], [8.5, 19.2], [20, 38.07], [17, 17.71],
                    [22, 12.46]]
labels_ = [3, 2, 4, 2, 1, 3, 3, 1, 3, 2, 2, 3, 3, 3, 2, 0, 3, 3, 1, 3, 2, 3, 2, 2, 3, 1, 2, 3, 3]

time = 16
energy = 100

prediction_label = ProjectLibrary.predict_class_of_new_household(transformed_data, labels_, time, energy)
print(prediction_label)
