#reading the labels

import numpy as np
import h5py
import csv

for i in [28]:#,27,23,19,15,14,12]: 
    with open('labels/'+ str(i)+'/labels.csv') as csv_file:
            csv_reader = list(csv.reader(csv_file, delimiter=','))
            MR=np.asarray(csv_reader[1])
            SR=np.asarray(csv_reader[3])
            MR=MR.astype(np.float)
            SR=SR.astype(np.float)
#             row_count= sum(1 for row_ in csv_reader)
