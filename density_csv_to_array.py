#retreival

import os
import numpy as np
import csv

for r in [28,27,23,19,15,14]: 
    for j in os.listdir('density50/'+str(r)): 
        den=[]
        with open('density50/'+str(r)+'/'+j) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:        
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                else: 
                    den= np.append(den,row)   #list of strings
                line_count += 1  
        Den=list(map(float,den))              #list of float values
        density= np.reshape(Den,(50,50,50))

