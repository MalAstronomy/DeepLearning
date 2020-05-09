import numpy as np
from scipy import stats
#from mayavi import mlab

# %matplotlib inline
#import matplotlib.pyplot as plt
#import seaborn as sns; sns.set()

import csv 
import scipy.stats as stats
import os

for r in [28,27,23,19,15,14]: 
    for j in os.listdir('coordinates/csv_coordinates/'+str(r)): 
        x=[]
        y=[]
        z=[]
        if j.endswith('.csv'): 
            print(j)
            with open('coordinates/csv_coordinates/'+ str(r)+'/'+ j) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        print(f'Column names are {", ".join(row)}')
                    else: 
                        x= np.append(x,float(row[0]))
                        y= np.append(y,float(row[1]))
                        z= np.append(z,float(row[2]))

                    line_count += 1
        else: continue
        loc= np.transpose(np.matrix([x,y,z]))
        rot_loc = np.squeeze(np.asarray(loc))

        xyz = np.vstack([rot_loc[:,0],rot_loc[:,1],rot_loc[:,2]])
        kde = stats.gaussian_kde(xyz)

        data= xyz.T

        # Create a regular 3D grid with 50 points in each dimension
        xmin, ymin, zmin = data.min(axis=0)
        xmax, ymax, zmax = data.max(axis=0)
        xi, yi, zi = np.mgrid[xmin:xmax:100j, ymin:ymax:100j, zmin:zmax:100j]

        # Evaluate the KDE on a regular grid...
        coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
        density = kde(coords).reshape(xi.shape)

                              
        with open('density/'+str(r)+'/'+j, 'w') as csvfile:
            spamwriter = csv.writer(csvfile,quotechar=',', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(['density'])
            spamwriter.writerow(density.ravel())   #1000000 elements




