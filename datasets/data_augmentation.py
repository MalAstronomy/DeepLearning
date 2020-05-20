

##############################################################
#                 DEEP LEARNING ASSIGMENT
#
#           Title: Predicting Merger Properties
#
#                - Data augmentation code -
#
#                        Authors:
#                     Carles Cantero 
#                     Malavika Vasist
#                     Maxime Quesnel
#
##############################################################


from __future__ import print_function, division
import os
import torch
import csv
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image 
import pandas as pd
import h5py
import natsort

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



def data_augmentation(path_read_cubes= './density_train_val/', path_read_labels='./labels_qtscaled/', path_save='./density_transformed/',
                      name_to_save = 'merger_train_val_cubes_with28.h5',n_rotations = 8, min_angle=1, max_angle=360, 
                      pad_original=True, flip_mode = 'full', axis_flip = None, verbose = True):
    
    '''
    Method to prepare the training dataset in a single h5py file applying some data augmentation 
    techniques into the galaxy merger density cubes. All epochs with their correponding sample of 
    3D cubes in each one are iterated and loaded with their corresponding label. 
    
    During the iteration, each of these cubes can be transformed according to the following techniques:
        1) Random rotation around one random axis (euclidean) + pad the rotated cube into (70,70,70)
        2) Pad the original cube (without any transformation) to (70,70,70)
        3) Flip the entire cube, or just along one axis.
    
    Each of these transformations can be applied independently or all together in a cube. 
    - To activate 1) introduce a value in the "n_rotations" parameter. 
    - To activate 2) keep as "True" the "pad_original" parameter. 
    - To activate 3) introduce a mode in the "flip_mode" parameter.
    
    PARAMETERS
    · path_read_cubes: (string) Path of the folder where the epochs (cubes) are located
    · path_read_labels: (string) Path of the folder where the epochs (labels) are located
    · path_save: (string) Path to save the h5py file
    · name_to_save: (string) Name of the h5py file (with the extension .h5)
    · n_rotations: (integer) Number of random rotations between [min_angle, max_angle] to be applied in one cube
    · min_angle: (integer) Minimum value in the angle range (degrees) to be choosen randomly. By default is 1
    · max_angle: (integer) Maximum value in the angle range (degrees) to be choosen randomly. By default is 360
    · pad_original: (boolean) Pad and save also the original cube without rotations 
    · flip_mode: (string) Flipping operation modes:
            - "full": the entire 3D cube is flipped
            - "axis": the cube is flipped along a choosen axis. The parameter "axis_flip" is needed in 
                      this mode. 
            - "random_axis": the cube is flipped along an axis choosen randomly.
            - "random_full": the cube is flipped randomly between two options: a random axis or the entire
                             cube.
    · axis_flip: (integer) Flipping axis: 0 (x-axis), 1 (y-axis), 2 (z-axis)
    · verbose: (boolean) Show prints while running
    '''

    # Create the h5py file with two groups (data and labels)
    hdf = h5py.File(str(path_save) + str(name_to_save), 'w')
    group_data = hdf.create_group('Density') 
    group_labels = hdf.create_group('Ratio') 

    # Iterate over epoch files
    counter_files = 0
    for epoch in os.listdir(path_read_cubes):
        
        if verbose == True:        
            if n_rotations: 
                name = str(n_rotations) + ' Random rotations'
            if flip_mode:
                if n_rotations == None: 
                    name = ' Flipping'
                else:  
                    name = name + ' + Flipping'
            if pad_original: 
                if flip_mode == None: 
                    name = ' Pad original cube'
                else:  
                    name = name + ' + Pad original cube'
            print('')
            print('-----------------------------------------------------------------------------------------------', flush=True)
            print('EPOCH = ' + str(epoch) + ' (Transformations in each cube = ' + str(name) + ')', flush=True)
            print('-----------------------------------------------------------------------------------------------', flush=True)

        
        # Read cubes and labels + sorting by name
        eagle_data_dict = load_density_mergers(path_dir = str(path_read_cubes)+str(epoch)+'/')
        eagle_data_dict = dict(natsort.natsorted(eagle_data_dict.items()))
        labels = load_labels(path_dir=str(path_read_labels)+ str(epoch) + '/')
        
        # Iterate over cube files in each epoch
        counter_label = 0
        for item in eagle_data_dict.items():
            density_index = item[0][:-4]
            label  = labels[int(density_index)]

            if verbose == True:
                print('')
                print('Cube = ' + density_index + '  shape = ' + str(item[1].shape), flush=True)

                
            # ROTATIONS + PADDING + FLIPPING LOOP
            if n_rotations:
                # Iterate over the number of rotations in each cube
                for k in range(n_rotations):

                    # Random angle and axis + rotation + padding + flipping
                    angl, ax = random_angle_axis(min_angle,max_angle)
                    image = rotate_cube(matrix=item[1], deg_angle = angl, axis= ax)
                    image = Pad_cube(image, axis = ax)
                    if flip_mode:
                        image = flip_cube(image, mode = flip_mode, axis = axis_flip)

                    if image.shape != (70,70,70):
                        if verbose == True:
                            print('Uncorrect shape. This file is not saved', flush=True)
                        continue

                    # Saving in the h5py file
                    group_data.create_dataset(str(counter_files), data = image) 
                    group_labels.create_dataset(str(counter_files), data = label) 

                    if verbose == True:
                        if flip_mode:
                            print(' > Filename = ' + str(counter_files) + ' New shape = ' + str(image.shape) + ' Labelname = ' + str(counter_files) + ': MR/SR =' + str(label), flush=True)
                        else:
                            print(' > Filename = ' + str(counter_files) + '  New shape = ' + str(image.shape) + ' Labelname = ' + str(counter_files) + ': MR/SR =' + str(label), flush=True)

                    counter_files = counter_files + 1
            
            # JUST FLIP
            if flip_mode and n_rotations == None:
                image = flip_cube(item[1], mode = flip_mode, axis = axis_flip)
                if verbose == True:
                    print(' > Filename = ' + str(counter_files) + '  New shape = ' +  str(image.shape) + ' Labelname = ' + str(counter_files) + ': MR/SR = ' + str(label), flush=True)
                group_data.create_dataset(str(counter_files), data = image) 
                group_labels.create_dataset(str(counter_files), data = label)
                counter_files = counter_files + 1
                

            # JUST PAD THE ORIGINAL CUBE
            if pad_original == True:
                original_padded = np.pad(item[1], ((10,10), (10,10), (10,10)), mode = 'edge')
                if verbose == True:
                    print(' > Filename = ' + str(counter_files) +  '(original)   New shape = ' + str(original_padded.shape) + ' Labelname = ' + str(counter_files) + ': MR/SR =' + str(label), flush=True)
                group_data.create_dataset(str(counter_files), data = original_padded)
                group_labels.create_dataset(str(counter_files), data = label) 

                counter_files = counter_files + 1

                
            counter_label = counter_label + 1
    print('h5py file saved correctly', flush=True)
    hdf.close()






def plot_images_grid(list_images, n_display = 5, columns=2, size=(10,10)):
    
    '''
    Plot a grid of images from a list.
    
    PARAMETERS
    · images: list of images 
    · n_display: integer regarding number of images to display
    · columns: integer regarding numbers of columns in the grid
    · figsize: tuple with grid shape
    '''
    
    fig = plt.figure(figsize=size)
    column = 0
    for i in range(n_display):
        column += 1
        #  check for end of column and create a new figure
        if column == columns+1:
            fig = plt.figure(figsize=size)
            column = 1
        fig.add_subplot(1,columns, column)
        plt.imshow(list_images[i])
        plt.axis('off')
        
        
            
def plot_image(image, figsize = (7,7), alpha = 1):
    
    '''
    Plot a single image.
    
    PARAMETERS
    · image: image array 
    · figsize: tuple with the image shape 
    · alpha: integer (0<alpha<1) regarding the transparency of the image
    '''
    
    fig = plt.figure(figsize=figsize)
    plt.imshow(image, alpha = alpha)
    plt.show()
    
    
    
def load_image(path_dir, filename):
    
    '''
    Load a single image with formats csv, png, or jpg.
    
    PARAMETERS
    · path_dir: Folder path of the image
    · filename: name of the file
    
    RETURN
    Array of the image loaded
    '''
    name, ext = os.path.splitext(path_dir + filename)
    print(name)
    formats = ['.csv', '.png', '.jpg', '.jpeg']
    if ext == formats[0]:
        image = pd.read_csv(path_dir + filename)
    elif ext == formats[1] or ext == formats[2] or ext == formats[3]:
        image = Image.open(path_dir + filename)
    return image



def load_images_set(path_dir):
    
    '''
    Load a set of images with formats png or jpg.
    
    PARAMETERS
    · path_dir: Folder path of the image set
    
    RETURN
    List of arrays (List of images in the set)
    '''
    
    formats = ['.png', '.jpg', '.jpeg']
    images = []
    for file in os.listdir(path_dir):
        if os.path.splitext(file)[1] in formats:
            image = load_image(path_dir= path_dir, filename=file)
            images.append(image)
        else:
            pass
    return images




def rotate_cube(matrix, deg_angle, axis):
    
    '''
    Rotate the 3D reference system given an angle(º) and the axis. The rotation matrices
    for each axis are computed and then applied into a 3D object. Therefore, the values 
    of this object will be adapted according to this new rotated reference system.
    
    PARAMETERS
    . matrix: 3D array (object) where apply the rotation
    · deg_angle: angle of rotation in degrees
    · axis: the axis of rotation (x, y or z)
    
    RETURN
    The new rotated 3D array (object)
    '''
    
    d = len(matrix)
    h = len(matrix[0])
    w = len(matrix[0][0])
    min_new_x = 0
    max_new_x = 0
    min_new_y = 0
    max_new_y = 0
    min_new_z = 0
    max_new_z = 0
    new_coords = []
    angle = np.radians(deg_angle)

    for z in range(d):
        for y in range(h):
            for x in range(w):

                new_x = None
                new_y = None
                new_z = None

                if axis == "x":
                    new_x = int(round(x))
                    new_y = int(round(y*np.cos(angle) - z*np.sin(angle)))
                    new_z = int(round(y*np.sin(angle) + z*np.cos(angle)))
                elif axis == "y":
                    new_x = int(round(z* np.sin(angle) + x*np.cos(angle)))
                    new_y = int(round(y))
                    new_z = int(round(z*np.cos(angle) - x*np.sin(angle)))
                elif axis == "z":
                    new_x = int(round(x*np.cos(angle) - y*np.sin(angle)))
                    new_y = int(round(x*np.sin(angle) + y*np.cos(angle)))
                    new_z = int(round(z))

                val = matrix.item((z, y, x))
                new_coords.append((val, new_x, new_y, new_z))
                if new_x < min_new_x: min_new_x = new_x
                if new_x > max_new_x: max_new_x = new_x
                if new_y < min_new_y: min_new_y = new_y
                if new_y > max_new_y: max_new_y = new_y
                if new_z < min_new_z: min_new_z = new_z
                if new_z > max_new_z: max_new_z = new_z

    new_x_offset = abs(min_new_x)
    new_y_offset = abs(min_new_y)
    new_z_offset = abs(min_new_z)

    new_width = abs(min_new_x - max_new_x)
    new_height = abs(min_new_y - max_new_y)
    new_depth = abs(min_new_z - max_new_z)

    rotated = np.empty((new_depth + 1, new_height + 1, new_width + 1))
    rotated.fill(0)
    for coord in new_coords:
        val = coord[0]
        x = coord[1]
        y = coord[2]
        z = coord[3]

        if rotated[new_z_offset + z][new_y_offset + y][new_x_offset + x] == 0:
            rotated[new_z_offset + z][new_y_offset + y][new_x_offset + x] = val

    matrix = rotated
    return matrix

  


def plot_3D_by_components(image, figsize = (15,10)):
    
    '''
    Plot a grid of all the axis planes (x-y, x-z, y-z) of a 3D object.
    
    PARAMETERS
    · image: 3D array (object)
    · figsize: tuple regarding the grid shape
    '''
    
    for i in range(len(image)):
        plt.figure(figsize=figsize)
        plt.subplot(1,3,1)
        plt.imshow(image[:,:,i])
        plt.xlabel('x-axis', fontsize= 14)
        plt.ylabel('y-axis',fontsize= 14)
        plt.text(x = 5, y = 5, s ='frame = ' + str(i), c = 'w' )
        plt.subplot(1,3,2)
        plt.imshow(image[:,i,:])
        plt.xlabel('x-axis',fontsize= 14)
        plt.ylabel('z-axis',fontsize= 14)
        plt.text(x = 5, y = 5, s ='frame = ' + str(i), c = 'w' )
        plt.subplot(1,3,3)
        plt.imshow(image[i,:,:])
        plt.xlabel('y-axis',fontsize= 14)
        plt.ylabel('z-axis',fontsize= 14)
        plt.text(x = 5, y = 5, s ='frame = ' + str(i), c = 'w' )
        plt.show()
        
  

        
        
def plot_3D_contourMap(array, contour = 5, cm = 'Blues'):
    
    '''
    Plot the 3D countours of an object (3D array) by means of the mayavi library.
    
    PARAMETERS
    · array: 3D array (object)
    · contour: integer regarding the number of contours when plotting the 3D object.
               The highest the number, the more details in the plot.
    · cm: string with the color style of that contours
    '''
    
    coordinates = load_image(path_dir='./redshifts/12/', filename='3.csv')
    xmin, ymin, zmin = coordinates.min()
    xmax, ymax, zmax = coordinates.max()

    d = []
    for i in range(3):
        a = array1.shape[i]*1j
        d.append(a)
        
    x, y, z = np.mgrid[xmin:xmax:d[0], ymin:ymax:d[1], zmin:zmax:d[2]]
#     mlab.contour3d(x,y,z, array1, opacity = 0.5, contours = contour,colormap = cm)
#     #mlab.axes()
#     mlab.show()

    
      
    
    
def load_density_mergers(path_dir):
    
    '''
    Specific method to load csv files generated from EAGLE
    simulations where they correspond to the density scatter plots 
    for each merger and for each epoch.
    
    PARAMETERS
    · path_dir: Folder path of an epoch where the merger csv files are located
    
    RETURNS
    Dictionary with all merger csv files for a specific epoch
    '''
    
    density_dict = {}
    for j in os.listdir(path_dir):#+str(r)): 
        den=[]
        with open(path_dir+str(j)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:        
                if line_count == 0:
                    #print(f'Column names are {", ".join(row)}')
                    pass
                else: 
                    den= np.append(den,row)   #list of strings
                line_count += 1  
        Den=list(map(float,den))              #list of float values
        density= np.reshape(Den,(50,50,50))

        density_dict[j] = density
    #print(density_dict.keys())
    return density_dict





def padding_by_axis(array, axis):
    
    '''
    Specific method used in Pad_3D_object() to find the padding coefficients for
    a rotated 2D square image. These coeficients will basically change depending 
    on the angle of rotation the 2D image did.
    
    PARAMETERS
    · array: 2D array (2D already rotated image of a given plane of the 3D object, 
             for example x-y plane while keeping a constant value of z).
    · axis: string regarding the axis of rotation of the 2D image 
    
    RETURN
    Two tuples regarding the padding coefficients, one tuple for each axis in the plane
    of the 2D image.
    '''
    
    maximum_padding = 70
    
    if axis == 'x': idx = [0,1]
    elif axis == 'y': idx = [0,2]
    elif axis == 'z': idx = [1,2]

    x_size = array.shape[idx[0]]
    y_size = array.shape[idx[1]]
    
    #print('x-size = ' + str(x_size))
    #print('y-size = ' + str(y_size))
    
    # Same initial shapes
    if x_size == y_size:
        size = x_size
        diff = maximum_padding - size
      
        if (size % 2) == 0:
            x_new = int(diff/2)
            y_new = int(diff/2)
        else:
            x_new = int(diff/2)
            y_new = int(diff/2) + 1
        
        if y_new - x_new == 1:
            x_new_shape = (x_new,y_new)
            y_new_shape = (x_new,y_new)
        else:
            x_new_shape = (x_new,x_new)
            y_new_shape = (y_new,y_new)
            
            
    # Different initial shapes        
    else:
        x_diff = maximum_padding - x_size
        y_diff = maximum_padding - y_size
        
        # X-axis
        if (x_size % 2) == 0: 
            x_new = int(x_diff/2)
            x_new_shape = (x_new,x_new)
        else:
            x_new_1 = int(x_diff/2)
            x_new_2 = int(x_diff/2) + 1
            x_new_shape = (x_new_1,x_new_2)
         
        # Y-axis
        if (y_size % 2) == 0: 
            y_new = int(y_diff/2)
            y_new_shape = (y_new,y_new)
        else:
            y_new_1 = int(y_diff/2)
            y_new_2 = int(y_diff/2) + 1
            y_new_shape = (y_new_1,y_new_2)
            
    return x_new_shape, y_new_shape




def Pad_cube(array, axis):
    
    '''
    Method used to pad a 2D square arrays (square images) from 3D objects 
    that have been rotated around a given axis. 
    
    When a reference system (RS) is rotated around a fixed square image (using the 
    method called rotate()), the shape of that image in the new RS can increase in 
    those axis depending on the angle. For angles of 90, 180 or 270 degrees,this 
    shape remains constant, but for intermediate angles it increases until a maximum. 
    
    If the goal is to rotate a 3D object (it is composed by several 2D images along 
    each axis) with random angles, padding is necessary after rotating if we want 
    to remain the same shape on each image plane in the new 3D object.
    
    Note**: As an alternative way, instead of padding, one could crop the new images in 
    order to obtain the original shape. 
    
    PARAMETERS
    · array: 2D array (2D already rotated image of a given plane of the 3D object, 
             for example x-y plane while keeping a constant value of z).
    · axis: string regarding the axis of rotation of the 2D image 
    
    RETURN
    The padded image 
    
    '''
    
    if axis == 'x':
        x_new_shape, y_new_shape = padding_by_axis(array, axis = 'x')       
        array_padded = np.pad(array, (x_new_shape, y_new_shape,(10,10)), mode = 'edge')
        
    elif axis == 'y':
        x_new_shape, y_new_shape = padding_by_axis(array, axis = 'y')       
        array_padded = np.pad(array, (x_new_shape, (10,10), y_new_shape), mode = 'edge')
        
    elif axis == 'z':
        x_new_shape, y_new_shape = padding_by_axis(array, axis = 'z')       
        array_padded = np.pad(array, ((10,10), x_new_shape, y_new_shape), mode = 'edge')
        
    else:
        print('Error: axis must be "x", "y" or "z".')

    return array_padded



def random_angle_axis(min_angle, max_angle):
    
    '''
    Choose randomly one angle (degres) and one axis of rotation
    
    PARAMETERs
    · min_angle: integer with the minimum value from which the angle 
                 will be generated randomly
    · max_angle: integer with the maximum value from which the angle 
                 will be generated randomly
                 
    RETURN
    Angle and axis randomly choosen
    '''
    
    # Angles
    angles = np.arange(min_angle, max_angle)
    angles_rand = int(np.random.randint(min_angle,max_angle-1,1))
    angle = angles[angles_rand]

    # Axis
    axis_range   = ['x','y', 'z']
    axis_rand = int(np.random.randint(0,3,1))
    axis = axis_range[axis_rand]
    
    return angle, axis

def load_labels(path_dir):
    with open(str(path_dir) + 'labels.csv') as csv_file:
        csv_reader = list(csv.reader(csv_file, delimiter=','))
        MR=np.asarray(csv_reader[1])
        SR=np.asarray(csv_reader[3])
        MR=MR.astype(np.float)
        SR=SR.astype(np.float)
        
        labels = np.vstack([MR,SR])
        labels = labels.T
        return labels   
    
    
    
    
def flip_cube(cube, mode = 'full', axis = None):
    
    '''
    Flip a 3D cube. It has 4 different operation modes.
    
    PARAMETERS
    · cube: (3d array) the cube to be flipped
    · mode: (string) operation modes:
            - "full": the entire cube is flipped
            - "axis": the cube is flipped along a choosen axis. The parameter axis is needed in this mode. 
                      The user must specify the axis with an integer:  0 (x-axis), 1 (y-axis), 2 (z-axis)
            - "random_axis": the cube is flipped along a random axis.
            - "random_full": the cube is flipped randomly between two options: a random axis or the entire
                             cube.
                            
    RETURN
    The flipped cube (3d array)
    '''
    
    # The entire cube
    if mode == 'full':
        flipped_cube = np.flip(cube)
        
    # Just one axis
    elif mode == 'axis':
        flipped_cube = np.flip(cube, axis = axis)
        
    # Random axis
    elif mode == 'random_axis':
        ax = int(np.random.randint(0,3,1))
        flipped_cube = np.flip(cube,axis = ax)
   
    # Random axis or full
    elif mode =='random_full':
        ax = int(np.random.randint(0,4,1))
        if ax == 3:
            flipped_cube = np.flip(cube)
        else:
            flipped_cube = np.flip(cube,axis = ax)

    else:
        print('Error: The mode is wrong. I can be "full", "axis", "random_axis" or "random_full"')
        
    return flipped_cube    

if __name__ == '__main__':
    data_augmentation()
    
    
    

