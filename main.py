import glob
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from mahotas.features import zernike_moments
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from skimage.transform import rotate

path_to_files = "data/"
    
def filter_data_and_create_database():
    x_name = "162 particles with 133 orientations and 7220 intensity values for each orientation.npy"
    y_name = "sizes of the 162 filtered particles.npy"

    if os.path.isfile(path_to_files + x_name) and os.path.isfile(path_to_files + y_name):
        return np.load(path_to_files + x_name), np.load(path_to_files + y_name)
        
    print("filtering data and saving to disk...")

    if os.path.isfile(path_to_files + x_name) is False:
        result = np.empty((162 * 133, 7220), dtype=np.float32)

        elevation = h5py.File(path_to_files + 'l1.1_d9.1_flat.h5', 'r')['coordinates']['elevation'][()] 
        index = ((elevation <= 25) & (elevation >= 6))[0]

        idx = 0

        for f in glob.glob(path_to_files + '*.h5'):
            with h5py.File(f, 'r') as f:
                for i in range(133):
                    # were explored and the one (log-transformation) that gives the best test performances 
                    # (for the machine-learning classifiers and estimators, MLCE) was retained
                    result[(idx*133)+i] = np.log(f['intensity'][i][index])
                    
                idx += 1
            
        np.save(path_to_files + x_name, result)
        
    if os.path.isfile(path_to_files + y_name) is False:
        result = np.empty((162 * 133,), dtype=np.float32)
        
        idx = 0
        
        for f in glob.glob(path_to_files + '*.h5'):
            with h5py.File(f, 'r') as f:
                result[(idx*133) : (idx + 1)*133] = f['size'][()].reshape((133,))
                
                idx += 1
            
        np.save(path_to_files + y_name, result)
    
    return np.load(path_to_files + x_name), np.load(path_to_files + y_name)
    
def calculate_zernike_moments(data, type, rotate_image = False, zernike_degree = 20):
    zernike_result_filename = path_to_files + "zernike_%s_with_degree_%d.npy" % (type, zernike_degree)

    if os.path.isfile(zernike_result_filename) is False:
        print("calculating %d zernike polynomials with degree %d for %s" % (data.shape[0], zernike_degree, type))
        
        zernike_size = zernike_moments(data[0].reshape(20, 361), 180, zernike_degree).size
        zernike_results = np.empty((data.shape[0], zernike_size), dtype=np.float32)
            
        for i in range(data.shape[0]):
            image = data[i].reshape(20, 361)
            
            if rotate_image:
                image = rotate(image, np.random.randint(1, 360), True)
        
            zernike_results[i] = zernike_moments(image, 180, zernike_degree)
            
        np.save(zernike_result_filename, zernike_results)
        
    return np.load(zernike_result_filename)
    
x, y = filter_data_and_create_database()

#histogram
hist, bins = np.histogram(x, bins='auto')
plt.plot(bins[:hist.size], hist / np.sum(hist))
plt.xlabel('Intensity')
plt.ylabel('Proportion')
plt.savefig('histogram.png')
plt.clf()

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 1, test_size = 0.3)

X_train = calculate_zernike_moments(X_train, 'train')
X_test = calculate_zernike_moments(X_test, 'test', True)

regr = MLPRegressor(hidden_layer_sizes = (90, 60, 30), 
                    activation = 'relu', 
                    solver = 'sgd', 
                    learning_rate = 'adaptive', 
                    max_iter=10000, 
                    random_state = 1).fit(X_train, y_train)

print("R² = %f" % regr.score(X_test, y_test))

plt.plot(regr.predict(X_test), y_test, '+')
plt.plot(y_test, y_test, '-')
plt.savefig('result.png')
