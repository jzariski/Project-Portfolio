import os
import numpy as np
import sys
import time
import random
from pathlib import Path
import scipy.ndimage 
#import matplotlib.pyplot as plt
import statistics as s
#import matplotlib.colors as colors
import numpy.linalg as lg
import h5py
sys.path.append('../shocktubecalc')

xdiskPath = "/xdisk/kkratter/jzariski/sodstuff/"
parPath = "/home/u5/jzariski/Sod_Stuff/simple_shock_tube_calculator/shocktubecalc/comparisons/setups/sod1d/sod1d.par"
parPath2 = "/home/u5/jzariski/Sod_Stuff/simple_shock_tube_calculator/shocktubecalc/comparisons/setups/sod1d/sod1d2.par"
outputPath = "/home/u5/jzariski/Sod_Stuff/simple_shock_tube_calculator/shocktubecalc/comparisons/sodoutput.out"
jobPath = "/home/u5/jzariski/Sod_Stuff/simple_shock_tube_calculator/shocktubecalc/comparisons/job.sh"
analytic = []
        
        
def runJob():
    os.system('sbatch job.sh')


def changePar(nz):
    og = open(parPath,'r')
    out = open(parPath2,'w')
    done1 = True
    for line in og:
        if line[0]=='N' and line[1]=='z' and done1:
            out.write('Nz                      '+str(nz)+'\n')
            done1 = False
        else:
            out.write(line)
        
    out.close()
    og.close()
    os.system("cp " + parPath2 + " " + parPath)
    os.system("rm " + parPath2)
        
        
        
def generateData(nzmin, nzmax, noise):
    #finalOutput = open('Outputdata.txt', 'w')
    final = []
    for nz in range(nzmin, nzmax):
        print('CURR NZ', nz)
        curr = np.asarray(getCalculated(nz, noise))
        for i in range(curr.shape[0]):
            final.append(curr[i,:])
    
    training_inputs, training_labels = [], []
    dev_inputs, dev_labels = [], []
    test_inputs, test_labels = [], []
    final = np.asarray(final)
   
    #print(final.shape)
    #final = final[0,:,:]
    np.random.shuffle(final)
    inputs = final.shape[0]
    print(final)
    print(final.shape)
    
    
    sizetraining = int(np.ceil(inputs * 0.7))
    sizedev = int(np.ceil(inputs * 0.2))
    sizetest = inputs - sizetraining - sizedev
    
    train_set = final[0:sizetraining, :]
    dev_set = final[sizetraining+1:sizetraining+sizedev+1,:]
    test_set = final[sizetraining+sizedev+2:final.shape[0],:]
    
    training_inputs_arr = np.asarray(train_set[:,1:])
    training_labels_arr = np.asarray(train_set[:,0])
    dev_inputs_arr = np.asarray(dev_set[:,1:])
    dev_labels_arr = np.asarray(dev_set[:,0])
    test_inputs_arr = np.asarray(test_set[:,1:])
    test_labels_arr = np.asarray(test_set[:,0])
    
    
    return training_inputs_arr, training_labels_arr, dev_inputs_arr, dev_labels_arr, test_inputs_arr, test_labels_arr
    
def makeFile(d):
    # create HDF5 file
    with h5py.File('datasetmid.hdf5', 'w') as hf:
        dset_inputs_train = hf.create_dataset('inputs_train', data=d[0], shape=d[0].shape, compression='gzip', chunks=True)
        dset_outputs_train = hf.create_dataset('labels_train', data=d[1], shape=d[1].shape, compression='gzip', chunks=True)
        dset_inputs_dev = hf.create_dataset('inputs_dev', data=d[2], shape=d[2].shape, compression='gzip', chunks=True)
        dset_outputs_dev = hf.create_dataset('labels_dev', data=d[3], shape=d[3].shape, compression='gzip', chunks=True)
        dset_inputs_test = hf.create_dataset('inputs_test', data=d[4], shape=d[4].shape, compression='gzip', chunks=True)
        dset_outputs_test = hf.create_dataset('labels_test', data=d[5], shape=d[5].shape, compression='gzip', chunks=True)
        
        
        
        

            
def getCalculated(nz, noise):
    changePar(nz)
    runJob()
    filepath = Path(xdiskPath+'gasdens4999.dat')
    while not filepath.is_file():
        time.sleep(5)
        filepath = Path(xdiskPath+'gasdens4999.dat')
        print('waiting')
        
    final = []
    for i in range(0,5000):
        gas_vals = []
        gas_vals.append(i * 0.00005)
        arr = np.fromfile(xdiskPath+"gasdens"+str(i)+".dat")
        newArr = scipy.ndimage.zoom(arr,500/len(arr))
        for j in newArr:
            noiseNum = 1
            if noise:
                noiseNum = random.uniform(0.9,1.1)
            gas_vals.append(noiseNum * j)
        os.system('rm ' +xdiskPath+"gasdens"+str(i)+".dat")
        final.append(gas_vals)
    return final
        
        
makeFile(generateData(250,255, False))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
