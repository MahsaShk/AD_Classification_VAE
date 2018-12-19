
"""
@author: Mahsa
"""
import os
import sys
import csv
import glob
import numpy as np

def meshFeatures(input_Dir):
    """
    Read a list of vtk files and save vertex coordinates of each mesh 
    in one row of the matrix data.
    
    input_Dir     : Directory, where the VTK meshes are saved
    data          : Output data array, where each row contains vertex 
                    coordinate of each mesh
    """
    filelist =glob.glob(input_Dir+"*.vtk") # list of the addresses of the vtk files 
    All_mesh_ver = [] # Each element includes list of vertex coordinates of one mesh 
           
    for i in range(0,len(filelist)):
        input_file = filelist[i] 
        print("current mesh: ", input_file)
        with open(input_file, 'r')  as f:  
            l=f.readline()
            l=f.readline()
            l=f.readline()
            l=f.readline()
            l=f.readline()

            # Extract number of vertex points 
            ind1=l.find('POINTS ')
            ind2=l.find('float')
            ind_start = ind1 +7
            ind_end = ind2
            pointnb = int (l[ind_start:ind_end])
            print("number of mesh vertices: ", pointnb)
            linenb = int(pointnb/3)

            # Current mesh vertices are saved as: x1 y1 z1 x2 y2 z2...x1002 y1002 z1002
            cur_mesh_ver = []  
            for ln in range(0,linenb):
                l = f.readline()
                temp = [float(x) for x in l.split()]
                cur_mesh_ver = cur_mesh_ver + temp
            All_mesh_ver.append(cur_mesh_ver)

    data = np.array(All_mesh_ver)
    return data
if __name__ == '__main__':
    # Read Normal (NC), left Hippocampus (17)
    input_Dir = './data/NC/17/'    
    data_NC_17 = meshFeatures(input_Dir)   

    # Read Normal (NC), left Hippocampus (53)
    input_Dir = './data/NC/53/'    
    data_NC_53 = meshFeatures(input_Dir)  

    # Concatenate data_NC_17 with data_NC_53
    data_NC = np.hstack((data_NC_17, data_NC_53))
    np.savetxt("./data/NC.csv", data_NC, delimiter=",")

    #----------------------------------------
    # Read AD, left Hippocampus (17)
    input_Dir = './data/AD/17/'    
    data_AD_17 = meshFeatures(input_Dir)   

    # Read AD, left Hippocampus (53)
    input_Dir = './data/AD/53/'    
    data_AD_53 = meshFeatures(input_Dir)  

    # Concatenate data_AD_17 with data_AD_53
    data_AD = np.hstack((data_AD_17, data_AD_53))
    np.savetxt("./data/AD.csv", data_AD, delimiter=",")

    #----------------------------------------
    # Read EMCI, left Hippocampus (17)
    input_Dir = './data/EMCI/17/'    
    data_EMCI_17 = meshFeatures(input_Dir)   

    # Read EMCI, left Hippocampus (53)
    input_Dir = './data/EMCI/53/'    
    data_EMCI_53 = meshFeatures(input_Dir)  

    # Concatenate data_EMCI_17 with data_EMCI_53
    data_EMCI = np.hstack((data_EMCI_17, data_EMCI_53))
    np.savetxt("./data/EMCI.csv", data_EMCI, delimiter=",")

    #----------------------------------------
    # Read LMCI, left Hippocampus (17)
    input_Dir = './data/LMCI/17/'    
    data_LMCI_17 = meshFeatures(input_Dir)   

    # Read LMCI, left Hippocampus (53)
    input_Dir = './data/LMCI/53/'    
    data_LMCI_53 = meshFeatures(input_Dir)  

    # Concatenate data_LMCI_17 with data_LMCI_53
    data_LMCI = np.hstack((data_LMCI_17, data_LMCI_53))
    np.savetxt("./data/LMCI.csv", data_LMCI, delimiter=",")
