# Deep spectral-based shape features for Alzheimer’s Disease classification

This repository contains the dataset and code used in our paper entitled "Deep spectral-based shape features for Alzheimer’s Disease classification"

# Manuscript abstract
Alzheimer’s disease (AD) and mild cognitive impairment (MCI) are the most prevalent neurodegenerative brain diseases in elderly population. Recent studies on medical imaging and biological data have shown morphological alterations of subcortical structures in patients with these pathologies. In this work, we take advantage of these structural deformations for classification purposes. First, triangulated surface meshes are extracted from segmented hippocampus structures in MRI and point-to-point correspondences are established among population of surfaces using a spectral matching method. Then, a deep learning variational auto-encoder is applied on the vertex coordinates of the mesh models to learn the low dimensional feature representation. A multi-layer perceptrons using softmax activation is trained simultaneously to classify Alzheimer’s patients from normal subjects. Experiments on ADNI dataset demonstrate the potential of the proposed method in classification of normal individuals from early MCI (EMCI), late MCI (LMCI), and AD subjects with classification rates outperforming standard SVM based approach.

# Dataset
The original data comes from the popular brain imaging dataset in Alzheimer’s disease, namely the Alzheimer’s Disease Neuroimaging Initiative (ADNI: adni.loni.usc.edu). For this study, a subset of latest 1.5 T MR images is used including 150 normal controls (NC), 90 AD patients, 160 early MCI (EMCI), and 160 individuals with late MCI (LMCI). Left (label number 17) and right (label number 53)hippocampi were segmented using FSL-FIRST automatic segmentation software package. Some subjects were removed because of the failure in the preprocessing steps. Therefore, in total, 142 normal controls (NC), 83 AD patients, 154 early MCI (EMCI), and 150 individuals with late MCI (LMCI) were included in our analysis.

The folder data includes 4 sub-folders (NC, AD, EMCI, LMCI). Each folder contains two sub-directories (17 and 53). All of the spectral meshes are saved here in vtk format. For instance, "data/NC/17/NC_1_17.vtk" is the left hippocampus mesh of healthy subject 1, while "data/NC/53/NC_1_53.vtk" is the right hippocampus mesh of the same subject!

# Code 
**InputData.py**: creates NC.csv, AD.csv, EMCI.csv, and LMCI.csv files. Each row in <X>.csv includes the list of all vertex coordinates of left and right hippocampus for one subject. 
  
The files **X.csv**formed our feature vectors and were directly fed to our classification framework. 

**variationalAE_MLP.py**: uses a deep variational autoencoder (**VAE**) to learn a latent feature representation from the low-level features and trains a multi-layer perceptron (**MLP**) for two class classification purpose.

# Citation
If you would like to use our **code** in your research, please the following paper.

```
@inproceedings{shakeri2016Spectral,
author="Shakeri, Mahsa and Lombaert, Herve and Tripathi, Shashank and Kadoury, Samuel",
editor="Reuter, Martin and Wachinger, Christian and Lombaert, Herv{\'e}",
title="Deep Spectral-Based Shape Features for Alzheimer's Disease Classification",
booktitle="Spectral and Shape Analysis in Medical Imaging",
year="2016",
publisher="Springer International Publishing",
pages="15--24",
isbn="978-3-319-51237-2"
}

```

If you find our **mesh dataset** useful for your research, please first cite the following paper and also please make sure to follow ADNI Data Sharing and Publication Policy available at [this link](https://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_DSP_Policy.pdf).

```
@inproceedings{shakeri2016Spectral,
author="Shakeri, Mahsa and Lombaert, Herve and Tripathi, Shashank and Kadoury, Samuel",
editor="Reuter, Martin and Wachinger, Christian and Lombaert, Herv{\'e}",
title="Deep Spectral-Based Shape Features for Alzheimer's Disease Classification",
booktitle="Spectral and Shape Analysis in Medical Imaging",
year="2016",
publisher="Springer International Publishing",
pages="15--24",
isbn="978-3-319-51237-2"
}

```


