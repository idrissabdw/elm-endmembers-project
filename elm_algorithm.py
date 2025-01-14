import glob
import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

# No spatial resolution
data_path = ['drive/MyDrive/Extensive_dataset/ASD_Spectroradiometer/ASD_Spectroradiometer_dataset.mat','drive/MyDrive/Extensive_dataset/PSR3500_spectral_evolution/PSR3500_spectral_evolution_dataset.mat','drive/MyDrive/Extensive_dataset/Agilent_4300_FTIR/Agilent_4300_FTIR_dataset.mat']
data_legend = ['ASD_Spectroradiometer','PSR3500_spectral_evolution','Agilent_4300_FTIR']

for i in range (len(data_path)):
    data = loadmat(data_path[i])
    data_matrix = data[data_legend[i]]
    N_s = len(data_matrix[:,1][0])
    N_a = data_matrix.shape[1]
    X = np.ones((N_s,N_a))

    for p in range (N_a):
        for j in range (N_s):
            X[j,p]=data_matrix[:,p][0][j][0]


    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1, ddof=1)
    X_centered = X - mean[:, None]
    X_normalized = X_centered / std[:, None]

    K = np.cov(X_centered, rowvar=True)
    R = np.corrcoef(X_normalized, rowvar=True)

    eig_vals_K, eig_vecs_K = np.linalg.eig(K)
    eig_vals_R, eig_vecs_R = np.linalg.eig(R)

    eig_R = eig_vals_R.real
    eig_K = eig_vals_K.real

    Z = eig_R-eig_K

    A = np.zeros(N_s)
    B = np.zeros(N_s)
    H = np.zeros(N_s)
    sigma = np.sqrt((2/N_a)*(eig_R**2 + eig_K**2))

    # Log-likelihood function
    for s in range (N_s):
        A[s] = -np.sum((Z[s:]**2)/(2*sigma[s:]))
        B[s] = -np.sum(np.log(np.sqrt(sigma[s:])))

    H_log = A+B

    N_c = np.argmax(H_log)-1

    V_endmembers_R = eig_vecs_R[:,:N_c]
    V_endmembers_R = V_endmembers_R.real

    V_endmembers_K = eig_vecs_K[:,:N_c]
    V_endmembers_K = V_endmembers_K.real

    S_R = X_normalized.T @ V_endmembers_R
    S_K = X_centered.T @ V_endmembers_K

    endmembers_index_R = np.argmax(np.abs(S_R), axis=0)+1
    endmembers_index_K = np.argmax(np.abs(S_K), axis=0)+1

    endmembers_index_final = np.unique(np.concatenate((endmembers_index_R, endmembers_index_K)))

    endmembers_correct = list(set(endmembers_index_final) & set(endmembers_index_gt))

    file_path = 'drive/MyDrive/Extensive_dataset/endmembers_no_spatial.txt'
    with open(file_path, 'a') as file:
        file.write(f"Nom de l'appareil : {data_legend[i]}\n")
        file.write(f"Nombre d'endmembers trouvés : {len(endmembers_index_final)}\n")
        file.write(f"Nombre total d'endmembers théoriques : {len(endmembers_index_gt)}\n")
        file.write(f"Nombre de bons endmembers trouvés : {len(endmembers_correct)}\n")
        file.write(f"Liste des bons endmembers trouvés : {endmembers_correct}\n")
        file.write(f"Liste des endmembers trouvés : {endmembers_index_final.tolist()}\n")
        file.write("\n")

    print(f"Résultats pour {data_legend[i]} sauvegardés dans {file_path}")


data_path = ['drive/MyDrive/Extensive_dataset/IMEC_hypespectral_camera/IMEC_dataset.mat',
             'drive/MyDrive/Extensive_dataset/Senops_HSC2/Senops_HSC2_dataset.mat',
             'drive/MyDrive/Extensive_dataset/Specim _AI(RGB)/Specim_JAI_RGB_dataset.mat',
             'drive/MyDrive/Extensive_dataset/Specim_AisaFenix/Specim_AisaFenix_dataset.mat',
             'drive/MyDrive/Extensive_dataset/Specim_AisaOwl/Specim_AisaOwl_dataset.mat',
             'drive/MyDrive/Extensive_dataset/Specim_FX50/Specim_FX50_dataset.mat',
             'drive/MyDrive/Extensive_dataset/Specim_sCMOS/Specim_sCMOS_dataset.mat',
             'drive/MyDrive/Extensive_dataset/Telops_HyperCam/Telops_HyperCam_dataset.mat',]

data_legend = ['IMEC_Image','Senops_HSC2_image','Specim_JAI_RGB_image','Specim_AisaFenix_image','Specim_AisaOwl_image',
               'Specim_FX50_image','Specim_sCMOS_image','Telops_HyperCam_image']

for i in range (len(data_path)):
    data = loadmat(data_path[i])

    data_matrix = data[data_legend[i]]
    N_s = data_matrix[0][0].shape[2] # Numbers of bands
    N_a = data_matrix[0].shape[0] # Numbers of samples
    N_x = data_matrix[0][0].shape[0]
    N_y = data_matrix[0][0].shape[1]
    X_data = np.ones((N_s,N_a*N_x*N_y))

    for p in range (N_a):
      for j in range(N_s):
          image = data_matrix[0][p]
          if image.size == 0:
              image = np.zeros((N_x,N_y,N_s))
          pixels = image[:,:,j]
          pixels = pixels.reshape(-1)
          X_data[j,p*(N_x*N_y):(p+1)*(N_x*N_y)] = pixels


    X_new = X_data
    N_a_new = X.shape[1]
    ## It's important to center the data in order to compute covariance and correlation
    mean = np.mean(X_new, axis=1)
    std = np.std(X_new, axis=1, ddof=1)
    X_centered = X_new - mean[:, None]
    X_normalized = X_centered / std[:, None]

    K = np.cov(X_centered, rowvar=True)
    R = np.corrcoef(X_normalized, rowvar=True)


    eig_vals_K, eig_vecs_K = np.linalg.eig(K)
    eig_vals_R, eig_vecs_R = np.linalg.eig(R)

    eig_R = eig_vals_R.real
    eig_K = eig_vals_K.real

    Z = eig_R-eig_K

    A = np.zeros(N_s)
    B = np.zeros(N_s)
    H = np.zeros(N_s)
    sigma = np.sqrt((2/N_a_new)*(eig_R**2 + eig_K**2))

    # Log-likelihood function
    for s in range (N_s):
        A[s] = -np.sum((Z[s:]**2)/(2*sigma[s:]))
        B[s] = -np.sum(np.log(np.sqrt(sigma[s:])))

    H_log = A+B

    N_c = np.argmax(H_log)-1

    V_endmembers_R = eig_vecs_R[:,:N_c]
    V_endmembers_R = V_endmembers_R.real

    V_endmembers_K = eig_vecs_K[:,:N_c]
    V_endmembers_K = V_endmembers_K.real

    S_R = X_normalized.T @ V_endmembers_R
    S_K = X_centered.T @ V_endmembers_K

    endmembers_index_R = np.argmax(np.abs(S_R), axis=0)//(N_x*N_y)+1
    endmembers_index_K = np.argmax(np.abs(S_K), axis=0)//(N_x*N_y)+1

    endmembers_index_final = np.unique(np.concatenate((endmembers_index_R, endmembers_index_K)))

    endmembers_correct = list(set(endmembers_index_final) & set(endmembers_index_gt))

    file_path = 'drive/MyDrive/Extensive_dataset/endmembers_spatial.txt'
    with open(file_path, 'a') as file:
        file.write(f"Nom de l'appareil : {data_legend[i]}\n")
        file.write(f"Nombre d'endmembers trouvés : {len(endmembers_index_final)}\n")
        file.write(f"Nombre total d'endmembers théoriques : {len(endmembers_index_gt)}\n")
        file.write(f"Nombre de bons endmembers trouvés : {len(endmembers_correct)}\n")
        file.write(f"Liste des bons endmembers trouvés : {endmembers_correct}\n")
        file.write(f"Liste des endmembers trouvés : {endmembers_index_final.tolist()}\n")
        file.write("\n")

    print(f"Résultats pour {data_legend[i]} sauvegardés dans {file_path}")

# Spatial resolution by mean
data_path = ['drive/MyDrive/Extensive_dataset/IMEC_hypespectral_camera/IMEC_dataset.mat',
             'drive/MyDrive/Extensive_dataset/Senops_HSC2/Senops_HSC2_dataset.mat',
             'drive/MyDrive/Extensive_dataset/Specim _AI(RGB)/Specim_JAI_RGB_dataset.mat',
             'drive/MyDrive/Extensive_dataset/Specim_AisaFenix/Specim_AisaFenix_dataset.mat',
             'drive/MyDrive/Extensive_dataset/Specim_AisaOwl/Specim_AisaOwl_dataset.mat',
             'drive/MyDrive/Extensive_dataset/Specim_FX50/Specim_FX50_dataset.mat',
             'drive/MyDrive/Extensive_dataset/Specim_sCMOS/Specim_sCMOS_dataset.mat',
             'drive/MyDrive/Extensive_dataset/Telops_HyperCam/Telops_HyperCam_dataset.mat',]

data_legend = ['IMEC_Image','Senops_HSC2_image','Specim_JAI_RGB_image','Specim_AisaFenix_image','Specim_AisaOwl_image',
               'Specim_FX50_image','Specim_sCMOS_image','Telops_HyperCam_image']

for i in range (len(data_path)):
    data = loadmat(data_path[i])
    data_matrix = data[data_legend[i]]
    N_s = data_matrix[0][0].shape[2] # Numbers of bands
    N_a = data_matrix[0].shape[0] # Numbers of samples
    N_x = data_matrix[0][0].shape[0]
    N_y = data_matrix[0][0].shape[1]
    X_data = np.ones((N_s,N_a*N_x*N_y))

    for p in range (N_a):
      for j in range(N_s):
          image = data_matrix[0][p]
          if image.size == 0:
              image = np.zeros((N_x,N_y,N_s))
          pixels = image[:,:,j]
          pixels = pixels.reshape(-1)
          mean_pixel = np.mean(pixels)
          X_data[j,p] = mean_pixel

    X = X_data
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1, ddof=1)
    X_centered = X - mean[:, None]
    X_normalized = X_centered / std[:, None]

    K = np.cov(X_centered, rowvar=True)
    R = np.corrcoef(X_normalized, rowvar=True)

    eig_vals_K, eig_vecs_K = np.linalg.eig(K)
    eig_vals_R, eig_vecs_R = np.linalg.eig(R)

    eig_R = eig_vals_R.real
    eig_K = eig_vals_K.real

    Z = eig_R-eig_K

    A = np.zeros(N_s)
    B = np.zeros(N_s)
    H = np.zeros(N_s)
    sigma = np.sqrt((2/N_a)*(eig_R**2 + eig_K**2))

    # Log-likelihood function
    for s in range (N_s):
        A[s] = -np.sum((Z[s:]**2)/(2*sigma[s:]))
        B[s] = -np.sum(np.log(np.sqrt(sigma[s:])))

    H_log = A+B

    N_c = np.argmax(H_log)-1

    V_endmembers_R = eig_vecs_R[:,:N_c]
    V_endmembers_R = V_endmembers_R.real

    V_endmembers_K = eig_vecs_K[:,:N_c]
    V_endmembers_K = V_endmembers_K.real

    S_R = X_normalized.T @ V_endmembers_R
    S_K = X_centered.T @ V_endmembers_K

    endmembers_index_R = np.argmax(np.abs(S_R), axis=0)+1
    endmembers_index_K = np.argmax(np.abs(S_K), axis=0)+1

    endmembers_index_final = np.unique(np.concatenate((endmembers_index_R, endmembers_index_K)))

    endmembers_correct = list(set(endmembers_index_final) & set(endmembers_index_gt))

    file_path = 'drive/MyDrive/Extensive_dataset/endmembers_spatial_mean.txt'
    with open(file_path, 'a') as file:
        file.write(f"Nom de l'appareil : {data_legend[i]}\n")
        file.write(f"Nombre d'endmembers trouvés : {len(endmembers_index_final)}\n")
        file.write(f"Nombre total d'endmembers théoriques : {len(endmembers_index_gt)}\n")
        file.write(f"Nombre de bons endmembers trouvés : {len(endmembers_correct)}\n")
        file.write(f"Liste des bons endmembers trouvés : {endmembers_correct}\n")
        file.write(f"Liste des endmembers trouvés : {endmembers_index_final.tolist()}\n")
        file.write("\n")

    print(f"Résultats pour {data_legend[i]} sauvegardés dans {file_path}")