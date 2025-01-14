# Endmembers estimation of hyperspectral images with ELM method
Estimation of endmembers of hyperspectral images by using the ELM method.

## Authors:
- Idriss ABDOULWAHAB  
- Amrin AKTER 

## Description:
This project aims to use the ELM method. Also known as the Eigenvalue Likelihood Maximization, this method is used to estimate endmembers of mixtured samples.
The method used is described in the paper by B. Luo et al. 2013.
- **Source:** B. Luo et al., "Empirical Automatic Estimation of the Number of Endmembers in Hyperspectral Images", IEEE Sensors Journal
- **DOI:** [10.1109/LGRS.2012.2189934](https://doi.org/10.1109/LGRS.2012.2189934)

## Project Structure:
- **empirical_method.ipynb**: Contains the notebook that computes the ELM method
- **elm_algorithm.py**: Contains the algorithm to compute endmembers for all dataset using the ELM method
- **results/** : Folder that contains .txt files with results of ELM method with different sensor dataset.


### Dataset:
- **Source:** B. Koirala et al., "A Multisensor Hyperspectral Benchmark Dataset for Unmixing of Intimate Mixtures", IEEE Sensors Journal.
- **DOI:** [10.1109/JSEN.2023.3343552](https://doi.org/10.1109/JSEN.2023.3343552)
- **Repository:** [VisionlabHyperspectral/Multisensor_datasets](https://github.com/VisionlabHyperspectral/Multisensor_datasets)
