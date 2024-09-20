# Machine-learning-and-high-throughput-computationa-guided-development-of-high-temperature-oxidation-resisting-Ni-Co-Cr-Al-Fe-based-high-entropy alloys
This is the code for the paper 'Machine learning and high-throughput computational guided development of high temperature oxidation-resisting Ni-Co-Cr-Al-Fe based high-entropy alloys'. High-entropy alloys (HEAs) offers a significant opportunity to further enhance the oxidation resistance properties of bond coat materials, surpassing that of today's state of the art MCrAlY. However, the vast HEAs composition space limited the effectiveness of conventional trial-and-error experimental methods for alloy design. Here, we propose an effective design framework with the aid of machine learning, enabling the rapid design of high-temperature oxidation-resistant non-equiatomic HEAs. Our design framework includes three main steps: (i) development of an ML model that accurately predicts the nonlinear relationship between composition and oxidation rate; (ii) alloy candidates screening guided by ML insights, high throughput CALPHAD thermodynamic calculations, high throughput CTE calculation, and engineering requirements specific to bond coat applications; and (iii) a ranking policy-based selection process that integrates ML predictions of phase-specific oxidation rates and the factor of phase elemental difference. 

![Figure 1](https://github.com/user-attachments/assets/1f7d4767-3df3-4e7f-986c-56c32d90fd66)
# Guide for using the code
The database is saved in folder docs, named Data_base.xlsx

functions.py contains all the function needed in ipynb files.

FeatureEngineering.ipynb is the code for the feature selection process. The original feature selection result is saved in folder docs, named featureEngineering_outputlog.xlsx

GBR.ipynb is the code for Gradient boosting regression model training. The code of the training of the remaining 6 models are in folder otherModels.The training code requires database file (Data_base.xlsx) and feature selection result file (featureEngineering_outputlog.xlsx). After training, the model needs to be saved.
