# DD-sim approximation to quantify mobile ions in perovskite solar cells

This repository contains the code to approximate drift-diffusion simulations of capacitance transient, current transient, and capacitance frequency measurements. The physics of the code is explained in the article xy. If you use the code to quantify mobile ions based on the measurements, please make sure to cite the publication. If you have questions or encounter issues, please email b.ehrler@amolf.nl . 

To use the approximation, please follow the following steps: 

## Download repository
Click on the Code button and then Download the Zip archive. 

## Install required packages 
Make sure that you have the following python packages installed: 
- numpy 
- matplotlib
- seaborn
- scipy
- pandas
- lmfit
- cython 

## Check files 
The code consists of six files: 
1. **Device**:
The Device.py file contains the Device class that stores all properties of the perovskite solar cell that you simulate. 
2. **Simulation**:
The Simulation.py file contains the Simulation class that computes the different measurements (capacitance transient, current transient, and capacitance frequency).
3. **params.py**:
The params.py file is the file where you define all device and simulation properties. 
4. **cbernoulli.pyx**
To improve the speed of the simulation we compute the bernoulli function in C. Therefore, we first have to compile the file **cbernoulli.pyx** into C. If you have cython installed, just follow the first line in the Tutorial
5. **setup.py**
File that has to be run to compile the cbernoulli.py file  
5. **Tutorial.ipynb**  
The jupyter notebook **Tutorial.ipynb**  provides a guide that explains some of the basic functionality of the approximation. 


**Thanks for your interest in this work :-)**# DD-Approximation_CtJtCf
