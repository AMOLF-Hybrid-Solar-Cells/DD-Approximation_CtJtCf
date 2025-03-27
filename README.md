# Drift-diffusion simulation approximation to quantify mobile ions in perovskite solar cells

This repository contains the code to approximate drift-diffusion simulations of capacitance transient, current transient, and capacitance frequency measurements. With these techniques, the ionic conductivity and in a range also the ion density and diffusion coefficient of perovskite solar cells can be determined. The theory behind the code is explained in the article xy. If you use the code please make sure to cite the publication. If you have questions or encounter issues, please email b.ehrler@amolf.nl . **Thanks for your interest in this work :-)**

To use the approximation, please follow the following steps: 

## Download repository
Click on the Code button and then download the Zip archive.

The code consists of six files: 
1. **Device**:
The Device.py file contains the Device class that stores all properties of the perovskite solar cell that you simulate. 
2. **Simulation**:
The Simulation.py file contains the Simulation class that computes the different measurements (capacitance transient, current transient, and capacitance frequency).
3. **params.py**:
The params.py file is the file where you define all device and simulation properties. Feel free to change the device parameters in this file! 
4. **cbernoulli.pyx**:
To improve the speed of the simulation we compute the bernoulli function in C. Therefore, we first have to compile the file **cbernoulli.pyx** into C (see above). 
5. **setup.py**:
File that has to be run to compile the cbernoulli.py file  
5. **Tutorial.ipynb**:  
The jupyter notebook **Tutorial.ipynb**  provides a guide that explains the basic functionality of the approximation. 

## Install required packages 
Make sure that you have the following python packages installed: 
- numpy 
- matplotlib
- seaborn
- scipy
- pandas
- lmfit
- cython 
- ipykernel (for the Tutorial)

## Compile cbernoulli.pyx
Open the folder in a jupyter notebook or the IDE of your choice. In order to run the code, you have to compile the cbernoulli.pyx with Cython. You can either do this by running `python setup.py build_ext --inplace` in your terminal or by following the first steps in the Tutorial. 

## Get started! 
Now you can use the code! Feel free to check out the rest of the tutorial to get familiar with it's capabilities!


