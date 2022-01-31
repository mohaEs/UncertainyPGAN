# UncertainyPGAN

This repository contrains the scripts used for the challenge 
https://qubiq21.grand-challenge.org/

The method is based on progressive GAN which is uncertainty aware: </br>
main.py </br>
it is developed based on the code and paper: </br>
https://github.com/ExplainableML/UncerGuidedI2I </br>
https://arxiv.org/abs/2106.15542

To improve the results, the soft dice criterion is added to the loss function: </br>
main_v2_dice.py

The complete description of the method is available at _report.pdf_

postprocessing and converting outputs to nifti files: </br>
main_convert2nifti.py 
