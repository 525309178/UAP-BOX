# UAP_TOOL
This is a universal adversarial perturbation toolbox, can be applied research on the robustness of neural networks in image classification.

Use this toolbox to make adversarial examples in seconds.

## Quick start：
#### First： 
  ```text
  git clone https://github.com/525309178/UAP_BOX.git
  ```
  
  #### Second: 
  cd Classfier, run this train_cifar10.py.
  ```text
  eg: python train_cifar10.py
  ```
  (Note: When the first run, it will download cifar10-dataset to the folder "RawDatasets". After training, the checkpoint of model will be saved to the "CIFAR10" folder of the current folder).
  
  #### Third:  
  cd Attacks, run this advres.py to train a generator model, it can generate universal adversarial perturbation and make adversarial sample. 
  ```text
   eg: python advres.py --cuda .
  ```
  (Note: Before run this *.py of the folder, you must have a trained model used for image classification.  After training the generator model will be saved to the param "outf" seted path. You can set the "save_adv" params run advresTest.py to save adversarial sample,eg: python advresTest.py --cuda --save_adv=1 )
  
  Such as: ![Alt text](Attacks/27_1.png?raw=true "")

  #### Fourth: 
  cd Evalutions, test  the robustness of image classification models on raw dataset or adversarial dataset. 
  ```text
    eg. python testadv.py
  ```


