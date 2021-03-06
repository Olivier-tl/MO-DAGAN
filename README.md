# MO-DAGAN: Minority Oversampling using Data Augmented GANs
*Abstract: Class imbalance is a common problem that reduces the performance of classification models. One typical solution is to oversample the minority class. However, classical oversampling techniques such as SMOTE or ADASYN are ill-suited for deep learning approaches since they work in feature space. Recently, Generative Adversarial Networks (GANs) have been successfully used to generate artificial training data to re-balance datasets. Nevertheless, these approaches are data hungry and it remains a challenge to train GANs on the limited data of the minority class. In this work, we plan to leverage recent advances in data-efficient GAN training to advance the state of the art in oversampling approaches.*

| Dataset <br /><br /> IR           | MNIST <br /><br /> 10 | <br /><br /> 50 | <br /><br /> 100 | Fashion-MNIST <br /> 10 |  <br /><br /> 50 | <br /><br /> 100 | CIFAR10 <br /><br /> 10 | <br /><br /> 50 | <br /><br /> 100 | SVHN <br /><br /> 10 | <br /><br /> 50 | <br /><br /> 100 |
| --------------------------- |:--:|:--:|:---:|:--:|:--:|:---:|:--:|:--:|:---:|:--:|:--:|:---:|
| EfficentNet                 |  - |  - |  -  |  - |  - |  -  |  - |  - |  -  |  - |  - |  -  |
| EfficentNet + Oversampling  |  - |  - |  -  |  - |  - |  -  |  - |  - |  -  |  - |  - |  -  |
| EfficientNet + WGAN         |  - |  - |  -  |  - |  - |  -  |  - |  - |  -  |  - |  - |  -  |
| EfficientNet + WGAN + ADA   |  - |  - |  -  |  - |  - |  -  |  - |  - |  -  |  - |  - |  -  |



## Getting started
Install the required dependencies:
```
pip install -r requirements.txt
```
Run the following for training the GAN:
```
python main.py --config_path=configs/gan.yaml
```
Run the following for training the classification model:
```
python main.py --config_path=configs/classification.yaml
```


## Refs
* Training Generative Adversarial Networks with Limited Data [[paper](https://arxiv.org/abs/2006.06676)][[code](https://github.com/NVlabs/stylegan2-ada-pytorch)]
* EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks [[paper](https://arxiv.org/abs/1905.11946)][[code](https://github.com/lukemelas/EfficientNet-PyTorch)]
