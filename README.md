![alt text][logo]

[logo]: https://github.com/metehancekic/HaH/blob/main/figs/hahblock.png

**Figure 1**: HaH block for image classification DNNs. 

# Hebbian/Anti-Hebbian Learning for Pytorch

If you have questions you can contact metehancekic [at] ucsb [dot] edu

## Pre-requisites

Install the dependencies

> numpy==1.20.2
> torch==1.10.2

## How to install

We have a pypi module which can be installed simply with following command:

```bash
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps hahtorch
```
Or one can clone the repository.

```bash
git clone git@github.com:metehancekic/HaH.git
```

## Experiments 

We used CIFAR-10 image classification to show the effectiveness of our module. We train a VGG16 in standard fashion and train another VGG16 that contains HaHblocks with layer-wise HaHCost as a supplement. 

### CIFAR10 Image Classification with VGG16 model as Backbone

![alt text][hahvgg]

[hahvgg]: https://github.com/metehancekic/HaH/blob/main/figs/hahvgg.png

**Figure 2**: HaH VGG16, our proposed architecture for HaH training.
