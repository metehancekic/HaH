![alt text][logo]

[logo]: https://github.com/metehancekic/HaH/blob/main/figs/hahblock.png

**Figure 1**: HaH block for image classification DNNs. 

# Hebbian/Anti-Hebbian Learning for Pytorch

Official repository for the paper entitled "Towards Robust, Interpretable Neural Networks via Hebbian/anti-Hebbian Learning: A Software Framework for
Training with Feature-Based Costs". If you have questions you can contact metehancekic [at] ucsb [dot] edu.

Maintainers:
    [WCSL Lab](https://wcsl.ece.ucsb.edu), 
    [Metehan Cekic](https://www.metehancekic.com), 
    [Can Bakiskan](https://wcsl.ece.ucsb.edu/people/can-bakiskan), 

## Dependencies

> numpy==1.20.2\
> torch==1.10.2

## How to install

The most recent stable version can be installed via python package installer "pip", or you can clone it from the git page.

```bash
pip install hahtorch
```
or

```bash
git clone git@github.com:metehancekic/HaH.git
```

## Experiments 

We used CIFAR-10 image classification to show the effectiveness of our module. We train a VGG16 in a standard fashion and train another VGG16 that contains HaHblocks with layer-wise HaHCost as a supplement. Details of our experiments can be found in our recent [paper](https://arxiv.org/abs/2202.13074)

### CIFAR10 Image Classification with VGG16 model as Backbone

![alt text][hahvgg]

[hahvgg]: https://github.com/metehancekic/HaH/blob/main/figs/hahvgg.png

**Figure 2**: HaH VGG16, our proposed architecture for HaH training, see [paper](https://arxiv.org/abs/2202.13074) for more detail.

![alt text][hahresults]

[hahresults]: https://github.com/metehancekic/HaH/blob/main/figs/hahresults.png

**Table 1**: CIFAR10 classification: Performance of the HaH trained network against different input corruptions on the test set. For all of the adversarial attacks, we use AutoAttack which is an ensemble of parameter-free attacks, see [paper](https://arxiv.org/abs/2202.13074) for more detail.

## Current Version #

0.0.5

## Sources #

- [PyPi page for the code](https://pypi.org/project/hahtorch/)

- [Git repo for the code](https://github.com/metehancekic/HaH)
