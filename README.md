
<h1 align="center">
  FedArc
  <br>
</h1>

<h4 align="center">Official code repository for paper "FedArc: Additive Angular Margin Loss for Federated Learning in Image Classification".</h4>

<p align="center">
<a href=""><img src="https://img.shields.io/github/stars/tpnam0901/FedArc?" alt="stars"></a>
<a href=""><img src="https://img.shields.io/github/forks/tpnam0901/FedArc?" alt="forks"></a>
<a href=""><img src="https://img.shields.io/github/license/tpnam0901/FedArc?" alt="license"></a>
</p>

<p align="center">
  <a href="#abstract">Abstract</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#citation">Citation</a> •
</p>

## Abstract
> Federated learning (FL) is a method that leverages data from multiple sources to enhance deep learning models while safeguarding data privacy. This approach finds applications in diverse domains such as healthcare, entertainment, and user experience enhancement. Despite its effectiveness, FL's performance is still low compared to centralized training methods, necessitating further improvements. Particularly in critical areas such as medicine, where accurate model predictions are crucial, conventional FL algorithms fall short compared to centralized training. This study introduces a new training approach called FedArc to further improve the vanilla FL algorithm. FedArc takes advantage of feature-based adjustments using cosine angles by incorporating an angular margin loss function alongside the cross-entropy loss. FedArc notably enhances the accuracy of the aggregated models on datasets such as MNIST, CIFAR10, and CIFAR100 while maintaining FL's privacy-preserving attributes. Moreover, we conduct a comprehensive comparative analysis of FedArc against existing FL algorithms to evaluate the impact of the angular margin loss function on the learning process. Our experimental results underscore the effectiveness of the proposed FedArc algorithm when compared with standard benchmarks, proving its potential to advance the field of FL.

## How To Use
- Clone this repository 
```bash
git clone https://github.com/tpnam0901/FedArc.git
cd FedArc
```
- Create a conda environment and install requirements
```bash
conda create -n fedarc python=3.10 -y
conda activate fedarc
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

```bash
cd src && python train.py -cfg configs/base.py
```

## Citation
```bibtex

```
---

> GitHub [@tpnam0901](https://github.com/tpnam0901) &nbsp;&middot;&nbsp;
