# APFormer

This repo is the official implementation for:\
Lighter Can Be Better: Rethinking Transformers in Medical Image Segmentation Through Adaptive Pruning.\
(The details of our APFormer can be found at the models directory in this repo or in the paper.)

## Requirements

* python 3.6
* pytorch 1.8.0
* torchvision 0.9.0
* more please see the requirements.txt

## Datasets

* The ISIC 2018 dataset could be acquired from [here](https://challenge.isic-archive.com/data/)
* The Synapse dataset could be acquired from [here](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)

## Training

Commands for training on the ISIC 2018 dataset
```
python train_ISIC.py
```
Commands for training on the Synapse dataset
```
python train_synapse.py
```
## Testing

Commands for testing
``` 
python test.py
```
## Reference

1. [vit-pytorch](https://github.com/lucidrains/vit-pytorch)
