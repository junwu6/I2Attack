# I2Attack
An implementation for "Indirect Invisible Poisoning Attacks on Domain Adaptation" (KDD'21) [[Paper]](http://publish.illinois.edu/junwu3/files/2021/06/KDD21_camera_ready_I2Attack-2.pdf).

## Environment Requirements
The code has been tested under Python 3.6.5. The required packages are as follows:
* numpy==1.18.1
* sklearn==0.22.1
* scikit-image==0.16.2
* Pillow==7.0.0
* torch===1.4.0
* torchvision===0.5.0

## Data sets
We used the following data sets in our experiments:
* [MNIST](http://yann.lecun.com/exdb/mnist/), [USPS](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/), [SVHN](http://ufldl.stanford.edu/housenumbers/)
* [Office-31](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/), [Office-Caltech10](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/), [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html)
* [Image-CLEF](https://drive.google.com/file/d/0B9kJH0-rJ2uRS3JILThaQXJhQlk/view), [VisDA2017](http://ai.bu.edu/visda-2017/)

## Run the Codes
For I2Attack on unsupervised domain adaptation on digital data (e.g., svhn and mnist), please run
```
python train_svhn2mnist.py
```

For I2Attack on unsupervised domain adaptation on real-world image data (e.g., office-31), please run
```
pyhton main.py
```

## Acknowledgement
This is the latest source code of **I2Attack** for KDD2021. If you find that it is helpful for your research, please consider to cite our paper:

```
@inproceedings{wu2021indirect,
  title={Indirect Invisible Poisoning Attacks on Domain Adaptation},
  author={Wu, Jun and He, Jingrui},
  booktitle={Proceedings of the 27th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  year={2021},
  organization={ACM}
}
```
