# Semantic segmentation for remote sensing images

This repo implements several deep learning methods for semantic segmentation of remote sensing images.

# Requirements

- Python 3.x
- [Tensorflow](https://www.tensorflow.org/install/docker) (<2.0)
- [SKLearn](https://scikit-learn.org/stable/install.html)
- [Scipy](https://www.scipy.org/install.html)
- [SKImage](https://scikit-image.org/docs/dev/install.html)
- [ImageIO](https://imageio.readthedocs.io/en/stable/installation.html)

```
pip install scikit-learn scipy scikit-image imageio
```

# Implemented Networks

- Pixelwise

    - [Paper](https://arxiv.org/abs/1804.04020)


    @inprocedings{knogueira_sibgrapi_2015,
        author={K. {Nogueira} and W. O. {Miranda} and J. A. D. {Santos}},
        booktitle={2015 28th SIBGRAPI Conference on Graphics, Patterns and Images},
        title={Improving Spatial Feature Representation from Aerial Scenes by Using Convolutional Networks},
        year={2015},
        pages={289-296},
        month={Aug}
    }

- Fully Convolutional Networks (FCN)

    - [Paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)

    
    @inproceedings{long2015fully,
        title={Fully convolutional networks for semantic segmentation},
        author={Long, Jonathan and Shelhamer, Evan and Darrell, Trevor},
        booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
        pages={3431--3440},
        year={2015}
    }

- SegNet

    - [Paper](https://arxiv.org/pdf/1511.00561.pdf)
    
    
    @article{badrinarayanan2017segnet,
        title={Segnet: A deep convolutional encoder-decoder architecture for image segmentation},
        author={Badrinarayanan, Vijay and Kendall, Alex and Cipolla, Roberto},
        journal={IEEE transactions on pattern analysis and machine intelligence},
        volume={39},
        number={12},
        pages={2481--2495},
        year={2017},
        publisher={IEEE}
    }

- Dynamic Dilated ConvNet

    - [Paper](https://arxiv.org/abs/1804.04020)
    
    
    @article{knogueira_tgrs_2019,
        author={K. {Nogueira} and M. {Dalla Mura} and J. {Chanussot} and W. R. {Schwartz} and J. A. {dos Santos}},
        journal={IEEE Transactions on Geoscience and Remote Sensing},
        title={Dynamic Multicontext Segmentation of Remote Sensing Images Based on Convolutional Networks},
        year={2019},
        volume={57},
        number={10},
        pages={7503-7520},
        month={Oct}
    }
    
