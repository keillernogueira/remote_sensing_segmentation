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

# Usage

You can train a new model using the following command:

```
python3 main.py --operation training \
                --output_path fcn/256/ \
                --dataset arvore \
                --dataset_input_path datasets/arvore/image.tif \
                --dataset_gt_path datasets/arvore/gt.tif \
                --num_classes 2 \
                --model_name fcn_50_3_2x \
                --learning_rate 0.01 \
                --weight_decay 0.05 \
                --batch_size 16 \
                --niter 200000 \
                --reference_crop_size 256 \
                --reference_stride_crop 256 \
                --values 256 \
                --distribution_type single_fixed
```

where,

1. `operation` represents the process that will be performed.
Two options:
	1. `training`, to train a new model,
	2. `generate_map`, to classify the whole image and generate an output map using a trained model 
	(must inform flag `--model_path` in this case)
2.  `output_path` is the path to save models, images, etc
3. `dataset` is the name of the current dataset. This is only used in the `dataloaders/factory.py`
4. `dataset_input_path` path to the input image
5. `dataset_gt_path` path to the input ground-truth
6. `num_classes` is the number of classes
7. `model_name` is the model name. This is only used in the `networks/factory.py`.
Options are:
	1. For FCN: `fcn_25_3_2x_icpr`, `fcn_50_3_2x`
	2. For U-Net: `unet`
	3. For SegNet: `segnet`, `segnet_4`
	4. For DeepLabV3+: `deeplabv3+`
	5. For DCNN: `dilated_grsl_rate8`
8. `learning_rate` corresponds to the learning rate used in the Stochastic Gradient Descent
9. `weight_decay` represents the weight decay used to regularize the learning
10. `batch_size` is the size of the batch
11. `niter` is the number of iterations of the algorithm (used related to the epoch)
12. `reference_crop_size` represents the reference crop size used to map the input. This will be used to 
probe the input image creating a set of positions (x,y) that will be further used to generate the patches during the processing.
This must be an integer.
13. `reference_stride_crop` represents the reference stride size to map the input. This must be an integer.
14. `distribution_type` represents the probability distribution that should be used to select the values.
Options are:
	1. `single_fixed`, which uses one patch size (provide in the flag `values`) during the whole training,
	2. `multi_fixed`, equally divides the probability into the provided values.
15. `values` represents the values of patch size that will be used (together with the distribution) during the processing.
This can be a single value if used with `distribution_type = single_fixed`, or a sequence of integers separated by a comma 
(ex.: 50,75,100) if used with the other distributions.

# Using your own data

To use your own data, you need to implement a dataloader (in the dataloaders folder) and map this dataloader to the 
`dataloaders\factory.py`.
This dataloader will be responsible to load the original input data and labels into the memory and to manipulate this data.
Your dataloader needs to have, at least, the following attributes:

1. `data` or `train_data and test_data`, which will store the original data
2. `labels` or `train_labels and test_labels`, which will store the original ground-truth
3. `train_distrib` and `test_distrib`, which will store the positions (x,y) that will be used to create the patches during the processing.
This can be created internally using a method similar to the `dataloaders\utils.py -> create_distrib_multi_images()`.
The flags `reference_crop_size` and `reference_stride_crop` are used to create the `train_distrib` and `test_distrib`,
which will store positions (x, y) and will be used to create dynamically create the input patches.
4. `num_classes`, which will store the number of classes,
5. `_mean` and `_std`, which will the mean and standard deviation for normalization purposes.
This can be created internally using a method similar to the `dataloaders\utils.py -> create_or_load_mean()`

After creating your dataloader, just run the code and everything else should work.

# Implemented Networks

## Pixelwise

[Pixelwise](https://arxiv.org/abs/1804.04020) classifies each pixel of the input image independently.
Precisely, each pixel is represented by a context window, i.e., overlapping fixed-size patches, in which each one is centered on a specific
pixel helping to understand the spatial patterns around that pixel. Observe that these context windows
are really necessary because the pixel itself has not enough information to be used in its classification.
Such patches are, in fact, used to train and evaluate the network. In both processes, the ConvNet
outputs a class for each input context window, which is associated with the central pixel of the window.

    @inprocedings{knogueira_sibgrapi_2015,
        author={K. {Nogueira} and W. O. {Miranda} and J. A. D. {Santos}},
        booktitle={2015 28th SIBGRAPI Conference on Graphics, Patterns and Images},
        title={Improving Spatial Feature Representation from Aerial Scenes by Using Convolutional Networks},
        year={2015},
        pages={289-296},
        month={Aug}
    }

## Fully Convolutional Networks (FCN)

[Fully Convolutional Network (FCN)](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) was one of the 
first deep learning-based techniques proposed to perform semantic segmentation.
This network extracts features and generates an initial coarse classification map using 
a set of convolutional layers that, due to their internal configuration, outputs a spatially reduced (when compared to 
the original input) outcome.
In order to restore the original resolution and output the thematic map, this approach 
employs [deconvolution layers](https://github.com/vdumoulin/conv_arithmetic) (also known as transposed convolution)
that learn how to upsample the initial classification map and 
produce the final dense prediction. 

    @inproceedings{long2015fully,
        title={Fully convolutional networks for semantic segmentation},
        author={Long, Jonathan and Shelhamer, Evan and Darrell, Trevor},
        booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
        pages={3431--3440},
        year={2015}
    }


## U-Net

[U-Net](https://arxiv.org/pdf/1505.04597.pdf) was one of the first networks to propose encoder-decoder architectures to
perform semantic segmentation. In this design, the encoder is usually composed of several convolution and 
pooling layers, and responsible to extract the features and generate an initial coarse prediction map. 
The decoder, commonly composed of convolution, deconvolution 
and/or unpooling layers, is responsible to further process the initial prediction map, 
increasing its spatial resolution gradually and generating the final prediction. 
Note that, normally, the decoder can be seen as a mirrored/symmetrical version of the encoder, 
with the same number of layers but replacing some of the operations with their counterparts 
(i.e., convolution with deconvolution, pooling with unpooling, etc).

    @inproceedings{ronneberger2015u,
      title={U-net: Convolutional networks for biomedical image segmentation},
      author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
      booktitle={International Conference on Medical image computing and computer-assisted intervention},
      pages={234--241},
      year={2015},
      organization={Springer}
    }

## SegNet

[SegNet](https://arxiv.org/pdf/1511.00561.pdf) is another type of encoder-decoder network proposed specifically 
for semantic segmentation. However, differently from the previous model, this network employs unpooling operations, 
instead of deconvolution layers, in the decoder to increase the spatial resolution of the coarse map generated by 
the encoder.

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

## DeepLabV3+

[DeepLabV3+](http://openaccess.thecvf.com/content_ECCV_2018/papers/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.pdf)
is another encoder-decoder network. In order to aggregate multi-scale information, this method uses:
(i) features extracted from multiple layers,
(ii) multi-parallel dilated convolutions in a module called Atrous Spatial Pyramid Pooling.

    @inproceedings{chen2018encoder,
        title={Encoder-decoder with atrous separable convolution for semantic image segmentation},
        author={Chen, Liang-Chieh and Zhu, Yukun and Papandreou, George and Schroff, Florian and Adam, Hartwig},
        booktitle={Proceedings of the European conference on computer vision (ECCV)},
        pages={801--818},
        year={2018}
    }

## Dynamic Dilated ConvNet

[Dynamic Dilated ConvNet (DDCNN)](https://arxiv.org/abs/1804.04020) proposes a novel multi-scale training strategy 
that uses dynamically-generated input images to converge a dilated model that never downsamples the input data.
Technically, this technique receives as input the original images and a probability distribution over the possible
input sizes, i.e., over the sizes that might be used to generate the input patches. In each iteration of the training
procedure, a size is randomly selected from this distribution and is then used to create a totally new batch.
By processing these batches, each composed of several images with one specific pre-selected size, the model is capable
of capturing multi-scale information. Furthermore, in the prediction step, the algorithm selects,
based on scores accumulated during the training phase for each evaluated input size, the best resolution.
Then, the technique processes the testing images using batches composed of images with the best-evaluated size.

    @article{knogueira_tgrs_2019,
        author={K. Nogueira and M. Dalla Mura and J. Chanussot and W. R. Schwartz and J. A. dos Santos},
        journal={IEEE Transactions on Geoscience and Remote Sensing},
        title={Dynamic Multicontext Segmentation of Remote Sensing Images Based on Convolutional Networks},
        year={2019},
        volume={57},
        number={10},
        pages={7503--7520},
        month={Oct}
    }
