

# TGRS
CUDA_VISIBLE_DEVICES=0 python3 main.py --operation training --output_path /home/kno/remote_sensing_segmentation/output/ddcnn/32_48/ --dataset_input_path /home/kno/dataset_laranjal/Dataset_Laranjal/Parrot\ Sequoia/all/ --dataset_gt_path /home/kno/dataset_laranjal/Dataset_Laranjal/Arvore_Segmentacao\ \(Sequoia\)/sequoia_raster.tif --num_classes 2 --model_name dilated_grsl_rate8 --reference_crop_size 32 --reference_stride_crop 20 --values 32,48 --distribution_type multi_fixed

# FCN
CUDA_VISIBLE_DEVICES=0 python3 main.py --operation training --output_path /home/kno/remote_sensing_segmentation/output/fcn/32/ --dataset_input_path /home/kno/dataset_laranjal/Dataset_Laranjal/Parrot\ Sequoia/all/ --dataset_gt_path /home/kno/dataset_laranjal/Dataset_Laranjal/Arvore_Segmentacao\ \(Sequoia\)/sequoia_raster.tif --num_classes 2 --model_name fcn_25_3_2x_icpr --values 32 --reference_crop_size 32 --reference_stride_crop 20 --distribution_type single_fixed

# SEGNET
CUDA_VISIBLE_DEVICES=0 python3 main.py --operation training --output_path /home/kno/remote_sensing_segmentation/output/segnet/32/ --dataset_input_path /home/kno/dataset_laranjal/Dataset_Laranjal/Parrot\ Sequoia/all/ --dataset_gt_path /home/kno/dataset_laranjal/Dataset_Laranjal/Arvore_Segmentacao\ \(Sequoia\)/sequoia_raster.tif --num_classes 2 --model_name segnet --values 32 --reference_crop_size 32 --reference_stride_crop 20 --distribution_type single_fixed

# UNET
# CUDA_VISIBLE_DEVICES=0 python3 main.py --operation training --output_path /home/kno/remote_sensing_segmentation/output/unet/32/ --dataset_input_path /home/kno/dataset_laranjal/Dataset_Laranjal/Parrot\ Sequoia/all/ --dataset_gt_path /home/kno/dataset_laranjal/Dataset_Laranjal/Arvore_Segmentacao\ \(Sequoia\)/sequoia_raster.tif --num_classes 2 --model_name unet --values 32 --reference_crop_size 32 --reference_stride_crop 20 --distribution_type single_fixed

# PIXELWISE
CUDA_VISIBLE_DEVICES=0 python3 main.py --operation training --output_path /home/kno/remote_sensing_segmentation/output/pixelwise/32_48/ --dataset_input_path /home/kno/dataset_laranjal/Dataset_Laranjal/Parrot\ Sequoia/all/ --dataset_gt_path /home/kno/dataset_laranjal/Dataset_Laranjal/Arvore_Segmentacao\ \(Sequoia\)/sequoia_raster.tif --num_classes 2 --model_name pixelwise --reference_crop_size 32 --reference_stride_crop 20 --values 32 --distribution_type single_fixed

CUDA_VISIBLE_DEVICES=1 python3 main.py --operation training --output_path /home/kno/remote_sensing_segmentation/output/deeplab/ --dataset_input_path /home/kno/dataset_laranjal/Dataset_Laranjal/Parrot\ Sequoia/all/ --dataset_gt_path /home/kno/dataset_laranjal/Dataset_Laranjal/Arvore_Segmentacao\ \(Sequoia\)/sequoia_raster.tif --num_classes 2 --model_name deeplabv3+ --reference_crop_size 32 --reference_stride_crop 20 --values 32 --distribution_type single_fixed
