
# FCN -- 25
CUDA_VISIBLE_DEVICES=0 python3 main.py --operation training --output_path /home/kno/remote_sensing_segmentation/output/fcn/25/ --dataset_input_path /home/kno/dataset_laranjal/Dataset_Laranjal/Parrot\ Sequoia/all/ --dataset_gt_path /home/kno/dataset_laranjal/Dataset_Laranjal/Arvore_Segmentacao\ \(Sequoia\)/sequoia_raster.tif --num_classes 2 --model_name fcn_25_3_2x_icpr --values 25 --reference_crop_size 25 --reference_stride_crop 15 --distribution_type single_fixed

# SEGNET -- 25
CUDA_VISIBLE_DEVICES=0 python3 main.py --operation training --output_path /home/kno/remote_sensing_segmentation/output/segnet/25/ --dataset_input_path /home/kno/dataset_laranjal/Dataset_Laranjal/Parrot\ Sequoia/all/ --dataset_gt_path /home/kno/dataset_laranjal/Dataset_Laranjal/Arvore_Segmentacao\ \(Sequoia\)/sequoia_raster.tif --num_classes 2 --model_name segnet --values 25 --reference_crop_size 25 --reference_stride_crop 15 --distribution_type single_fixed

# UNET  -- 32
CUDA_VISIBLE_DEVICES=0 python3 main.py --operation training --output_path /home/kno/remote_sensing_segmentation/output/unet/32/ --dataset_input_path /home/kno/dataset_laranjal/Dataset_Laranjal/Parrot\ Sequoia/all/ --dataset_gt_path /home/kno/dataset_laranjal/Dataset_Laranjal/Arvore_Segmentacao\ \(Sequoia\)/sequoia_raster.tif --num_classes 2 --model_name unet --values 32 --reference_crop_size 32 --reference_stride_crop 20 --distribution_type single_fixed

#### to run

# TGRS -- 25,50
CUDA_VISIBLE_DEVICES=0 python3 main.py --operation training --output_path /home/kno/remote_sensing_segmentation/output/ddcnn/25_50_v2/ --dataset_input_path /home/kno/dataset_laranjal/Dataset_Laranjal/Parrot\ Sequoia/all/ --dataset_gt_path /home/kno/dataset_laranjal/Dataset_Laranjal/Arvore_Segmentacao\ \(Sequoia\)/sequoia_raster.tif --num_classes 2 --model_name dilated_grsl_rate8 --reference_crop_size 25 --reference_stride_crop 15 --values 25,50 --distribution_type multi_fixed

# DEEPLAB
CUDA_VISIBLE_DEVICES=1 python3 main.py --operation training --output_path /home/kno/remote_sensing_segmentation/output/deeplab/25/ --dataset_input_path /home/kno/dataset_laranjal/Dataset_Laranjal/Parrot\ Sequoia/all/ --dataset_gt_path /home/kno/dataset_laranjal/Dataset_Laranjal/Arvore_Segmentacao\ \(Sequoia\)/sequoia_raster.tif --num_classes 2 --model_name deeplabv3+ --reference_crop_size 25 --reference_stride_crop 15 --values 25 --distribution_type single_fixed

# PIXELWISE
CUDA_VISIBLE_DEVICES=0 python3 main.py --operation training --output_path /home/kno/remote_sensing_segmentation/output/pixelwise/32_48/ --dataset_input_path /home/kno/dataset_laranjal/Dataset_Laranjal/Parrot\ Sequoia/all/ --dataset_gt_path /home/kno/dataset_laranjal/Dataset_Laranjal/Arvore_Segmentacao\ \(Sequoia\)/sequoia_raster.tif --num_classes 2 --model_name pixelwise --reference_crop_size 32 --reference_stride_crop 20 --values 32 --distribution_type single_fixed
