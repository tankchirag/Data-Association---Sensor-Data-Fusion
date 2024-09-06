Object Detection and Tracking performed on RGB Camera data from INFRA-3DRC-Dataset provided by Fraunhofer IVI. 

The following models are tested and used for the same: YOLO v8 (Ultralytics) and Fast-RCNN (Detectron 2). 

## Dataset download from below link

https://github.com/FraunhoferIVI/INFRA-3DRC-Dataset/blob/main/docs/DOWNLOAD_DATASET.md

## Data Association

You need to replace following path in inference block.
1. Path_to_images 
2. Path_to_radar_data
3. calibration_file_path
4. model (best.pt filepath)

