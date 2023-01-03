# Lane-Segmentation-along-with-Traffic-Sign-Detection
Performed real-time semantic segmentation for lane detection combined with traffic sign recognition on a video stream. Trained Faster R-CNN and Resnet50 on the GTSRB dataset for traffic sign recognition and trained an FCN on custom dataset for lane segmentation
## Train
Train traffic sign recognizer model and place it in `outputs/`
## Run
Run `src/inference_video.py` with `--input` flag followed by input video path
## Demo
https://youtu.be/et6YM88DWls
