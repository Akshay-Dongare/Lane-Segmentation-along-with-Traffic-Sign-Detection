import numpy as np
import cv2
import torch
import os
import time
import argparse
import pathlib
import custom_utils

from models.fasterrcnn_mobilenetv3_large_fpn import create_model
from config import (
    NUM_CLASSES, DEVICE, CLASSES
)

#from moviepy.editor import VideoFileClip
#from IPython.display import HTML
from keras.models import load_model


# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def road_lines(image):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """

    # Get image ready for feeding into model
    small_img = cv2.resize(image, (160,80))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = keras_model.predict(small_img)[0] * 255

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > 10:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    lane_image = cv2.resize(lane_drawn, (1280,720))
    

    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 1, lane_image.astype(np.uint8), 1, 0)

    return result

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', help='path to input video',
    default='data/inference_data/video_1.mp4'
)
parser.add_argument(
    '-rs', '--resize', default=None, type=int,
    help='provide an integer to resize the image,\
          e.g. 300 will resize image to 300x300',
)
args = vars(parser.parse_args())

# For same annotation colors each time.
np.random.seed(42)

# Create inference result dir if not present.
os.makedirs(os.path.join('../inference_outputs', 'videos'), exist_ok=True)

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load the best model and trained weights
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('../outputs/last_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()


# Load Keras model
keras_model = load_model('Lane_Detector.h5')
# Create lanes object
lanes = Lanes()


# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.3

cap = cv2.VideoCapture(args['input'])

if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

save_name = str(pathlib.Path(args['input'])).split(os.path.sep)[-1].split('.')[0]
# define codec and create VideoWriter object 
out = cv2.VideoWriter(f"../inference_outputs/videos/{save_name}.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (frame_width, frame_height))

frame_count = 0 # to count total frames
total_fps = 0 # to get the final frames per second

# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret:

       
        image = frame.copy()

        frame = road_lines(image)

        if args['resize'] is not None:
            image = cv2.resize(image, (args['resize'], args['resize']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).cuda()
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        # get the start time
        start_time = time.time()

        with torch.no_grad():
            # get predictions for the current frame
            outputs = model(image.to(DEVICE))
        end_time = time.time()
        
        # get the current fps
        fps = 1 / (end_time - start_time)
        # add `fps` to `total_fps`
        total_fps += fps
        # increment frame count
        frame_count += 1
        
        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            # get all the predicited class names
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
            
            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                class_name = pred_classes[j]
                color = COLORS[CLASSES.index(class_name)]
                frame = custom_utils.draw_boxes(frame, box, color, args['resize'])
                frame = custom_utils.put_class_text(
                    frame, box, class_name,
                    color, args['resize']
                )
        cv2.putText(frame, f"{fps:.1f} FPS", 
                    (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                    2, lineType=cv2.LINE_AA)


        
        # Where to save the output video
        #vid_output = 'video_1_trimmed_1_OUTPUT.mp4'
        # Location of the input video
        #clip1 = VideoFileClip("video_1_trimmed_1.mp4")
        # Create the clip
        #vid_clip = frame.fl_image(road_lines)
        

        #vid_clip.ipython_display()   

        #vid_clip.write_videofile("experiiimeennttallll.mp4", audio=False)

        #lets seeee if this works
        #frame = road_lines(frame)

        cv2.imshow('image', frame)
        out.write(frame)
        # press `q` to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()

# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")