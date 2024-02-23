from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

def extract_frames(video_file, output_folder):
    # Modified from https://github.com/PyImageSearch/imutils/blob/master/demos/read_frames_fast.py


    def filterFrame(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame


    fvs = FileVideoStream(video_file, transform=filterFrame).start()
    time.sleep(1.0)
    
    video_name = os.path.basename(video_file)
    video_name = os.path.splitext(video_name)[0]
    
    # start the FPS timer
    fps = FPS().start()

    # loop over frames from the video file stream
    frame_num = 0
    while fvs.running():
        frame = fvs.read()
        if fvs.Q.qsize() < 2:  # If we are low on frames, give time to producer
            time.sleep(0.001)  # Ensures producer runs now, so 2 is sufficient
        frame_name = f"{video_name}-{frame_num:06d}.jpg"
        cv2.imwrite
        fps.update()
        

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    fvs.stop()
    