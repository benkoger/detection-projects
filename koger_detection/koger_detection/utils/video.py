from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2


def extract_crops(video_names, video_folder_path, save_folder, num_extract, 
                  crop_size, min_frame=0, max_frame=None, save_triplet=False, 
                  triplet_spacing=30):
    """Extract random crops from video clips.
    
    Can extract frames a set time before and after each crop to
    make annotation easier so the annotator can use local movement.
    
    video_names: list of names of video clips
    video_folder_path: full path to folder where videos are stored
    save_folder: full path to folder where crops should be saved
    num_extract: how many random focal frames to extract from each
        video clip
    crop_size: length of square crop in pixels
    min_frame: min frame number in video frames should be chosen from 
    max_frame: max frame number in video frames should be chosen from
    save_triplet: if True, save frames a fixed number of frames before
        and after focal frame to aid in annoation effort
    triplet_spacing: how many frames before and after focal frame triplet
        frames should be. (Only relevant if triplet spacing is True)
    
    """
    for video_name in video_names:
        video_file = os.path.join(video_folder_path, video_name)
        video_name = os.path.splitext(video_name)[0] # remove extension

        cap = cv2.VideoCapture(video_file)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if max_frame is None:
            max_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) + 1
        if save_triplet:
            # Only choose frames for annotation that have space before and after
            # for all frames in triplet
            max_frame -= triplet_spacing - 1
            min_frame += triplet_spacing
        # Randomly choose the specified number of frames to extract from the given range
        frame_nums_to_save = np.random.randint(min_frame, max_frame, num_extract)
        for frame_num in frame_nums_to_save:
            frame_file = os.path.join(save_folder, f"{video_name}_frame_{frame_num}")
            if crop_size:
                # If gaussian is True, then random crops sampled from gaussian 
                # centered at center of frame with 1 std equal to half height/width 
                # of the frame
                top_left = random_top_left([height, width], crop_size, gaussian=False)
                if top_left is False:
                    print(f"skipping {frame_file}")
                    print(f"No valid crop found. Check frame size and crop size.")
                    print(f"Frame size h:{height}, w:{width}, crop_size: {crop_size}.")
                    continue
                # Add where crop comes from to file name so we can find it in the 
                # original image later if we want to
                frame_file += f"_top_{top_left[0]}_left_{top_left[1]}"
            # Naming convention here is to append an 'f' if the focal frame that will
            # be annotated and a 'a' or 'b' if the first or last frame in a triplet
            save_frame(cap, frame_num, frame_file+"_f.jpg", crop_size, top_left)
            if save_triplet:
                next_frame_num = frame_num + triplet_spacing
                frame_file = os.path.join(save_folder, f"{video_name}_frame_{frame_num}")
                if crop_size:
                    frame_file += f"_top_{top_left[0]}_left_{top_left[1]}"
                save_frame(cap, next_frame_num, frame_file+"_a.jpg", 
                           crop_size, top_left
                          )
                prev_frame_num = frame_num - triplet_spacing 
                frame_file = os.path.join(save_folder, f"{video_name}_frame_{frame_num}")
                if crop_size:
                    frame_file += f"_top_{top_left[0]}_left_{top_left[1]}"
                save_frame(cap, prev_frame_num, frame_file+"_b.jpg",
                           crop_size, top_left
                          )
        cap.release()
    

def random_top_left(im_shape, crop_size, gaussian=False):
    """ Get a random top left coordinate for a crop of size (crop_size * crop_size).
    
    Args:
        im_shape: (h, w, ...)
        crop_size: size of ultimate crop
        gaussian: If True, then pull coordinates from gaussian with mean
            at the center of possible range of top left values and 1 standard 
            deviation to the min and max top left values
    
    Returns [top, left] 
        or False if no valid top_left value based on image and crop size
    """
    
    height, width = im_shape[:2]
    if gaussian:
        mean_top = (width-crop_size) / 2
        mean_left = (width-crop_size) / 2
        top = -1
        left = -1
        while ((top >= (height-crop_size)) or (top < 0)):
            top = int(np.random.normal(mean_top, mean_top))
        while ((left >= (width-crop_size)) or (left < 0)):
            left = int(np.random.normal(mean_left, mean_left))
    else:
        if height - crop_size < 0:
            return False
        top = np.random.randint(0, height-crop_size)

        if width - crop_size < 0:
            return False
        left = np.random.randint(0, width-crop_size)

    top_left = [top, left]
    return top_left


def save_frame(cap, frame_num, outfile, crop_size=None, top_left=None):
    """ Save frame from cv2 VideoCapture
    
    Args: 
        cap: cv2.VideoCapture object
        frame_num: the frame number to save
        outfile: where to save frame
        crop_size: pixels, size of crop (square). If None, no crop.
        top_left: (i, j) coordinate of top left corner of crop (if not None)
            if None and crop_size is not None, then choose random values
            
    Return crop_top_left
        
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if frame is not None:
        if crop_size:
            if not top_left:
                raise ValueError(f"If cropping, must provide top_left: {top_left}")
            top, left = top_left
            frame = frame[top:top+crop_size, left:left+crop_size]

        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        cv2.imwrite(outfile, frame)
    else:
        print(f"Frame doesn't exist {outfile}.")


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
    