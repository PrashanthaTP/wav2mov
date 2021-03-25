import cv2
import numpy as np
import dlib
import imutils#for image resizing 
from imutils import face_utils 

import os

shape_predictor_path = os.path.join(os.path.dirname(__file__),'shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

mouth_pos = face_utils.FACIAL_LANDMARKS_IDXS['mouth']
mouth_start_pos , mouth_end_pos = mouth_pos

def shape_to_np(shape, dtype: str = 'int'):
    """
    Description of shape_to_np
    ---------------------------

    Function that creates numpy array from shape which is  68 face landmarks.

    Args:
    ---------

    + shape (object)
            predictor object containing x,y coordinates
    + dtype='int' (numpy dtype)

    """

    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def get_face_landmarks(gray):

    rects = detector(gray, 1)

    for face in rects:  # Expecting only one face in the video,but still looping 😂
        shape = predictor(gray, face)
        shape = shape_to_np(shape)
        return shape

def get_framewise_mouth_coords(video_path:str,frame_count:int):
    
    mouth_coords = []
    cap = None
    try:
      
        cap = cv2.VideoCapture(str(video_path))

        if(not cap.isOpened()):
            print("Cannot open video stream or file!")
            
        while cap.isOpened():
            frameId = cap.get(1)
           
            ret, image = cap.read()

            if not ret:
                break
            
            # image = imutils.resize(image, width=128)
            image = imutils.resize(image, width=500)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            shape = get_face_landmarks(gray)
            
            # if debug:
            #     for (x, y) in shape[48:68]:
            #         cv2.circle(gray, (x, y), 1, (0, 0, 255), -1)
            #     cv2.imshow(f"{video_path}",gray)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            #     return 
            #taking only moth coordinates 
            #shape : [(x1,y1),(x2,y2),...(x68,y68)] ==> mouth coords [x_mouthpos_start,....,x_mouthpos_end,y_mouthpos_start,....y_mouthpos_end]
            # mouth_coords.append( np.hstack( tuple(zip(*shape[mouth_pos[0]:mouth_pos[1]]) )) )
            mouth_coords.append(shape[mouth_start_pos:mouth_end_pos])
            
            
    except Exception as e :
        print(e) 
    finally:    
        
        cap.release()
        cv2.destroyAllWindows()
        # print("Cap closed")
        return mouth_coords

def show_mouth_coords(video_path:str):
    
   
    cap = None
    try:
      
        cap = cv2.VideoCapture(str(video_path))

        if(not cap.isOpened()):
            print("Cannot open video stream or file!")
            
        while cap.isOpened():
            frameId = cap.get(1)
           
            ret, image = cap.read()

            if not ret:
                break
            
            # image = imutils.resize(image, width=128)
            image = imutils.resize(image, width=500)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            shape = get_face_landmarks(gray)
            
           
            for (x, y) in shape[48:68]:
                cv2.circle(gray, (x, y), 1, (0, 0, 255), -1)
            cv2.imshow(f"{video_path}",gray)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return 
        
            
            
    except Exception as e :
        print(e) 
    finally:    
        
        cap.release()
        cv2.destroyAllWindows()
        # print("Cap closed")
      