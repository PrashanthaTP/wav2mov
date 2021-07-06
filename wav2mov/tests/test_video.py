import dlib
import cv2
import os
from wav2mov.logger import get_module_level_logger
logger = get_module_level_logger(__name__)


DIR = r'D:\dataset_lip\GRID\video_6sub_500'
video = os.path.join(DIR,'s4_p_swbxza.mov')


face_detector = dlib.get_frontal_face_detector()

def convert_and_trim_bb(image, rect):
    """ from pyimagesearch
    https://www.pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/
    """
    # extract the starting and ending (x, y)-coordinates of the
    # bounding box
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()
    # ensure the bounding box coordinates fall within the spatial
    # dimensions of the image
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    # compute the width and height of the bounding box
    w = endX - startX
    h = endY - startY
    # return our bounding box coordinates
    return (startX, startY, w, h)
def show_img(image_name,image):
    cv2.imshow(str(image_name),image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def get_video_frames(video_path,img_size:tuple):
    try:
        logger.debug(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if(not cap.isOpened()):
            logger.error("Cannot open video stream or file!")
        frames = []
        i = 0
        while cap.isOpened():
            frameId = cap.get(1)
            ret, image = cap.read()
            if not ret:
                break
            try:
                i+=1
                #image[top_row:bottom_row,left_column:right_column]
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)#other libraries including matplotlib,dlib expects image in RGB
                print(image.shape,i)
                face = face_detector(image)[0]#get first face object
                print(face.top(),face.bottom())
                x,y,w,h = convert_and_trim_bb(image,face)
                # image = cv2.resize(image[y:y+h,x:x+w],img_size,interpolation=cv2.INTER_CUBIC)
                
                # show_img(len(frames),image)
            except Exception as e:
                # print(e)
                # logger.error(e)
                continue
            frames.append(image)
        return frames 
    except Exception as e:
        logger.exception(e)
        
        
if __name__ == '__main__':
    get_video_frames(video,(256,256))