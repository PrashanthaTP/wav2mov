import cv2
import dlib
import torch
from torchvision import utils as vutils
from torchvision.transforms import Normalize
from inference import params

face_detector = None
def load_face_detector():
    return dlib.get_frontal_face_detector()

def convert_and_trim_bb(image, rect):
    """ from pyimagesearch
    https://www.pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/
    """
    # extract the starting and ending (x, y)-coordinates of the
    # bounding box
    start_x = rect.left()
    start_y = rect.top()
    endX = rect.right()
    endY = rect.bottom()
    # ensure the bounding box coordinates fall within the spatial
    # dimensions of the image
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    # compute the width and height of the bounding box
    w = endX - start_x
    h = endY - start_y
    # return our bounding box coordinates
    return (start_x, start_y, w, h)


def load_image(image_path):
    img = cv2.imread(image_path,cv2.IMREAD_COLOR)
    if params.IMAGE_CHANNELS==1:
        return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # height,width = params.IMAGE_SIZE
    # img = cv2.resize(img,(width,height),cv2.INTER_CUBIC)
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def save_image(image,path,normalize=False):
    vutils.save_image(image,path,normalize=normalize)

def show_image(image):
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def preprocess_image(image):
    global face_detector
    if face_detector is None:
         face_detector = load_face_detector()
    face = face_detector(image)[0]
    x,y,w,h = convert_and_trim_bb(image,face)
    target_height,target_width = params.IMAGE_SIZE
    image = cv2.resize(image[y:y+h,x:x+w],(target_width,target_height),interpolation=cv2.INTER_CUBIC)
    if len(image.shape)==2:
        image = image.reshape(image.shape[0],image.shape[1],1)
    # show_image(image)
    image = torch.from_numpy(image).float().permute(2,0,1)/255
    image= Normalize(params.VIDEO_MEAN,params.VIDEO_STD)(image)
    # show_image(image.permute(1,2,0).numpy())
    return image

def repeat_img(img,n):
    if img.shape==3:
        img = img.unsqueeze(0)
    
    return img.repeat(n,1,1,1)
    