#!/usr/bin/env python
# coding: utf-8

# In[1]:


import face_recognition


# In[2]:


image = face_recognition.load_image_file('fam3.jpeg')


# In[3]:


face_locations = face_recognition.face_locations(image)


# In[4]:


print(len(face_locations))


# In[5]:


image_of_bill =  face_recognition.load_image_file('billgates.jpg')
bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]


# In[6]:


unknown_image = face_recognition.load_image_file('billgates2.jpg')
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]


# In[7]:


results = face_recognition.compare_faces([bill_face_encoding], unknown_face_encoding)
if results[0]:
    print("bill gates")
else:
    print("not bill gates")


# In[6]:


import PIL
from PIL import Image
import face_recognition


# In[9]:


image = face_recognition.load_image_file('fam3.jpeg')
face_locations = face_recognition.face_locations(image)


# In[11]:


for face_location in face_locations:
    top,right,bottom,left = face_location
    
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()
    pil_image.save(f'{top}.jpg')


# In[7]:


from PIL import ImageDraw


# In[6]:


image_of_kundana =  face_recognition.load_image_file('Kundana_Lal.jpeg')
kundana_face_encoding = face_recognition.face_encodings(image_of_kundana)[0]


# In[8]:


image_of_kundana =  face_recognition.load_image_file('Kundana_Lal.jpeg')
kundana_face_encoding = face_recognition.face_encodings(image_of_kundana)[0]
known_face_encoding= [kundana_face_encoding]
known_face_names=["Kundana Lal"]
test_image = face_recognition.load_image_file('family1.jpg')
face_locations= face_recognition.face_locations(test_image)
face_encodings= face_recognition.face_encodings(test_image, face_locations)

pil_image= Image.fromarray(test_image)

draw= ImageDraw.Draw(pil_image)

for(top,right,bottom,left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
    name= "Unknown Person"
    if True in  matches:
        first_match_index= matches.index(True)
        name=known_face_names[first_match_index]
        
    draw.rectangle(((left,top),(right,bottom)),outline=(0,0,0))
    
    text_width,text_height = draw.textsize(name)
    draw.rectangle(((left,bottom - text_height - 10),(right,bottom)),fill=(0,0,0),outline=(0,0,0))
    draw.text((left+6,bottom-text_height - 5), name, fill=(255,255,255,255))
    
del draw

pil_image.show()


# In[2]:


from fer import FER


# In[10]:


from fer import FER
import matplotlib.pyplot as plt
img = plt.imread("Kundana_Lal.jpeg")
detector = FER(mtcnn=True)
detector.detect_emotions(img)
emotion, score = detector.top_emotion(img)
plt.imshow(img)
print(emotion,score)


# In[6]:


from fer import FER
import cv2

img = cv2.imread("disgustface.jpg")
detector = FER(mtcnn=True)
detector.detect_emotions(img)


# In[7]:


emotion, score = detector.top_emotion(img)
print(emotion,score)


# In[18]:


import cv2
from fer import Video
from fer import FER

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    video_filename = cap
    video = Video(video_filename)

    detector = FER(mtcnn=True)
    raw_data = video.analyze(detector, display=True)
    df = video.to_pandas(raw_data)
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()


# In[3]:


from fer import Video
from fer import FER

video_filename = "https://youtu.be/8oZpIFDqC-E"
video = Video(video_filename)

detector = FER(mtcnn=True)
raw_data = video.analyze(detector, display=True)
df = video.to_pandas(raw_data)


# In[3]:


from fer import Video
from fer import FER 
import cv2
cap = cv2.VideoCapture(0)
video_filename = cap
video = Video(video_filename)

# Analyze video, displaying the output
detector = FER(mtcnn=True)
raw_data = video.analyze(detector, display=True)
df = video.to_pandas(raw_data)


# In[14]:


import cv2
from fer import Video
from fer import FER 
import numpy as np
import face_recognition


# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
#cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
        
# Release the VideoCapture object
cap.release()


# In[6]:


import cv2
from fer import Video
from fer import FER 
import numpy as np
import face_recognition
from PIL import Image
from PIL import ImageDraw
import deepface


# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
#cap = cv2.VideoCapture('filename.mp4')
image_of_kundana =  face_recognition.load_image_file('Kundana_Lal.jpeg')
kundana_face_encoding = face_recognition.face_encodings(image_of_kundana)[0]
known_face_encoding= [kundana_face_encoding]
known_face_names=["Kundana Lal"]

while True:
    # Read the frame
    _, img = cap.read()
    face_locations = face_recognition.face_locations(img)
    print(face_locations)
    """if len(face_locations)!=0:
        test_image = face_recognition.load_image_file(img)
        face_locations= face_recognition.face_locations(test_image)
        face_encodings= face_recognition.face_encodings(test_image, face_locations)

        pil_image= Image.fromarray(test_image)

        draw= ImageDraw.Draw(pil_image)

        for(top,right,bottom,left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name= "Unknown Person"
            if True in  matches:
                first_match_index= matches.index(True)
                name=known_face_names[first_match_index]
        
            draw.rectangle(((left,top),(right,bottom)),outline=(0,0,0))
    
            text_width,text_height = draw.textsize(name)
            draw.rectangle(((left,bottom - text_height - 10),(right,bottom)),fill=(0,0,0),outline=(0,0,0))
            draw.text((left+6,bottom-text_height - 5), name, fill=(255,255,255,255))
    
        del draw

        pil_image.show()
        print(name)"""
    #pil_image= Image.fromarray(img)
    #draw= ImageDraw.Draw(pil_image)
    """detector = FER(mtcnn=True)
    detector.detect_emotions(img)
    emotion, score = detector.top_emotion(img)
    print(emotion,score)"""
    #demography = DeepFace.analyze(img)
    #demography['dominant_emotion']
    obj = DeepFace.analyze(img_path = "img", actions = ['age', 'gender', 'race', 'emotion'])
        
    

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    #Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
        
# Release the VideoCapture object
cap.release()


# In[1]:


import deepface


# In[2]:


from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
img_path = 'Kundana_Lal.jpeg'
img = cv2.imread(img_path)
plt.imshow(img[:, :,::-1])


# In[3]:


demography = DeepFace.analyze(img_path)
demography


# In[21]:


from deepface import DeepFace
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
DeepFace.stream(db_path = "C://Users//lalga//OneDrive//Desktop//database",detector_backend = backends[4])


# RetinaFace and MTCNN seem to overperform in detection and alignment stages but they are much slower. If the speed of your pipeline is more important, then you should use opencv or ssd. On the other hand, if you consider the accuracy, then you should use retinaface or mtcnn.

# In[ ]:


import plotly.express as px
import pandas as pd
df = pd.DataFrame(dict(
    r=[1, 5, 2, 2, 3],
    theta=['neutral','sad','chemical stability',
           'thermal stability', 'device integration']))
fig = px.line_polar(df, r='r', theta='theta', line_close=True)
fig.update_traces(fill='toself')
fig.show()


# In[22]:


# Human pose estimator


import cv2
import mediapipe as mp


# In[23]:


# initialize Pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


# In[30]:


# create capture object
cap = cv2.VideoCapture('Amy.mp4')

while cap.isOpened():
    # read frame from capture object
    _, frame = cap.read()

    try:
      # convert the frame to RGB format
      RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # process the RGB frame to get the result
    results = pose.process(RGB)
    print(results.pose_landmarks)


# In[25]:


# draw detected skeleton on the frame
    mp_drawing.draw_landmarks(
      frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # show the final output
    cv2.imshow('Output', frame)
except:
      break
if cv2.waitKey(1) == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()


# #second pose estimation code

# In[7]:


import cv2
import mediapipe as mp
import time


# In[9]:


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)
pTime = 0


    
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    #print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            print(id, lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)


# In[2]:


import cv2
import mediapipe as mp
import time
import cv2
from fer import Video
from fer import FER 
import numpy as np
import face_recognition
from PIL import Image
from PIL import ImageDraw


# In[3]:




#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def emotionandfacedetection():
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #Display
    cv2.imshow('img', img)

    #emotion
    detector = FER(mtcnn=True)
    detector.detect_emotions(img)
    emotion, score = detector.top_emotion(img)
    print(emotion,score)

def pose():
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils
    pTime = 0
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    #print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            print(id, lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
    cv2.imshow("Image", img)
while True:
    _ ,img = cap.read()
    for i in range(1):
        emotionandfacedetection()
        pose()
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    


# In[1]:


import cv2
cap = cv2.VideoCapture(0)
while True:
    _ ,img=cap.read()
    cv2.imshow("Image",img)
    cv2.waitKey(1)


# # Combine faceandemo with pose

# In[2]:


import cv2
import mediapipe as mp
import time
import cv2
from fer import Video
from fer import FER 
import numpy as np
import face_recognition
from PIL import Image
from PIL import ImageDraw


# In[5]:




#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    _ ,img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces)>1:
        for i in faces:
            detector = FER(mtcnn=True)
            detector.detect_emotions(img)
            emotion, score = detector.top_emotion(img)
            print(emotion,score,i)
            
    # Draw the rectangle around each face
    for (a, b, c, d) in faces: #x,y,h,w
        cv2.rectangle(img, (a, b), (a+c, b+d), (255, 0, 0), 2)

    #emotion
    detector = FER(mtcnn=True)
    detector.detect_emotions(img)
    emotion, score = detector.top_emotion(img)
    print(emotion,score)
    
    #pose
    mediapipePose = mp.solutions.pose
    pose = mpPose.Pose()
    mediapipeDraw = mp.solutions.drawing_utils
    pTime = 0
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(imgRGB)
    #print(results.pose_landmarks)
    if result.pose_landmarks:
        mediapipeDraw.draw_landmarks(img, result.pose_landmarks, mediapipePose.POSE_CONNECTIONS)
        for id, lm in enumerate(result.pose_landmarks.landmark):
            height,width,c = img.shape
            #print(id, lm)
            cx, cy = int(lm.x*width), int(lm.y*height)
            cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
    
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)


# # second code

# In[6]:


import cv2
import mediapipe as mp
import time
import cv2
from fer import Video
from fer import FER 
import numpy as np
import face_recognition
from PIL import Image
from PIL import ImageDraw
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt


# In[5]:



cap = cv2.VideoCapture(0)
import time

b = True #declare boolean so that code can be executed only if it is still True
t1 = time.time()
answer = input("Question")
t2 = time.time()
t = t2 - t1
if t > 15:
  print("You have run out of time!")
  b = False

if b == True:
    backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
    DeepFace.stream(db_path = "C://Users//lalga//OneDrive//Desktop//database",detector_backend = backends[5])
while True:
    _ ,img = cap.read()
    

    
    
    #pose
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils
    pTime = 0
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    #print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            print(id, lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
    
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)


# In[8]:


import time

b = True #declare boolean so that code can be executed only if it is still True
t1 = time.time()
answer = input("Question")
t2 = time.time()
t = t2 - t1
if t > 15:
  print("You have run out of time!")
  b = False

if b == True:
    print(answer)


# In[17]:


import cv2
import mediapipe as mp
import time


class PoseDetector:

    def __init__(self, mode = False, upBody = False, smooth=True, detectionCon = 0.5, trackCon = 0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        #print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def getPosition(self, img, draw=True):
        lmList= []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.getPosition(img)
        print(lmList)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()


# In[ ]:


import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()


# In[16]:



def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.

    '''

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle == 0:

        # Add 360 to the found angle.
        angle += 360
    
    # Return the calculated angle.
    return angle


# # Code Execution: Success

# In[6]:


import cv2
import mediapipe as mp
import time
import cv2
import mediapipe as mp
import time
import cv2
from fer import Video
from fer import FER 
import numpy as np
import face_recognition
from PIL import Image
from PIL import ImageDraw
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

class face:
    def facedetect(x,img):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (a, b, c, d) in faces: #x,y,h,w
            cv2.rectangle(img, (a, b), (a+c, b+d), (255, 0, 0), 2)
    def facerecog(x,img):
        image_of_gahan =  face_recognition.load_image_file('GahanLal.jpg')
        gahan_face_encoding = face_recognition.face_encodings(image_of_gahan)[0]
        known_face_encoding= [gahan_face_encoding]
        known_face_names=["Gahan Lal"]
        test_image = face_recognition.load_image_file('family1.jpg')
        face_locations= face_recognition.face_locations(test_image)
        face_encodings= face_recognition.face_encodings(test_image, face_locations)

        pil_image= Image.fromarray(test_image)

        draw= ImageDraw.Draw(pil_image)

        for(top,right,bottom,left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name= "Unknown Person"
            if True in  matches:
                first_match_index= matches.index(True)
                name=known_face_names[first_match_index]
        
            draw.rectangle(((left,top),(right,bottom)),outline=(0,0,0))
        
            text_width,text_height = draw.textsize(name)
            draw.rectangle(((left,bottom - text_height - 10),(right,bottom)),fill=(0,0,0),outline=(0,0,0))
            draw.text((left+6,bottom-text_height - 5), name, fill=(255,255,255,255))
class emotiondetector:
    def detectemo(x,img):
        detector = FER(mtcnn=True)
        detector.detect_emotions(img)
        emotion, score = detector.top_emotion(img)
        return emotion
        
       
class PoseDetector:
    def getpose(x,img):
        mpPose = mp.solutions.pose
        pose = mpPose.Pose()
        mpDraw = mp.solutions.drawing_utils
        pTime = 0
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        #print(results.pose_landmarks)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h,w,c = img.shape
                print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        

        # Plot Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        """right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])  
        return right_elbow_angle"""

        

        #cv2.putText(img, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
        
def main():
    cap = cv2.VideoCapture(0)
    #pTime = 0
    
    fdetector= face()
    emodetector = emotiondetector()
    detector = PoseDetector()
    
    while True:
        _, img = cap.read()
        fdetector.facedetect(img)
        x=emodetector.detectemo(img)
        detector.getpose(img)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,x , (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()


# In[39]:


# Python program to write
# text on video
  
  
import cv2
  
  
cap = cv2.VideoCapture(0)
  
while(True):
      
    # Capture frames in the video
    ret, frame = cap.read()
  
    # describe the type of font
    # to be used.
    font = cv2.FONT_HERSHEY_SIMPLEX
  
    # Use putText() method for
    # inserting text on video
    cv2.putText(frame, 
                'TEXT ON VIDEO', 
                (50, 50), 
                font, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
  
    # Display the resulting frame
    cv2.imshow('video', frame)
  
    # creating 'q' as the quit 
    # button for the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# release the cap object
cap.release()
# close all windows
cv2.destroyAllWindows()


# In[10]:


import fer
from fer import FER
img=("GahanLal.jpg")
class emotiondetector:
    def detectemo(x,img):
        detector = FER(mtcnn=True)
        detector.detect_emotions(img)
        emotion, score = detector.top_emotion(img)
        print(emotion,score)
        y = emotion
        return y
if 


# In[8]:




def classifyPose(landmarks, output_image, display=False):
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.
 
    '''
    
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'
 
    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    # Get the angle between the left elbow, shoulder and hip points. 
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
 
    # Get the angle between the right hip, shoulder and elbow points. 
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
 
    # Get the angle between the left hip, knee and ankle points. 
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
 
    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
   """ #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the warrior II pose or the T pose.
    # As for both of them, both arms should be straight and shoulders should be at the specific angle.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the both arms are straight.
    if left_elbow_angle == 165 and left_elbow_angle == 195 and right_elbow_angle == 165 and right_elbow_angle == 195:
 
        # Check if shoulders are at the required angle.
        if left_shoulder_angle == 80 and left_shoulder_angle == 110 and right_shoulder_angle == 80 and right_shoulder_angle &lt; 110:
 
    # Check if it is the warrior II pose.
    #----------------------------------------------------------------------------------------------------------------
 
            # Check if one leg is straight.
            if left_knee_angle == 165 and left_knee_angle == 195 or right_knee_angle == 165 and right_knee_angle == 195:
 
                # Check if the other leg is bended at the required angle.
                if left_knee_angle &gt; 90 and left_knee_angle &lt; 120 or right_knee_angle &gt; 90 and right_knee_angle &lt; 120:
 
                    # Specify the label of the pose that is Warrior II pose.
                    label = 'Warrior II Pose' 
                        
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the T pose.
    #----------------------------------------------------------------------------------------------------------------
    
            # Check if both legs are straight
            if left_knee_angle &gt; 160 and left_knee_angle &lt; 195 and right_knee_angle &gt; 160 and right_knee_angle &lt; 195:
 
                # Specify the label of the pose that is tree pose.
                label = 'T Pose'
 
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the tree pose.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if one leg is straight
    if left_knee_angle &gt; 165 and left_knee_angle &lt; 195 or right_knee_angle &gt; 165 and right_knee_angle &lt; 195:
 
        # Check if the other leg is bended at the required angle.
        if left_knee_angle &gt; 315 and left_knee_angle &lt; 335 or right_knee_angle &gt; 25 and right_knee_angle &lt; 45:
 
            # Specify the label of the pose that is tree pose.
            label = 'Tree Pose'
                
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    # Check if the resultant image is specified to be displayed.
    if display:
    
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        
        # Return the output image and the classified label.
        return output_image, label"""


# In[7]:


image = cv2.imread('m.jpg')
output_image, landmarks = detectPose(image, pose, display=False)
if landmarks:
    classifyPose(landmarks, output_image, display=True)


# In[24]:



def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Otherwise
    else:
        
        # Return the output image and the found landmarks.
        return output_image, landmarks


# In[25]:


import cv2
import mediapipe as mp
import time
import cv2
import mediapipe as mp
import time
import cv2
from fer import Video
from fer import FER 
import numpy as np
import face_recognition
from PIL import Image
from PIL import ImageDraw
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils


pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)


# Initialize the VideoCapture object to read from the webcam.
#video = cv2.VideoCapture(0)
 
# Initialize the VideoCapture object to read from a video stored in the disk.
video = cv2.VideoCapture("standing.jpg")
 
 
# Initialize a variable to store the time of the previous frame.
time1 = 0
 
# Iterate until the video is accessed successfully.
while video.isOpened():
    
    # Read a frame.
    ok, frame = video.read()
    
    # Check if frame is not read properly.
    if not ok:
        
        # Break the loop.
        break
    
    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)
    
    # Get the width and height of the frame
    frame_height, frame_width, _ =  frame.shape
    
    # Resize the frame while keeping the aspect ratio.
    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
    
    # Perform Pose landmark detection.
    frame, _ = detectPose(frame, pose_video, display=False)
    
    # Set the time for this frame to the current time.
    time2 = time()
    
    # Check if the difference between the previous and this frame time &gt; 0 to avoid division by zero.
    if (time2 - time1) == 0:
    
        # Calculate the number of frames per second.
        frames_per_second = 1.0 / (time2 - time1)
        
        # Write the calculated number of frames per second on the frame. 
        cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    
    # Update the previous frame time to this frame time.
    # As this frame will become previous frame in next iteration.
    time1 = time2
    
    # Display the frame.
    cv2.imshow('Pose Detection', frame)
    
    # Wait until a key is pressed.
    # Retreive the ASCII code of the key pressed
    k = cv2.waitKey(1) == 0xFF
    
    # Check if 'ESC' is pressed.
    if(k == 27):
        
        # Break the loop.
        break
 
# Release the VideoCapture object.
video.release()
 
# Close the windows.
cv2.destroyAllWindows()


# # Pose classification

# In[5]:



import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils
  
cap = cv2.VideoCapture(0)
while True:
    _,img= cap.read()
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
 
    # Check if any landmarks are found.
    if results.pose_landmarks:
    
        # Iterate two times as we only want to display first two landmarks.
        for i in range(2):
        
            # Display the found normalized landmarks.
            print(f'{mp_pose.PoseLandmark(i).name}:\n{results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]}') 

    # Create a copy of the sample image to draw landmarks on.
    img_copy = img.copy()
 
    # Check if any landmarks are found.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the sample image.
        mp_drawing.draw_landmarks(image=img_copy, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
       
        # Specify a size of the figure.
        fig = plt.figure(figsize = [10, 10])
 
        # Display the output image with the landmarks drawn, also convert BGR to RGB for display. 
        plt.title("Output");plt.axis('off');plt.imshow(img_copy[:,:,::-1]);plt.show()
        
    
    # Plot Pose landmarks in 3D.
    mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)


# In[7]:





def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Otherwise
    else:
        
        # Return the output image and the found landmarks.
        return output_image, landmarks

cap= cv2.VideoCapture(0)
while True:
    _, img= cap.read()
    # Read another sample image and perform pose detection on it.
    detectPose(img, pose, display=True)


# In[12]:


import time
import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt

def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Otherwise
    else:
        
        # Return the output image and the found landmarks.
        return output_image, landmarks

def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.
 
    '''
 
    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
 
    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle == 0:
 
        # Add 360 to the found angle.
        angle += 360
    
    # Return the calculated angle.
    return angle


def classifyPose(landmarks, output_image, display=False):
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.
 
    '''
    
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'
 
    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    # Get the angle between the left elbow, shoulder and hip points. 
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
 
    # Get the angle between the right hip, shoulder and elbow points. 
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
 
    # Get the angle between the left hip, knee and ankle points. 
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
 
    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the warrior II pose or the T pose.
    # As for both of them, both arms should be straight and shoulders should be at the specific angle.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the both arms are straight.
    if left_elbow_angle == 165 and left_elbow_angle == 195 and right_elbow_angle == 165 and right_elbow_angle == 195:
 
        # Check if shoulders are at the required angle.
        if left_shoulder_angle == 80 and left_shoulder_angle == 110 and right_shoulder_angle == 80 and right_shoulder_angle == 110:
 
    # Check if it is the warrior II pose.
    #----------------------------------------------------------------------------------------------------------------
 
            # Check if one leg is straight.
            if left_knee_angle == 165 and left_knee_angle == 195 or right_knee_angle == 165 and right_knee_angle == 195:
 
                # Check if the other leg is bended at the required angle.
                if left_knee_angle == 90 and left_knee_angle == 120 or right_knee_angle == 90 and right_knee_angle == 120:
 
                    # Specify the label of the pose that is Warrior II pose.
                    label = 'Warrior II Pose' 
                        
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the T pose.
    #----------------------------------------------------------------------------------------------------------------
    
            # Check if both legs are straight
            if left_knee_angle == 160 and left_knee_angle == 195 and right_knee_angle == 160 and right_knee_angle == 195:
 
                # Specify the label of the pose that is tree pose.
                label = 'T Pose'
 
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the tree pose.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if one leg is straight
    if left_knee_angle == 165 and left_knee_angle == 195 or right_knee_angle == 165 and right_knee_angle == 195:
 
        # Check if the other leg is bended at the required angle.
        if left_knee_angle == 315 and left_knee_angle == 335 or right_knee_angle == 25 and right_knee_angle == 45:
 
            # Specify the label of the pose that is tree pose.
            label = 'Tree Pose'
                
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    # Check if the resultant image is specified to be displayed.
    if display:
    
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        
        # Return the output image and the classified label.
        return output_image, label
    
# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils
  

# Setup Pose function for video.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
 
# Initialize the VideoCapture object to read from the webcam.
video = cv2.VideoCapture(0)
 
# Initialize the VideoCapture object to read from a video stored in the disk.
#video = cv2.VideoCapture('media/running.mp4')
 
 
# Initialize a variable to store the time of the previous frame.
time1 = 0
 
# Iterate until the video is accessed successfully.
while video.isOpened():
    _, img = video.read()
    """# Read a frame.
    ok, frame = video.read()
    
    # Check if frame is not read properly.
    if not ok:
        
        # Break the loop.
        break
    
    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)
    
    # Get the width and height of the frame
    frame_height, frame_width, _ =  frame.shape
    
    # Resize the frame while keeping the aspect ratio.
    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
    
    # Perform Pose landmark detection.
    frame, _ = detectPose(frame, pose_video, display=False)
    
    # Set the time for this frame to the current time.
    time2 = time()
    
    # Check if the difference between the previous and this frame time &gt; 0 to avoid division by zero.
    if (time2 - time1) == 0:
    
        # Calculate the number of frames per second.
        frames_per_second = 1.0 / (time2 - time1)
        
        # Write the calculated number of frames per second on the frame. 
        cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    
    # Update the previous frame time to this frame time.
    # As this frame will become previous frame in next iteration.
    time1 = time2
    
    # Display the frame.
    cv2.imshow('Pose Detection', frame)
    
    # Wait until a key is pressed.
    # Retreive the ASCII code of the key pressed
    k = cv2.waitKey(1) == 0xFF
    
    # Check if 'ESC' is pressed.
    if(k == 27):
        
        # Break the loop.
        break
 
# Release the VideoCapture object.
video.release()
 
# Close the windows.
cv2.destroyAllWindows()"""
    output_image, landmarks = detectPose(img, pose, display=False)
    plt.figure(figsize=[10,10])
    if landmarks:
        classifyPose(landmarks, output_image, display=True)


# In[10]:



# Read a sample image and perform pose classification on it.

output_image, landmarks = detectPose(img, pose, display=False)
if landmarks:
    classifyPose(landmarks, output_image, display=True)


# In[11]:


import mediapipe as mp
import cv2
import time
import numpy as np
import pandas as pd
import os
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils # For drawing keypoints
points = mpPose.PoseLandmark # Landmarks
path = "C:\\Users\\lalga\\OneDrive\\Documents\\standing" # enter dataset path
data = []
for p in points:
        x = str(p)[13:]
        data.append(x + "_x")
        data.append(x + "_y")
        data.append(x + "_z")
        data.append(x + "_vis")
data = pd.DataFrame(columns = data) # Empty dataset
count = 0

for img in os.listdir(path):

        temp = []

        img = cv2.imread(path + "/" + img)

        imageWidth, imageHeight = img.shape[:2]

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        blackie = np.zeros(img.shape) # Blank image

        results = pose.process(imgRGB)

        if results.pose_landmarks:

                # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS) #draw landmarks on image

                mpDraw.draw_landmarks(blackie, results.pose_landmarks, mpPose.POSE_CONNECTIONS) # draw landmarks on blackie

                landmarks = results.pose_landmarks.landmark

                for i,j in zip(points,landmarks):

                        temp = temp + [j.x, j.y, j.z, j.visibility]

                data.loc[count] = temp

                count +=1

        cv2.imshow("Image", img)

        cv2.imshow("blackie",blackie)

        cv2.waitKey(100)

data.to_csv("dataset3.csv") # save the data as a csv file
from sklearn.svm import SVC
data = pd.read_csv("dataset3.csv")
X,Y = data.iloc[:,:-1],data['target']
model = SVC(kernel = 'poly')
model.fit(X,Y)
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
path = "C:\\Users\\lalga\\OneDrive\\Documents\\standing\\standing.jpg"
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = pose.process(imgRGB)
if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        for j in landmarks:
                temp = temp + [j.x, j.y, j.z, j.visibility]
        y = model.predict([temp])
        if y == 0:
            asan = "plank"
        else:
            asan = "goddess"
        print(asan)
        cv2.putText(img, asan, (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),3)
        cv2.imshow("image",img)


# In[1]:


import time
import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import time
import cv2
import mediapipe as mp
import time
import cv2
from fer import Video
from fer import FER 
import numpy as np
import face_recognition
from PIL import Image
from PIL import ImageDraw
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

class face:
    def detect(x,img):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (a, b, c, d) in faces: #x,y,h,w
            cv2.rectangle(img, (a, b), (a+c, b+d), (255, 0, 0), 2)
        
    def count(x,img):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        count=str((len(faces)))
        cv2.putText(img, count, (600, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
        
    def recog(x,img):
        face_locations= face_recognition.face_locations(img)
        face_encodings= face_recognition.face_encodings(img, face_locations)
        name= "Unknown Person"
        


        for(top,right,bottom,left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            #name= "Unknown Person"
            if True in  matches:
                first_match_index= matches.index(True)
                name=known_face_names[first_match_index]
            
    
            
        cv2.putText(img,name,(25,50),font,1,(0, 255, 255), 2, cv2.LINE_4)
    
    def emotion(x,img):
        detector = FER(mtcnn=True)
        detector.detect_emotions(img)
        emotion, score = detector.top_emotion(img)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,emotion , (25, 100), font, 1, (0, 255, 255), 2, cv2.LINE_4)
        
        
def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Otherwise
    else:
        
        # Return the output image and the found landmarks.
        return output_image, landmarks


def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.
 
    '''
 
    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
 
    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle == 0:
 
        # Add 360 to the found angle.
        angle += 360
    
    # Return the calculated angle.
    return angle

    
def classifyPose(landmarks, output_image, display=False):
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.
 
    '''
    
        # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown'
 
        # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    # Get the angle between the left elbow, shoulder and hip points. 
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
 
    # Get the angle between the right hip, shoulder and elbow points. 
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
 
    # Get the angle between the left hip, knee and ankle points. 
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
 
    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    if left_knee_angle >= 170 and left_knee_angle<=185 and right_knee_angle >= 170 and right_knee_angle <=185:
        label="Standing"
    if label != 'Unknown Pose':
        
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    
    
    # Check if the resultant image is specified to be displayed.
    """if display:
    
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');"""
        
    #else:
        
        # Return the output image and the classified label.
    #return label
    cv2.putText(img, label, (25, 150),font, 1, (0, 255, 255), 2, cv2.LINE_4)

################################################################################################################################

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils

#faces
image_of_gahan =  face_recognition.load_image_file('GahanLal.jpg')
gahan_face_encoding = face_recognition.face_encodings(image_of_gahan)[0]
image_of_mukul = face_recognition.load_image_file('Mukul.jfif')
mukul_face_encoding = face_recognition.face_encodings(image_of_mukul)[0]
known_face_encoding= [gahan_face_encoding,mukul_face_encoding]
known_face_names=["Gahan Lal","Mukul"]
#"Gahan Lal",
#gahan_face_encoding
#font
font = cv2.FONT_HERSHEY_SIMPLEX

# Read another sample image and perform pose classification on it.
cap=cv2.VideoCapture(0)
fdetector= face()
while True:
    _,img = cap.read()
    #fdetector.detect(img)
    fdetector.recog(img)
    fdetector.count(img)
    fdetector.emotion(img)
    output_image, landmarks = detectPose(img, pose, display=False)
    if landmarks:
        classifyPose(landmarks, output_image, display=True)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)


# In[2]:


import cv2
cap= cv2.VideoCapture(0)
while True:
    _,img = cap.read()
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (a, b, c, d) in faces: #x,y,h,w
        cv2.rectangle(img, (a, b), (a+c, b+d), (255, 0, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    count=str((len(faces)))
    
    cv2.putText(img, count, (50, 100), font, 1, (0, 255, 255), 2, cv2.LINE_4)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    

    
    


# In[ ]:


import cv2
import face_recognition
import PIL
from PIL import Image
from PIL import ImageDraw

image_of_gahan =  face_recognition.load_image_file('GahanLal.jpg')
gahan_face_encoding = face_recognition.face_encodings(image_of_gahan)[0]
known_face_encoding= [gahan_face_encoding]
known_face_names=["Gahan Lal"]

cap= cv2.VideoCapture(0)
while True:
    _,img=cap.read()
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, h, w) in faces: #x,y,h,w
        image = cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 1)
        cv2.putText(image, 'F', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    """face_locations= face_recognition.face_locations(img)
    face_encodings= face_recognition.face_encodings(img, face_locations)
    name= "Unknown Person"

    for(top,right,bottom,left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        
        if True in  matches:
            first_match_index= matches.index(True)
            name=known_face_names[first_match_index]"""
            # For bounding box
            img = cv2.rectangle(img, (top, left), (bottom, right), (255, 0, 0), 2)
 
            # For the text background
            # Finds space required by the text so that we can put a background with that amount of width.
            (w, h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            # Prints the text.    
            img = cv2.rectangle(img, (top, bottom - 20), (top + w, bottom), (255, 0, 0), -1)
            img = cv2.putText(img, name, (top, bottom - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)

            # For printing text
            img = cv2.putText(img, name, (top, bottom),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            
            
            #image = cv2.rectangle(img, (top, right), (top + bottom, right + left), (36,255,12), 1)
            #cv2.putText(image, name, (top, right-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
        
    #print(name)
    cv2.imshow("image",img)
    cv2.waitKey(1)


# In[ ]:




