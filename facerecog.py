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

    # Detect the faces
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


# In[4]:


import cv2
from fer import Video
from fer import FER 
import numpy as np
import face_recognition
from PIL import Image
from PIL import ImageDraw


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
    detector = FER(mtcnn=True)
    detector.detect_emotions(img)
    emotion, score = detector.top_emotion(img)
    print(emotion,score)
        
    

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


# In[ ]:





# In[ ]:





# In[ ]:




