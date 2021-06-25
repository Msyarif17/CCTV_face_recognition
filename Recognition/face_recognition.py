import cv2
import numpy as np
import os 
import mysql.connector
from datetime import date,datetime


# DB
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="face_recognition"
)
sql = ""
mycursor = mydb.cursor()


# init

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('E:/Informatika 2019/Semester 4/Teori Bahasa dan Otomata/OpenCV-Face-Recognition/trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX
data = []
today = date.today()
now = datetime.now()
#menyimpan video
current_time = now.strftime("%H:%M:%S")
out = cv2.VideoWriter('E:/Informatika 2019/Semester 4/Teori Bahasa dan Otomata/OpenCV-Face-Recognition/%s-%s.avi'%(str(today),current_time), cv2.VideoWriter_fourcc(*'MPEG'), 20.0, (640,480))
#sql
id = 0
unknown = False
mycursor.execute("SELECT * FROM faces ")
           
faceU = mycursor.fetchall()
names = [s[1] for s in faceU]
nim = [s[2] for s in faceU]
fakultas = [s[3] for s in faceU]
jurusan = [s[4] for s in faceU]
 
mycursor.execute("SELECT count(Nama) FROM faces ")
cUser = mycursor.fetchall()
countUser = [s[0] for s in cUser]
# video capture
cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(3, 640) 
cam.set(4, 480) 

count = 0
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img =cam.read()
    out.write(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    
    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        
        if (confidence < 100):
            data = [str(names[id-2]),str(nim[id-2]),str(fakultas[id-2]),str(jurusan[id-2])]
            id = names[id-2]
            confidence = "  {0}%".format(round(100 - confidence))
            
        else:
            id = "unknown"
            unknown = True
            confidence = "  {0}%".format(round(100 - confidence))
           
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2) 
                 

            # Save the captured image into the datasets folder
            cv2.imwrite("E:/Informatika 2019/Semester 4/Teori Bahasa dan Otomata/OpenCV-Face-Recognition/dataset/User." + str(0) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

            cv2.imshow('image', img)
        count +=1
        
        cv2.putText(img, "Nama : "+str(id), (x+w+5,y+10), font, 0.5, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 2)  
        cv2.putText(img, "NIM : "+str(data[1]), (x+w+5,y+50), font, 0.5, (0,255,255), 1)  
        cv2.putText(img, "Fakultas : "+str(data[2]), (x+w+5,y+90), font, 0.5, (0,255,255), 1)  
        cv2.putText(img, "Jurusan : "+str(data[3]), (x+w+5,y+130), font, 0.5, (0,255,255), 1)  
         
    
    cv2.imshow('camera',img) 
    
    k = cv2.waitKey(10) & 0xff 
    if k == 27:
        break

if(unknown) :
    os.system('python face_training.py')
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
out.release()
cv2.destroyAllWindows()
