import cv2
import os,sys
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="face_recognition"
)
sql = ""
mycursor = mydb.cursor()

cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mycursor.execute("SELECT count(Nama) FROM faces ")
cUser = mycursor.fetchall()

# For each person, enter one numeric face id
face_id = [s[0] for s in cUser] + 1
nama = input('nama : ')
nim = input('nim : ')
fc = input('fakultas : ')
jurusan = input('jurusan : ')
sql = "INSERT INTO faces (Nama, NIM, Fakultas, Jurusan) VALUES (%s,%s,%s, %s)"
val = (nama, nim,fc,jurusan)
mycursor.execute(sql, val)

mydb.commit()

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read()
    # img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("E:/Informatika 2019/Semester 4/Teori Bahasa dan Otomata/OpenCV-Face-Recognition/dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


