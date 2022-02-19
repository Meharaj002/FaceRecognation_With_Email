from operator import truediv
from pickle import TRUE
import cv2
import numpy as np
import face_recognition
import os
import time as tm
from datetime import datetime

import smtplib

path=('E:\\Personal Project\\Face_identification\\Face_ID001_Basic\\Attendece_test_PG\\train images')     #train image 'file' path

images=[]
ClassName=[]
Mylist= os.listdir(path)
#print(Mylist)

pTime=0
cTime=0

for ClNm in Mylist:
    CurIm = cv2.imread(f'{path}/{ClNm}')
    images.append(CurIm)
    ClassName.append(os.path.splitext(ClNm)[0])
#print(ClassName)

def duplicates(lst, item):
  return [i for i, x in enumerate(lst) if x == item]

def FindEncoding(images):
    encodings=[]
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodings.append(encode)
    return encodings


def AutoMateEmail(UnKnown=False):
    if UnKnown==True:
        Server=smtplib.SMTP("smtp.gmail.com",587)
        Server.starttls()
        Server.login('projectdurjo@gmail.com','ProjectDurjo@101')
        Server.sendmail('projectdurjo@gamil.com',
                        'mssiam12@gmail.com',
                        'messageUnknown Person Detected in Room')  #from, to, message
        print('1st Sended')

        Server.sendmail('projectdurjo@gamil.com',
                        'md.shaeak.ibna.salim@g.bracu.ac.bd',
                        'messageUnknown Person Detected in Room')  #from, to, message
        print('2nd Sended')

        Server.sendmail('projectdurjo@gamil.com',
                        'tasmia.azrine@g.bracu.ac.bd',
                        'messageUnknown Person Detected in Room')  #from, to, 
        print('3rd Sended')

        Server.quit()

def MakeAttencence(name,unknown=False):
    with open('E:\\Personal Project\\Face_identification\\Face_ID001_Basic\\Attendece_test_PG\\Attendence.csv', 'r+') as f:
        
        MyDataList=f.readlines()
        #print(MyDataList)
        NameList=[]
        TimeList=[]
        for line in MyDataList:
            entry=line.split(',')
            NameList.append(entry[0])
            TimeList.append(entry[1])
        #print(TimeList)
        if name not in NameList:
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
        if name in NameList:
            EnrollSerial=duplicates(NameList,name)
            LstEnrollSerial=EnrollSerial[-1]
            #print(LstEnrollSerial)
            LstEnrollTime=TimeList[LstEnrollSerial]
            if '\n'in LstEnrollTime:
                LstEnrollTime=LstEnrollTime.replace('\n','')
            #print('>>>>LstEnrollTime >>>>',LstEnrollTime,'>>>>',type(LstEnrollTime))
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            #print('1>>>',dtString,'>>>',type(dtString))
            Format='%H:%M:%S'
            #preEnrollTime=dtString.split(':')
            PTime=datetime.strptime(dtString, Format)
            LETime=datetime.strptime(LstEnrollTime, Format)
            Interval = PTime - LETime
            IntvlTimeLst=str(Interval)
            IntvlTimeLst=IntvlTimeLst.split(':')
            #print('2>>>',IntvlTimeLst,'>>>',type(IntvlTimeLst))
            if int(IntvlTimeLst[1])>=0 and int(IntvlTimeLst[2])>=30:
                #print('***Enrolled***')
                now=datetime.now()
                dtString=now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')
                if unknown==True:
                    AutoMateEmail(True)
            

encodeListKnown= FindEncoding(images)
print('Encoding Completed')

cap=cv2.VideoCapture(0)

while True:
    sucess, img= cap.read()
    img=cv2.flip(img, 1)
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    faceCurFrame=face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis= face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        MatchIndex= np.argmin(faceDis)

        if matches[MatchIndex]:
            name= ClassName[MatchIndex].upper()
            print (name)
            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            MakeAttencence(name)
        else:
            name= "UnKnown"
            print (name)
            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            MakeAttencence(name,True)
            
    
    cTime=tm.time()
    fps= 1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'FPS:{int(fps)}', (40,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0),2)

    cv2.imshow('WebCam', img)
    key= cv2.waitKey(1)
    if key== ord('q'):
        break

cap.release()
cv2.destroyAllWindows()