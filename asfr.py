import cv2
import pandas as pd
import numpy as np
import os
import tkinter as tk
import csv
import time
import datetime
import pickle
from PIL import Image



root = tk.Tk()
root.title('Attendance System Using Face Recognition')

lbl = tk.Label(root, text ='Student Id')
lbl.place(x = 100, y = 20)

txt = tk.Entry(root, width = 20)
txt.place(x = 200, y = 20)

lbl2 = tk.Label(root, text ='Student Name')
lbl2.place(x = 100, y = 50)

txt2 = tk.Entry(root, width = 20)
txt2.place(x = 200, y = 50)

def take_images():
    Id = (txt.get())
    Name = (txt2.get())


    if(Id.isdigit() and Name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = 'D:\Github\Attendance-System-Facial-Recognition\Code\cascade\data\haarcascade_frontalface_default.xml'
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
                sampleNum = sampleNum + 1
                if not os.path.exists('Training_Images/'+Name):
                    os.makedirs('Training_Images/' + Name)
                path = 'Training_Images/' + Name + '/'
                #os.path.join(path,str(sampleNum))
                cv2.imwrite(path + str(sampleNum) + '.jpg', gray[y:y+h, x:x+w])
                #cv2.imwrite('Training_Images\Name\ ' str(sampleNum) + '.jpg', gray[y:y+h, x:x+w])
                #cv2.imwrite('Training_Images\ ' +Name +'.' +Id +'.' + str(sampleNum) + '.jpg', gray[y:y+h, x:x+w])
                cv2.imshow('frame', img)
                #else:
                    #cv2.imwrite('Training_Images\ ' +Name+'.'+Id+'.'+str(sampleNum) + '.jpg', gray[y:y+h, x:x+w])
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

            elif sampleNum > 60:
                break

        cam.release()
        cv2.destroyAllWindows()
        #res = "Images saved for SRN :" + Id + "Name" + Name
        row = [Id, Name]
        with open('Student_Details\Student_Details.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()

    else:
        print('Either SRN or Name is invalid')


def train_images():

    #harcascadePath = 'haarcascade_frontalface_alt2.xml'
    #face_cascade = cv2.CascadeClassifier(harcascadePath)
    #harcascadePath = 'haarcascade_frontalface_alt2.xml'
    #detector = cv2.CascadeClassifier(harcascadePath)

    face_cascade = cv2.CascadeClassifier('D:\Github\Attendance-System-Facial-Recognition\Code\cascade\data\haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()


    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []



    for root, dirs, files in os.walk('Training_Images'):
        for file in files:
            #print(file)
            if file.endswith('jpg'):
                path = os.path.join(root, file)
                label = os.path.basename(os.path.dirname(path)).replace(' ','-').lower()
                #print(label, path)
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id +=1
                id_ = label_ids[label]
                #print(label_ids)

                pil_image = Image.open(path).convert('L')
                #size = (550,550)
                #final_image = pil_image.resize(size, Image.ANTIALIAS)
                image_array = np.array(pil_image, 'uint8')
                #print(image_array)
                #faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
                #print(faces)
                #for (x, y, w, h) in faces:
                    #roi = image_array[y:y+h, x:x+w]
                    #print(roi)
                x_train.append(image_array)
                y_labels.append(id_)
    #print(y_labels)
    #print(x_train)

    with open('labels.pickle', 'wb') as f:
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save('TrainingImageLabel/trainner.yml')


def track_images():

    harcascadePath = 'D:\Github\Attendance-System-Facial-Recognition\Code\cascade\data\haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(harcascadePath)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('TrainingImageLabel/trainner.yml')

    labels = {}
    with open('labels.pickle', 'rb') as f:
        labels = pickle.load(f)
        labels = {v:k for k,v in labels.items()}


    df = pd.read_csv('Student_Details\Student_Details.csv')
    #print(df.loc['Id'][0])
    cap = cv2.VideoCapture(0)
    col_names = ['ID', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns = col_names)
    while(True):

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.5, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(roi_gray)
            #print(id_)
            if(conf<70):
                date = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
                print(df,id_)
                get_name = df.loc[id_]['Name']
                print(get_name)
                tt = str(id_)+'-'+get_name
                attendance.loc[len(attendance)] = [id_, get_name, date, timeStamp]
                #print(attendance)
            else:
                Id = 'Unknown'
                tt = str(id_)

            if (conf > 75):
                noOfFile = len(os.listdir('ImagesUnknown')) + 1
                cv2.imwrite('ImagesUnknown/Image'+ str(noOfFile) +'.jpg', frame[y:y+h,x:x+w])




            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        attendance = attendance.drop_duplicates(subset = ['ID'], keep = 'first')
        cv2.imshow('frame', frame)
        if(cv2.waitKey(1) == ord('q')):
            break


    date = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    #print(attendance)
    attendance.to_csv(fileName,index=False)
    cap.release()
    cv2.destroyAllWindows()



takeImg = tk.Button(root, text = "Take Images", command = take_images)
takeImg.place(x = 50, y = 100)

trainImg = tk.Button(root, text = 'Train Images', command = train_images)
trainImg.place(x = 150, y = 100)

trackImg = tk.Button(root, text = 'Track Images', command = track_images)
trackImg.place(x = 250, y = 100)

root.mainloop()
