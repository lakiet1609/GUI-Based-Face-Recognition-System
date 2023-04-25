import cv2
import numpy as np
import PySimpleGUI as sg
import sys
import pandas as pd
import os
import csv
from PIL import Image

data_path = 'data'
data_file = 'database.txt'

register = False
recognizeFrame = False
sampleNum = 20
name = ''
id = ''
course = ''

Faces = []
id = []

recognizer = cv2.face.LBPHFaceRecognizer_create()

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#take image from the webcam - using the haarcascade to find and locate face - save them into data
def createImages(frame,count):
    global data_path,name,id,detector
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
        #Format of the file: name.ID.count.jpg
        cv2.imwrite(data_path + '\\' + name + '.' + id + '.' + str(count) + '.jpg', gray[y:y+h,x:x+w])
    return frame

#Save the student information into the the .txt file
def writedata(data_file,row):
    #Create a student record file if does not exist
    if not os.path.exists(data_file):
        with open(data_file, 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['id','name','course'])
        csvFile.close()
    with open(data_file, 'a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()
    return 'Save to database'

#Get the faces and ids from data - put into list variables
def getimgsandlables(path):
    imgpaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    ids = []
    for imgpath in imgpaths:
        extension = os.path.splitext(imgpath)[-1]
        if extension != '.jpg':
            print(extension)
            continue
        pil_img = Image.open(imgpath).convert('L')
        imagenp = np.array(pil_img,'uint8')
        id = int(os.path.split(imgpath)[-1].split('.')[1])
        faces.append(imagenp)
        ids.append(id)
    return faces, ids

def train(path):
    global recognizer,detector
    Faces, id = getimgsandlables(path)
    recognizer.train(Faces,np.array(id))
    recognizer.save('trainer.yml')
    return 'Training finished...'

#Loading .yml file and testing
def imagetrack(frame):
    global recognizer,detector
    recognizer.read('trainer.yml')
    df = pd.read_csv(data_file, delimiter=',')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    person = ''
    person_to_show = ''
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),1)
        id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        # print('ID: ' + str(id) + ' conf: ' + str(conf))
        # print(df['id'])
    
        if (conf < 50):
            person = df.loc[df['id'] == id]['name'].values
            cr = df.loc[df['id'] == id]['course'].values
            person = str(id) + '-' + np.array2string(person) + '-' + np.array2string(cr)
            # print(person)
        else:
            id = 'Unknown'
            person = str(id)
        person = (str(person)+ ' ' + str(int(conf)) + '%')
        if (conf < 50):
            person_to_show = person
    return frame, person_to_show

def main():
    global register, sampleNum, data_path, name, id, course, recognizeFrame
    sg.change_look_and_feel('LightGreen')

    #define your window layout
    leftpanel = [
        [sg.Text('Face Rcognition', size=(20,1), justification='center', font='Helvetica 20', key='title')],
        [sg.Image(filename='', key='image')]
    ]

    rightpanel = [
        [sg.Text('Id: ',size=(10,1))],
        [sg.InputText('',key='id')],
        [sg.Text('Name: ',size=(10,1))],
        [sg.InputText('',key='name')],
        [sg.Text('Course: ',size=(10,1))],
        [sg.InputText('',key='course')],
        [sg.Button('1. Register', size = (15,1), font='Helvetica 14')],
        [sg.Button('2. Train', size = (15,1), font='Any 14')],
        [sg.Button('3. Recognize', size = (15,1), font='Helvetica 14')]
    ]

    layout = [
        [
        sg.Column(leftpanel),
        sg.VSeparator(),
        sg.Column(rightpanel)
        ]
    ]

    window = sg.Window('Face Recognition System', location=(100,100))
    window.Layout(layout).Finalize()

    info = ''
    cap = cv2.VideoCapture(0)
    while True:
        event, values = window.read(timeout=20, timeout_key='timeout')
        if event == sg.WIN_CLOSED:
            break
        elif event == '1. Register':
            register = True
            count = 0
        elif event == '2. Train':
            info = train(data_path)
        elif event == '3. Recognize':
            recognizeFrame=True
        
        name = values['name']
        id = values['id']
        course = values['course']

        ret,frame = cap.read()
        
        if register:
            createImages(frame,count)
            info = 'Saving ' + str(count)
            count +=1
            if count > sampleNum:
                row = [id,name,course]
                info = writedata(data_file, row)
                register = False
        
        if recognizeFrame:
            frame,info = imagetrack(frame)
        
        imgbytes  = cv2.imencode('.png', frame)[1].tobytes()
        window['image'].update(data=imgbytes)
        window['title'].update(info)
    
    cap.release()
    cv2.destroyAllWindows()
    window.close()

main()