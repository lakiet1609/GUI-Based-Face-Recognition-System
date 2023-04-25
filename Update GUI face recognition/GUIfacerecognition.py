import cv2
import os
import face_recognition
import numpy as np
import PySimpleGUI as sg
import sys
import pandas as pd
import csv

data_path = 'data1'
datafile  = 'database1.txt'

register = False
recogniseFrame = False
sampleNum = 1
name = ''
id = ''
course = ''
faces = []
id = []

def getface(path):
    imagePaths = [os.path.join(f,path) for f in os.listdir(data_path)]
    Faces = []
    IDs = []
    count = 0
    for imgpath in imagePaths:
        extension = os.path.split(imgpath)[-1].split('.')[-1]
        if (extension!='jpg'):
            continue
        face_img = face_recognition.load_image_file(imgpath)
        face_encoding = face_recognition.face_encodings(face_img)
        if len(face_encoding)<1:
            continue
        Faces.append(face_encoding[0])
        ID = os.path.split(imgpath)[-1].split('.')[0]
        IDs.append(ID)
        count+=1
    return Faces, IDs

def createimages(frame,count):
    global data_path,name,id
    # cvt_frame = frame[:,:,::-1]
    faces_location = face_recognition.face_locations(frame)
    faces_encoding = face_recognition.face_encodings(frame,faces_location)

    for (top,right,bottom,left), face_encoding in zip(faces_location, faces_encoding):
        cv2.imwrite(data_path + '\\' + name + '.' + id + '.' + str(count) + '.jpg', frame)
        cv2.rectangle(frame, (left,top), (bottom,right), (255,0,0), 2)
    
    return frame

def writedata(datafile,row):
    if not os.path.exists(datafile):
        with open(datafile, '+a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['id','naame','course'])
        csvFile.close()
    with open(datafile,'+a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()

    return 'Save to database'

#For test image phase
def imagetrack(frame, known_face_name, known_face_encoding):
    # global detector, recognizer
    # cvt_frame = frame[:,:,::-1]
    df = pd.read_csv(datafile, delimiter='.')
    faces_location = face_recognition.face_locations(frame)
    faces_encoding  = face_recognition.face_encodings(frame, faces_location)

    name = 'Unknown'

    for (top, right, bottom, left), face_encoding in zip(faces_location,faces_encoding):
        matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
        if (len(matches)<1):
            continue
        face_distances = face_recognition.face_distance(known_face_name,face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_name[best_match_index]
            cr = df.loc[df['name'] == name]['course'].values
            name = name + '-' + cr[0]
            print(name)

        cv2.rectangle(frame,(top,left), (bottom,right), (255,255,0), 2)
        cv2.putText(frame, name, (top+6,left+6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0.0,255), 2)

    return frame, name

def main():
    global register, sampleNum,data_path,id,course,name,recogniseFrame
    sg.change_look_and_feel('LightGreen')

    leftpanel = [
        [sg.Text('Face Recognition', size=(20,1), justification='center',font='Helvetica 20', key='title')],
        [sg.Image(filename='', key='image')]
    ]

    rightpanel = [
        [sg.Text('Id', size=(10,1))],
        [sg.InputText('', key='id')],
        [sg.Text('Name', size=(10,1))],
        [sg.InputText('', key='name')],
        [sg.Text('Course', size=(10,1))],
        [sg.InputText('', key='course')],
        [sg.Button('1. Register', size=(15,1), font='Helvetica 14')],
        [sg.Button('2. Update', size=(15,1), font='Helvetica 14')],
        [sg.Button('3. Recognize', size=(15,1), font='Helvetica 14')],
    ]

    layout = [
        [
        sg.Column(leftpanel),
        sg.VSeparator(),
        sg.Column(rightpanel)
        ]
    ]

    window = sg.Window('Face Recognition', location=(100,100))
    window.Layout(layout).Finalize()

    known_face_name, known_face_encoding = getface(data_path)

    info = ''
    cap = cv2.VideoCapture(0)
    while True:
        event,value = window.read(timeout=20, timeout_key='timeout')
        if event == sg.WIN_CLOSED:
            break
        elif event == '1. Register':
            register = True
            count = 0
        elif event == '2. Update':
            known_face_name, known_face_encoding = getface(data_path)
        elif event == '3. Recognize':
            recogniseFrame = True
        
        name = value['name']
        id = value['id']
        course = value['course']

        ret,frame = cap.read()
        
        if register:
            createimages(frame,count)
            info = 'Saving' + str(count)
            count +=1
            if count >= sampleNum:
                row = [id,name,course]
                info = writedata(datafile,row)
                register = False
        
        if recogniseFrame:
            frame,info = imagetrack(frame,known_face_name,known_face_encoding)
        
        imgbytes = cv2.imencode('.png',frame)[1].tobytes()
        window['image'].update(data = imgbytes)
        window['title'].update(info)
    
    cap.release()
    cv2.destroyAllWindows
    window.close()

main()

