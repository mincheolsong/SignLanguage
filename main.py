import math

import cv2
import mediapipe as mp
import keyboard
import time
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image, ImageTk
from tkinter import Tk, Label, Button


class App:
    def __init__(self, windowtk):
        self.width, self.height = 1280,720
        self.delay = 33

        self.gui = windowtk
        self.gui.geometry("1280x900")
        self.imageLabel = Label(self.gui, text="GUI")
        self.imageLabel.pack()
        
        self.textLabel = []
        self.text = []
        for i in range(0, 3) :  
            self.text.append("")
            self.textLabel.append(Label())
            self.textLabel[i].pack(side="top", anchor="s")
            self.textLabel[i].configure(text="", font=('Arial', 25))
        print(self.text)

    def draw(self, image):
        imgtk = ImageTk.PhotoImage(image = Image.fromarray(image)) # ImageTk 객체로 변환
        # OpenCV 동영상
        self.imageLabel.imgtk = imgtk
        self.imageLabel.configure(image=imgtk)
 
    def clear(self):
        self.text[0] = ""
        self.text[1] = ""
        self.text[2] = ""
        self.textLabel[0].configure(text=self.text[0])
        self.textLabel[1].configure(text=self.text[0])
        self.textLabel[2].configure(text=self.text[0])
    
    def drawText(self, index ,text):
        self.text[index] = self.text[index] + str(text)
        self.textLabel[index].configure(text=self.text[index], font=('Arial', 25))

    def ClaerAndDrawText(self, index ,text):
        self.text[index] = str(text)
        self.textLabel[index].configure(text=str(text), font=('Arial', 25))


##########################################################################################################

def update(gui):

    global sentence, selected_words, complete, font, fontpath, b,g,r,a, recognizeDelay, sentence, prev_index, startTime, cap, knn, label, angle,labelFile, angleFile, dic_file, file, f, hands, mp_hands, mp_drawing, gesture, next_cnt, previous_cnt,stop_cnt, model, last_action, i, word
    global action_seq, seq, actions, seq_length, isStop, startActionTime

    ret, image = cap.read() # 캠 연결

    if not ret: #연결확인
        return

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) # OpenCV : BGR 사용, MediaPipe : RGB 사용
    result = hands.process(image) # 전처리 및 모델 추론을 함께 실행한다.
    
    img_pil = Image.fromarray(image) # 한글을 출력하기 위해 image를 변환(PIL라이브러리 사용)
    draw = ImageDraw.Draw(img_pil) 

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks: # 여러개의 손을 인식 할 수 있으니까, for문 반복
            joint = np.zeros((21,4)) # 손 관절 (joint) 넘파이 배열로 생성
            for j, lm in enumerate(hand_landmarks.landmark): # media pipe의 landmark를 반복하며 joint에 대입
                joint[j] = [lm.x,lm.y,lm.z,lm.visibility]

            v1 = joint[[0,1,2,3,0,5,6,7,0, 9,10,11, 0,13,14,15, 0,17,18,19],:3] # 벡터를 구하기 위해 생성하는 v1,v2 (v2에서 v2을 빼면 v백터가 된다)
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:3]

            v = v2-v1
            v = v / np.linalg.norm(v,axis=1)[:,np.newaxis] # 정규화

            compareV1 = v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:] #각 벡터의 각도를 비교하기 위해 생성하는 compare벡터
            compareV2 = v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]

            angle = np.arccos(np.einsum('nt,nt->n',compareV1,compareV2)) # compare벡터를 사용하여 각도를 구함
            angle = np.degrees(angle)

            #############움직이는 동작('next', 'prev')을 위한 코드(RNN)###################
            d = np.concatenate([joint.flatten(),angle]) # ★
            seq.append(d) # ★
            #############################################################################

            if keyboard.is_pressed('a'): # gesture를 학습하기 위한 조건문(a키를 누르고 있으면 각도와 라벨이 test.txt에 write됨)
                for num in angle:
                    num = round(num,6)
                    f.write(str(num))
                    f.write(',')

                f.write("13.000000") # 학습하고자 하는 손 동작의 인덱스를 입력

                f.write('\n')
                print('next')

            data = np.array([angle],dtype=np.float32)

            ret, results, neighbours, dist = knn.findNearest(data,3) #knn알고리즘 적용

            index = int(results[0][0])
            
            if index in gesture.keys():
                if index != prev_index:
                    startTime = time.time()
                    prev_index = index

                elif time.time() - startTime > recognizeDelay:
                    if index == 26:
                        sentence += ' '
                    elif index == 27:
                        sentence = ''
                    elif index == 25: # done동작(25) 하면 위에서 읽은 dic_file에서 sentence를 검색하고, 그 위치의 단어를 출력
                        for i in range(0, dic_file.shape[0]):
                            if (sentence == dic_file['초성'][i]):
                                selected_words.append(dic_file['단어'][i])
                                
                        complete=1 #complete 를 표시하기 위해 1로 변경한다
                        i=0
                        word=''
                    if complete==0 and index!=27 and index!=26:
                        sentence += gesture[index]
                    startTime = time.time()

                if complete==0:
                    word = gesture[index]
                draw.text((int(hand_landmarks.landmark[0].x*image.shape[1]),int(hand_landmarks.landmark[0].y*image.shape[0])), word, font=font, fill=(b, g, r, a))
                
            image = np.array(img_pil)
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    gui.ClaerAndDrawText(0, sentence)
    #draw.text((20,400),sentence,font=font,fill=(b, g, r, a))
    if complete==1:
        print(sentence)
        print(selected_words)
        complete = 2

    if complete==2: # ★
        if len(seq) < seq_length:
            gui.imageLabel.after(100, update, gui)
            return

        input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
        y_pred = model.predict(input_data).squeeze()
        i_pred = int(np.argmax(y_pred))
        conf = y_pred[i_pred]

        if conf < 0.9:
            gui.imageLabel.after(1, update, gui)
            return

        action = actions[i_pred]
        action_seq.append(action)

        if len(action_seq) < 3:
            gui.imageLabel.after(100, update, gui)
            return

        
        this_action = '?'
        
        # if(hand_landmarks.landmark[0].x < 0.2 or hand_landmarks.landmark[0].x > 0.8):
        #     continue
        if action_seq[-1] != action_seq[-2]:
            startActionTime = time.time()

        if action_seq[-1] == action_seq[-2] and action_seq[-2] == action_seq[-3] : # 같은 동작을 세번 연속하면 this_action으로 인식
            this_action = action
        if this_action == 'next':
            next_cnt += 1 
            if next_cnt > 5 :
                print("next")
                next_cnt = 0
                i+=1
                if(i==len(selected_words)):
                    i=0
                         
        elif this_action == 'prev':
            previous_cnt += 1
            if previous_cnt > 5 :
                print("previous")
                previous_cnt = 0
                i-=1
                if(i==-1):
                    i=len(selected_words)-1
                
        elif this_action == 'stop':
            if time.time() - startActionTime > 2:
                print(time.time())
                print(startActionTime)
                isStop = True
                print("stop")

        if result.multi_hand_landmarks:
            draw.text((int(result.multi_hand_landmarks[0].landmark[0].x * image.shape[1]),
                    int(result.multi_hand_landmarks[0].landmark[0].y * image.shape[0] + 20)), this_action , font=font, fill=(255,255,255))
        if not isStop:
            printWords = list(selected_words)
            if len(printWords) >= 9:
                while printWords[4] != selected_words[i]:
                    tmp = printWords[0]
                    printWords.append(tmp)
                    del printWords[0]
                del printWords[9:]
            else:
                while len(selected_words) and selected_words[i] != printWords[int(len(printWords)/2)]:
                    tmp = printWords[0]
                    printWords.append(tmp)
                    del printWords[0]
            #print(printWords)
            gui.ClaerAndDrawText(1,printWords)
            gui.ClaerAndDrawText(2, str(selected_words[i]))
        if isStop:
            gui.ClaerAndDrawText(2, str(selected_words[i])+ " FIX")

    image = np.array(img_pil)

    cv2.waitKey(1)
    if keyboard.is_pressed('b'):
        return
    gui.draw(image)
    gui.imageLabel.after(100, update, gui)



################################################################################################################
#main

# ★ 표시는 움직이는 손 동작('next', 'prev')을 학습시키기 위한 코드

gesture = {
    0:'ㄱ',1:'ㄴ',2:'ㄷ',3:'ㄹ',4:'ㅁ',5:'ㅂ',6:'ㅅ',7:'ㅇ',
    8:'ㅈ',9:'ㅊ',10:'ㅋ',11:'ㅌ',12:'ㅍ',13:'ㅎ', 25:'done',26:'spacing',27:'clear'
}
actions = ['prev', 'next', 'stop'] # ★
seq_length = 30 # ★
next_cnt = 0 # ★
previous_cnt = 0 # ★
stop_cnt = 0 # ★
model = load_model('models/model.h5') # ★ 학습한 모델 load
        
seq = [] # ★
action_seq = [] # ★
last_action = None # ★

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

f = open('test.txt','w')
file = np.genfromtxt('dataSet.txt',delimiter=',')
dic_file = pd.read_csv('dictionary.csv') #사전 파일


angleFile = file[:,:-1]
labelFile = file[:,-1]
angle = angleFile.astype(np.float32)
label = labelFile.astype(np.float32)

knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)

startTime = time.time()

prev_index = 0
sentence = ''
recognizeDelay = 2

b,g,r,a = 0,0,0,0
fontpath = "C:\Windows\Fonts\gulim.ttc"
font = ImageFont.truetype(fontpath, 50)

startActionTime = 0;
complete = 0 # 글씨 입력을 완료 했는지 확인하는 변수(done 동작을 하면 1로 바꿈)
selected_words = []
word=''
isStop = False
i = 0

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
gui = App(Tk())
update(gui)
gui.gui.mainloop()


f.close()

cap.release()