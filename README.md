```c
import cv2
import mediapipe as mp
import numpy as np
import serial

ser = serial.Serial('COM8', 9600)
max_num_hands = 2
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}
rps_gesture = {0:'rock', 5:'paper', 9:'scissors'}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition modelq
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)
ser.write(b' ')
while cap.isOpened():
    
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        rps_result = []

        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            # Draw gesture result
            if idx in rps_gesture.keys():
                org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
                cv2.putText(img, text=rps_gesture[idx].upper(), org=(org[0], org[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                rps_result.append({
                    'rps': rps_gesture[idx],
                    'org': org
                })

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            # Who wins?
            if len(rps_result) >= 2:
                winner = None
                text = ''
               
                

                if rps_result[0]['rps']=='rock':
                    if rps_result[1]['rps']=='rock'     : text = 'Tie'; ser.write(b' ');
                    elif rps_result[1]['rps']=='paper'  : text = 'Paper wins' ; ser.write(b'M');ser.write(b'L'); winner = 1
                    elif rps_result[1]['rps']=='scissors': text = 'Rock wins' ; ser.write(b'H');ser.write(b'N'); winner = 0
                elif rps_result[0]['rps']=='paper':
                    if rps_result[1]['rps']=='rock'     : text = 'Paper wins'  ; ser.write(b'H');ser.write(b'N');winner = 0
                    elif rps_result[1]['rps']=='paper'  : text = 'Tie'; ser.write(b' ');
                    elif rps_result[1]['rps']=='scissors': text = 'Scissors wins';ser.write(b'M');ser.write(b'L'); winner = 1
                elif rps_result[0]['rps']=='scissors':
                    if rps_result[1]['rps']=='rock'     : text = 'Rock wins'   ;ser.write(b'M');ser.write(b'L'); winner = 1
                    elif rps_result[1]['rps']=='paper'  : text = 'Scissors wins';ser.write(b'H');ser.write(b'N'); winner = 0
                    elif rps_result[1]['rps']=='scissors': text = 'Tie'; ser.write(b' ');

                if winner is not None:
                    cv2.putText(img, text='Winner', org=(rps_result[winner]['org'][0], rps_result[winner]['org'][1] + 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3)
                cv2.putText(img, text=text, org=(int(img.shape[1] / 2), 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=3)
    
    cv2.imshow('Game', img)
    if cv2.waitKey(1) == ord('q'):
        break
```

# 마크다운

> * 이름 : 김영빈 
> * 학번 : 2217110188 
>  
>   > ### 학력  
>   > + *안골포초등학교*  
>   > + *용원중학교*  
>   > + *창원기계공고*
>   >   * 스마트전기과  
>   > + **창원폴리텍**  
>   >   * *스마트팩토리과*

***

자격증 | 이름
------ | -----
 1 | 전기기능사
 2 |   
 3 | 

***





# AI CONTROL

## 내가 만든 첫 마크다운입니다.
### 크기
#### 크기
##### 크기
###### 크기

>  kyb
>>  kyb
>>>  kyb
>>>> kyb

1. 첫 번째
2. 두 번째
3. 세 번째
1. k
3. y 
2. b

* 빨
    + 파
        - 초
* 남        

hellow
world

hellow

world

---
***
___

+ 참조링크

Link: [Google][googlelink]

[googlelink]: https://google.com "Go google"


* 참조링크

[naver][naverlink]

[naverlink]: https://naver.com "Go naver"

***

- 외부링크

[Google](https://google.com)

***

* 자동연결

 외부링크: <http://example.com/>

 이메일링크: <address@example.com>

 ***

 *1개*

 _1개_

 **2개**

 __2개__

 ~~취소선~~

***

smartfactory
kimyoungbin

smartfactory   
kimyoungbin

***

순서없는 분류
* 중1
* 중2
    * 소1
    * 소2

순서있는 분류
1. 중1
1. 중2
    1. 소1
    1. 소2

***

![dog](/dog.jfif)

***

smart | factory
----- | -------
kim | youngbin

***

- [x] 체크박스  
- [ ] 췌크박스

***

```c
void main(){
    printf("hello");
}
```

***

dsdsdsddsdsd `printf("hellow");` dsdsddsd

---
\*text\*  
\_text\*  
\*\*text\*\*  
\_\_text\_\_  
