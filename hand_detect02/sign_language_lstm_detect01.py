from tensorflow import keras
import numpy as np
import cv2
import mediapipe as mp
import numpy.linalg as LA  #선형 방정식 계산

# 손동작 5번 바다 LSTM으로 예측할 것임
seq_length = 5

# 손동작을 저장할 리스트
seq = []

# 예측값과 문자열
gesture = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I',
    9:'J', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q',
    17:'R', 18:'U', 19:'S', 20:'T', 21:'U', 22:'V', 23:'W', 24:'Y', 25:'Z'
}

# keras.models.load_model('C://ai_project01/hand_lstm_train_result') :
# C://ai_project01/hand_lstm_train_result 디렉토리에 저장한 LSTM 모델 읽어서 리턴
model = keras.models.load_model('C://ai_project01/hand_lstm_train_result')

# 읽어온 모델의 내용 출력
model.summary()

# 창의 크기를 수정 가능하도록 설정 cv2.WINDOW_NORMAL
# cv2.namedWindow(윈도우 창 타이틀, flags=cv2.WINDOW_NORMAL)
cv2.namedWindow('webcam_window01', flags=cv2.WINDOW_NORMAL)

# 창의 크기를 수정
# cv2.namedWindow(윈도우 창 타이틀, width=윈도우 가로 크기, height=윈도우 세로 크기)
cv2.resizeWindow('webcam_window01', width=1024, height=800)

# mp.solutions.hands : 화면에서 손과 손가락 관절 위치 정보를 탐지 객체 리턴
mp_hands = mp.solutions.hands

# 인식한 손의 keypoint를 그릴 객체
mp_drawing = mp.solutions.drawing_utils

# cv2.VideoCapture(0) : 웹캠의 화면을 가져올 객체를 생성해서 리턴
cap = cv2.VideoCapture(0)

# with mp_hands.Hands : 여기부터 화면에서 손과 손가락 위치 탐지
with mp_hands.Hands() as hands:

    # cap.isOpened() : 웹캠이 정상적으로 동작하면 True, 못하면 False 리턴
    while cap.isOpened()==True: # 웹캠이 정상적으로 동작하는 동안 반복
        # cap.read() : 웹캠의 화면을 가져옴

        # 웹캠의 화면을 정상적으로 가져오면 success에 True에 저장
        # 실패하면 success에 False 저장

        # 웹캠이 가져온 화면은 image에 저장
        success, image = cap.read()

        # cv2.flip(image, 1) : 웹캠의 이미지를 좌우반전
        # 1은 좌우반전 0은 상하반전
        image = cv2.flip(image, 1)

        if success == False : # 웹캠의 화면을 가져오는데 실패하면
             continue

        # cv2.cvtColor(image, cv2.COLOR_BGR2RGB) : 웹캠 이미지 image를 BGR에서 RGB로 변환
        # hands.process() : 웹캠에서 손과 손가락 관절 위치 탐지해서 리턴
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # results.multi_hand_landmarks : 웹캠 화면에서 손이 없으면 None 리턴
        # results.multi_hand_landmarks != None : 화면에 손이 존재하면
        if results.multi_hand_landmarks != None:

            # results.multi_hand_landmarks : 탐지한 손의 keypoint들이 저장
            # for hand_handmarks in results.multi_hand_landmarks : 탐지한 손의 keypoint 순서대로 1개씩
            #                                                      hand_landmarks에 저장하고 반복분 실행
            for hand_landmarks in results.multi_hand_landmarks:
                # 탐지한 손의 keypoint를 화면에 그림
                mp_drawing.draw_landmarks(
                    image, # keypoint를 그릴 이미지(웹캠화면)
                    hand_landmarks, # keypoint 좌표
                    mp_hands.HAND_CONNECTIONS, # 탐지한 손의 keypoint를 선으로 연결함
                    # keypoint를 표시할 원의 모양
                    # color=(0, 255, 0) : 녹색 thickness=2 : 선두께 circle_radius=4 : 원반지름
                    # color=(255, 0, 0) : 파란색 thickness=2 : 선두께 circle_radius=2 : 원반지름
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(244, 0, 0), thickness=2, circle_radius=2)
                )

                # np.zeros((21, 3)) : 21줄 3칸의 0으로 초기화된 배열 생성
                joint = np.zeros((21, 3))

                # hand_landmarks.landmark : 손의 keypoint 좌표 리턴
                # j : keypoint의 index
                # lm : keypoint 좌표
                for j, lm in enumerate(hand_landmarks.landmark):
                    # j : keypoint index
                    print("j=", j)
                    # lm : keypoint 좌표
                    print("lm=", lm)
                    # lm.x : keypoint의 x좌표
                    print("lm.x=", lm.x)
                    # lm.y : keypoint의 y좌표
                    print("lm.y=", lm.y)
                    # lm.z : keypoint의 z좌표
                    print("lm.z=", lm.z)

                    # 각도를 구하기 위해 keypoint의 x, y, z좌표를
                    # joint 배열 j번째 행에 대입
                    joint[j] = [lm.x, lm.y, lm.z]
                    print("="*100)

                print("joint=", joint)

                # v1에서 v2를 빼서 차를 계산
                # 배열의 행 인덱스                                                       배열의 열 인덱스
                # x, y, z
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]

                # v2에서 v1 빼기
                v = v2 - v1
                print("="*100)
                print("v=", v)
                print("="*100)

                # v를 정규화 시킴
                # 정규화 결과는 1차원 배열
                v_normal = LA.norm(v, axis=1)
                print("="*100)
                print("v_normal=", v_normal)
                print("="*100)

                # v와 연산하기 위해서 v_normal을 2차원 배열로 변환
                # [:, np.newaxis]
                #                : => 모든 행
                #                np.newaxis => 차원 추가
                # 모든 행의 차원을 추가해서 1차원 배열 v_normal을 2차원 배열 v_normal2로 변환
                v_normal2 = v_normal[:, np.newaxis]
                print("v_normal2=", v_normal2)

                # v를 v_normal2로 나눠서 거리를 정규화 시킴
                v2 = v / v_normal2
                print("="*100)
                print("v2=", v2)
                print("="*100)

                # a, b 배열의 곱을 계산
                a = v2[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :]
                b = v2[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
                ein = np.einsum('ij,ij->i', a, b)
                print("="*100)
                # 행렬의 곱 조회
                print("ein=", ein)
                print("="*100)

                # 코사인값 계산 (라디안)
                # np.arccos(0.5) 1.04771975511965979 -> pi/3
                radian = np.arccos(ein)
                print("radian=", radian)

                # 코사인값을 각도로 변환
                # np.degrees(1.0471975511965979) : 60도
                angle = np.degrees(radian)
                print("angle=", angle)

                # joint.flatten() : 관절 좌표를 1차원 배열로 변환
                # np.concatenate([joing.flatten(), angle]) : 관절 좌표와 각도를 data에 저장
                data = np.concatenate([joint.flatten(), angle])
                print("="*100)
                print("data=", data)
                print("="*100)

                # seq에 손동작 관절의 좌표와 각도가 저장된 data 추가
                seq.append(data)

                # seq의 한행은 손동작 한개의 좌표와 개수
                # len(seq) : seq에 저장된 행의 개수가 5미만
                if len(seq) < 5 :
                    # 다음 화면으로 넘어감
                    continue

                # 마지막 손동작 5개 행을 last_seq에 대입
                last_seq = seq[-5: ]
                # last_seq를 배열로 변환해서 input_arr에 대입
                input_arr = np.array(last_seq,dtype=np.float32)
                # input_arr (2차원 배열)을 3차원 배열로 변환
                # input_arr.reshape(2차원 배열의 개수 => 1, 행의 개수 => 5, 열의 개수 => 78)
                input_lstm_arr = input_arr.reshape(1, 5, 78)
                print("="*100)
                print("input_lstm_arr=", input_lstm_arr)
                print("=" * 100)
                print("=" * 100)
                print("input_lstm_arr.shape=", input_lstm_arr.shape)
                print("=" * 100)

                # input_lstm_arr을 이용해서 수어를 예측하고 확률을 y_pred 대입
                y_pred = model.predict(input_lstm_arr)
                print("=" * 100)
                print("y_pred=", y_pred)
                print("=" * 100)

                # np.argmax(y_pred) :  y_pred에서 확률이 가장 높은 인덱스를 idx에 대입
                idx = int(np.argmax(y_pred))
                print("=" * 100)
                print("idx=", idx)
                print("=" * 100)

                # idx번째 알파벳을 gesture에서 리턴
                letter = gesture[idx]
                print("=" * 100)
                print("letter=", letter)
                print("=" * 100)

                # 확률이 가장 높은 idx번째 데이터의 확률을 conf에 대입
                conf = y_pred[0, idx]
                print("=" * 100)
                print("conf=", conf)
                print("=" * 100)

                # 예측값 출력
                cv2.putText(image,
                            # round(conf*100, 2) : 확률 conf*100을 소숫점 2자리 반올림
                            text=f"{letter}{round(conf*100, 2)} percent!!", # 예측값
                            org=(
                                int(hand_landmarks.landmark[0].x * image.shape[1]), # 손위치 x좌표
                                int(hand_landmarks.landmark[0].y * image.shape[0]) # 손위치 y좌표
                            ),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, # 글자체
                            fontScale=1, # 글자크기
                            color=(0, 0, 255), # 글자색
                            thickness=2 # 폰트 두께

                )

        # cv2.imshow(윈도우창 타이틀, 윈도우창에 출력할 이미지) : 웹캠 화면을 화면에 출력
        cv2.imshow('webcam_window01', image)

        # cv2.waitKey(1) : 사용자가 키보드 입력하도록 1초 기다림
        #                  기다리는 시간 동안 사용자가 키보드 입력을 하면 입력한 키보드값을 리턴
        #                  기다리는 동안 사용자가 입력한 키보드가 없으면 None 리턴
        if cv2.waitKey(1) == ord('q'): # 키보드 입력이 q이면
            break # 반복문 종료
# 웹캠의 화면 그만 가져오도록 설정
cap.release()