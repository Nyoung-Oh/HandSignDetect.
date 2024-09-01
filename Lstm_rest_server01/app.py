from flask import Flask
from flask import request
import base64
import json
from tensorflow import keras
import mediapipe as mp
import numpy.linalg as LA
import numpy as np
import cv2
import matplotlib.pyplot as plt

# keras.models.load_model('C://ai_project01/hand_lstm_train_result') :
# C://ai_project01/hand_lstm_train_result 디렉토리에 저장한 LSTM 모델 읽어서 리턴
model = keras.models.load_model('C://ai_project01/hand_lstm_train_result')

# 읽어온 모델의 내용 출력
model.summary()

# 손동작 5번 마다 LSTM으로 예측할 것임
seq_length = 5

# 예측값과 문자열
gesture = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I',
    9:'J', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q',
    17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 25:'Y', 26:'Z'
}

# mp.solution.hands : 화면에서 손과 손가락 관절 위치 정보를 탐지 객체 리턴
mp_hands = mp.solutions.hands

app = Flask(__name__)

# post 방식의 lstm_detect url을 입력하면 함수가 호출되는 rest server 메소드
@app.route("/lstm_detect", methods=['POST'])
def lste_detect01():
    # LSTM 탐지 결과자 저장될 변수
    lstm_result = []
    # 손동작을 저장할 리스트
    seq = []

    # with mp_hands.Hands : 여기부터 화면에서 손과 손가락 위치 탐지
    with mp_hands.Hands() as hands:
        # request.get_json() 스프링 컨트롤러에서 보낼 웹캠 이미지를 리턴
        # 5개의 B 이미지 저장(JSON 배열)
        json_image = request.get_json()
        print("="*100)
        print("json_image=", json_image)
        print("=" * 100)

        # image에서 key data의 value 리턴(5개의 B 이미지 저장된 배열)
        encoded_data_arr = json_image.get("data")
        print("=" * 100)
        print("encoded_data_arr=", encoded_data_arr)
        print("=" * 100)

        # encoded_data_arr에서 이미지 1개를 순서대로 encoded_data에 저장
        # 몇번째 이미지인지 index에 저장
        for index, encoded_data in enumerate(encoded_data_arr):
            print("=" * 100)
            print("index=", index)
            print("=" * 100)
            print("encoded_data=", encoded_data)
            print("=" * 100)

            # encoded_data.replace("image/jpeg;base64,", "") : encoded_data에서 image/jpeg;base64, 삭제
            encoded_data = encoded_data.replace("image/jpeg;base64,", "")

            # base64.b64decode(encoded_data) : encoded_data를 원래 이미지로 복원
            decoded_data = base64.b64decode(encoded_data)

            # np.fromstring(decoded_data, np.uint8) : decoded_data를
            #         정수 np.uint8가 저장된 1차원 배열로 변환해서 nparr에 저장
            nparr = np.fromstring(decoded_data, np.uint8)
            print("=" * 100)
            print("nparr=", nparr)
            print("=" * 100)

            # cv2.imdecode(nparr, cv2.IMREAD_COLOR) : 1차월 배열 nparr을 BGR 형태의 3차원 배열로 변환
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # cv2.flip(image, 1) : 이미지 좌우 반전
            # 1은 좌우 반전, 0은 상하 반전
            image = cv2.flip(image, 1)
            

            print("=" * 100)
            print("image=", image)
            print("=" * 100)

            # cv2.cvtColor(image, cv2.COLOR_BGR2RGB) : 웹캠 이미지 image를 BGR에서 RGB로 변환
            # hands.process() : 웹캠에서 손과 손가락 관절 위치 탐지해서 리턴
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # results.multi_hand_landmarks : 이미지에서 손이 없으면 None 리턴
            # 존재하면
            if results.multi_hand_landmarks != None:

                # results.multi_hand_landmarks : 탐지한 손의 keypoint들이 저장
                # for hand_landmarks in results.multi_hand_landmarks : 탐지한 손의 keypoint 순서대로 1개씩
                #                                                      hand_lanmarks에 저장하고 반복문 실행
                for hand_landmarks in results.multi_hand_landmarks:
                    # np.zeros((21, 3)) : 21줄 3칸의 0으로 초기화면 배열 생성
                    joint = np.zeros((21, 3))

                    # hand_landmarks.landmark : 손의 keypoint 좌표 리턴
                    # j : keypoint의 index
                    # lm : keypoint 좌표

                    for j, lm in enumerate(hand_landmarks.landmark):

                        # j : keypoint index
                        print("j=", j)
                        # lm : keypoint 좌표
                        print("lm=", lm)
                        # lm.x : keypoint x좌표
                        print("lm.x=", lm.x)
                        # lm.y : keypoint y좌표
                        print("lm.y=", lm.y)
                        # lm.z : keypoint z좌표
                        print("lm.z=", lm.z)

                        # 각도를 구하기 위해 keypoint의 x, y, z좌표를 joint 배열 j번째 행에 대입
                        joint[j] = [lm.x, lm.y, lm.z]
                        print("=" * 100)
                    print("=" * 100)
                    print("joint=", joint)
                    print("=" * 100)

                    # v1에서 v2를 빼서 차를 계산
                    # 배열의 행 인덱스
                    # x, y, z
                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
                    # v2에서 v1 빼기
                    v = v2 - v1
                    print("=" * 100)
                    print("v=", v)
                    print("=" * 100)

                    # v를 정규화 시킴
                    # 정규화 결과는 1차원 배열
                    v_normal = LA.norm(v, axis=1)
                    print("=" * 100)
                    print("v_normal=", v_normal)
                    print("=" * 100)

                    # v와 연산하기 위해서 v_normal을 2차원 배열로 변환
                    # [:, np.newaxis]
                    #                : => 모든 행
                    #                np.newaxis => 차원 추가
                    # 모든 행의 차원을 추가해서 1차원 배열 v_normal을 2차원 배열 v_normal2로 변환
                    v_normal2 = v_normal[:, np.newaxis]
                    print("=" * 100)
                    print("v_normal2=", v_normal2)
                    print("=" * 100)

                    # v를 v_normal2로 나눠서 거리를 정규화 시킴
                    v2 = v / v_normal2
                    print("=" * 100)
                    print("v2=", v2)
                    print("=" * 100)

                    # a, b 배열의 곱을 계산
                    a = v2[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :]
                    b = v2[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
                    ein = np.einsum('ij,ij->i', a, b)
                    print("=" * 100)
                    # 행렬의 곱 조회
                    print("ein=", ein)
                    print("=" * 100)

                    # 코사인값 계산(라디안)
                    # np.arccos(0.5) 1.0471975511965979 -> pi/3
                    radian = np.arccos(ein)
                    print("=" * 100)
                    print("radian=", radian)
                    print("=" * 100)

                    # 코사인값을 각도롣 변환
                    # np.degrees(1.0471975511965979) : 60도
                    angle = np.degrees(radian)
                    print("=" * 100)
                    print("angle=", angle)
                    print("=" * 100)

                    # joint.flatten() : 관절 좌표를 1차원 배열로 변환
                    # np. concatenate([joint.flatten(), angle]) : 관절 좌표와 각도를 data에 저장
                    data = np.concatenate([joint.flatten(), angle])
                    print("=" * 100)
                    print("data=", data)
                    print("=" * 100)

                    # seq에 손동작 관절의 좌표와 각도가 저장된 data 추가
                    seq.append(data)

                    # seq의 한행은 손동작 한개의 좌표와 개수
                    # len(seq) : seq에 저장된 행의 개수가 5미만
                    if len(seq) < 5 :
                        # 반복문 다음으로 넘어감
                        continue

                    # 마지막 손동작 5개 행을 last_seq에 대입
                    last_seq = seq[-5:]

                    # last_seq를 배열로 변환해서 input_arr에 대입
                    input_arr = np.array(last_seq, dtype=np.float32)
                    print("input_arr=", input_arr.shape)
                    # input_arr (2차원 배열)을 3채원 배열로 변환
                    # input_arr.reshape(2차원 배열의 개수 => 1, 행의 개수 => 5, 열의 개수 => 78)
                    input_lstm_arr = input_arr.reshape(1, 5, 78)

                    print("=" * 100)
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
                    
                    # np.argmax(y_pred) : y_pred에서 확률이 가장 높은 인덱스를 idx에 대입
                    idx = np.argmax(y_pred)
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
                    
                    # 탐지 결과를 lstm_result에 추가
                    lstm_result.append({
                        # round(conf*100, 2) : 확률 conf*100을 소숫점 2자리 반올림
                        "text":f"{letter} {round(conf*100, 2)} percent!!", # 예측값
                        "x": int(hand_landmarks.landmark[0].x * image.shape[1]), # 손위치 x좌표
                        "y": int(hand_landmarks.landmark[0].y * image.shape[0])  # 손위치 y좌표
                    })

                    print("=" * 100)
                    print("lstm_result=", lstm_result)
                    print("=" * 100)
                    
    # 결과를 JSON으로 변환해서 리턴                
    return json.dumps(lstm_result)


# post 방식의 hello_rest_server_url을 입력하면 호출되는 rest server의 메소드
@app.route('/hello_rest_server', methods=['POST'])
def hello_rest1():
    return '안녕 난 rest server야'

@app.route("/image_test01", methods=["POST"])
def image_send_test01():
    json_image = request.get_json()
    print("json_image=", json_image)

    return "스프링 컨트롤러가 보낸 이미지 잘 받았습니다."

# post 방식의 image_test01 url을 입력하면 함수가 호출되는 rest server 메소드
@app.route("/image_test02", methods=["POST"])
def image_send_test02():
    # request.get_json() 스프링 컨트롤러에서 보낼 웹캠 이미지를 리턴
    # 5개의 B 이미지 저장(JSON 배열)
    json_image = request.get_json()
    print("="*100)
    print("json_image=", json_image)
    print("="*100)

    # image에서 key data의 value 리턴 (5개의 B 이미지 저장된 배열)
    encoded_data_arr = json_image.get("data")
    print("="*100)
    print("encoded_data_arr=", encoded_data_arr)
    print("="*100)

    # encoded_data_arr에서 이미지 1개를 순서대로 encoded_data에 저장
    # 몇번째 이미지인지 indes에 저장
    for index, encoded_data in enumerate(encoded_data_arr):
        print("=" * 100)
        print("index=", index)
        print("=" * 100)
        print("encoded_data=", encoded_data)
        print("=" * 100)

        # encoded_data.replace("image/jpeg;base64,","") : encoded_data에서 image/jpeg;base64 삭제
        encoded_data = encoded_data.replace("image/jpeg;base64", "")
        # base64.b64decode(encoded_data) : encoded_data를 원래 이미지로 복원
        decoded_data = base64.b64decode(encoded_data)

        # image index 번호.jpg로  파일을 저장할 객체 f 생성
        with open(f"image{index}.jpg", "wb") as f:
            # decoded_data : 스프링 컨트롤러부터 전송받은 문자열을 이미지로 변환
            # f.write(decoded_data) : decoded_data를 파일로 저장
            f.write(decoded_data)

    return "스프링 컨트롤러가 보낸 이미지 잘 저장했습니다."
if __name__ == '__main__':
    app.run()