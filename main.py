import argparse
import io
import time
import numpy as np
from camera_pi import Camera
from tflite_runtime.interpreter import Interpreter
import cv2
from lite_lib import load_labels, set_input_tensor, classify_image
import RPi.GPIO as GPIO

paper = 8 #設定燈泡腳位，其中紅色是布，綠色是石頭，黃色是剪刀
rock = 5
sciss0rs = 3
GPIO.setmode(GPIO.BOARD)

GPIO.setup(paper, GPIO.OUT)
GPIO.setup(rock, GPIO.OUT)
GPIO.setup(sciss0rs, GPIO.OUT)

try:
    GPIO.output(paper, GPIO.LOW)
    GPIO.output(rock, GPIO.LOW)
    GPIO.output(sciss0rs, GPIO.LOW)
    while True:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument(
            '--model', help='File path of .tflite file.', required=True)
        parser.add_argument(
            '--labels', help='File path of labels file.', required=True)
        args = parser.parse_args()

        labels = load_labels(args.labels)
        camera = Camera()
        interpreter = Interpreter(args.model)
        interpreter.allocate_tensors()
        #得到model所需要的input大小，將輸入image resize成該大小
        _, height, width, _ = interpreter.get_input_details()[0]['shape']
        while(True):
            #get image from camera
            img = camera.get_frame()
            image = cv2.resize(img,(width, height))

            #inference image
            results = classify_image(interpreter, image)
            label_id, prob = results[0]
            

            #put inference result on image
            label_text = labels[label_id].split(" ")[1]
            prob_text = 'Prob:' + str(int(prob*100)) + '%'
            cv2.putText(image, label_text, (90, 40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, prob_text, (80, 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            if label_id == 0:
                GPIO.output(paper, GPIO.HIGH)
                GPIO.output(rock, GPIO.LOW)
                GPIO.output(sciss0rs, GPIO.LOW)
            elif label_id == 1:
                GPIO.output(paper, GPIO.LOW)
                GPIO.output(rock, GPIO.HIGH)
                GPIO.output(sciss0rs, GPIO.LOW)
            elif label_id == 2:
                GPIO.output(paper, GPIO.LOW)
                GPIO.output(rock, GPIO.LOW)
                GPIO.output(sciss0rs, GPIO.HIGH)

            #show image
            cv2.imshow('img', image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

except KeyboardInterrupt:
    print("kb")
finally:
    GPIO.cleanup() #程式結束，記得釋放腳位
