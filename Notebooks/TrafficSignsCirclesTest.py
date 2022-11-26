import cv2 as cv
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import load_model

def empty(x):
	pass
 
capture = cv.VideoCapture(0, cv.CAP_DSHOW)
width = 640
height = 480
cyv = 0 

model = load_model('transfer_learning_modelo_optimizado.h5')

cv.namedWindow("Parameters")
cv.resizeWindow("Parameters",480,240)
cv.createTrackbar("MinRadius","Parameters",45,255,empty)
cv.createTrackbar("MaxRadius","Parameters",100,255,empty)
cv.createTrackbar("Param1","Parameters",60,200,empty)
cv.createTrackbar("Param2","Parameters",40,100,empty)
 
while (True):
 
    (ret, framet) = capture.read()
    framet = cv.resize(framet,(width,height))
    

    #Guarda los valores de las barras de parametros del circulo
    MinRadius = cv.getTrackbarPos("MinRadius","Parameters")
    MaxRadius = cv.getTrackbarPos("MaxRadius","Parameters")
    Param1 = cv.getTrackbarPos("Param1","Parameters")
    Param2 = cv.getTrackbarPos("Param2","Parameters")

	#Procesamiento de image
    imgGray = cv.cvtColor(framet,cv.COLOR_BGR2GRAY)
    imgGray = cv.medianBlur(imgGray, 5)
    rows = imgGray.shape[0]

	#Deteccoion de circulo en la imagen
    
    circles = cv.HoughCircles(imgGray,cv.HOUGH_GRADIENT,1,rows/8, 
                         		param1 = Param1, param2 = Param2, 
                         		minRadius = MinRadius, maxRadius = MaxRadius)

    if circles is not None:		#Condicional - si detecta circulos
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cy = i[0]
            cyv = cy
            for j in circles[0, :]:
                cx = j[1]
                center = (cx, cyv) 	
                # centro del circulo
                #cv.circle(framet, center, 1, (0, 100, 100), 3)
                # esquema del circulo
                radius = j[2]
                #cv.circle(framet, center, radius, (255, 0, 255), 3)
                minrec = (cx - radius, cyv - radius)
                maxrec = (cx + radius, cyv + radius)
                #rect = cv.rectangle(framet, minrec, maxrec, (255, 0, 255), 3)
            
                print(circles)
                
                frame = (framet[cx-radius:cx+radius,cyv-radius:cyv+radius])

                frameNN = cv.resize(frame,(30,30))

                tests = []
                tests.append(np.array(frameNN))
                X = np.array(tests)

                pred = np.argmax(model.predict(X))
                print("El número de la señal de tránsito es: ", pred)

                cv.imshow('video', frame)
    
    cv.imshow('videototal', framet)
 
    if(cv.waitKey(1) == ord('q')):
            break
 
capture.release()
cv.destroyAllWindows()