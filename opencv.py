import cv2
import time
i=0
def kiba():
	i=0
	while i<10:
		video = cv2.VideoCapture(0)
		i = str(i)
		path = "/root/ai/"+i+".png"
		i = int(i)
		print path
		ret, frame = video.read()
		print frame
		cv2.imwrite(path, frame)
		i=i+1
		time.sleep(2)
		video.release()
def bal():
	 ret, frame = video.read()
 	 print frame
 	 cv2.imwrite("kiba5.png", frame)
         video.release()
kiba()
