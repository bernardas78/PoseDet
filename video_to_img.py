import cv2

#cap = cv2.VideoCapture("VID_20250905_070444.mp4")
cap = cv2.VideoCapture("ALDI.mp4")
i = 1

while True:
    ret, frame = cap.read()
    # print ("ret: {}".format(ret))
    if not ret:
        break
		
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
	
    cv2.imwrite(f"imgs\\{i:04d}.jpg", frame)
    i += 1

cap.release()
