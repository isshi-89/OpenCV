import cv2
import numpy as np

qrCodeDetector = cv2.QRCodeDetector()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    
    ret,frame = cap.read()
    if not ret:
        break

    decodedText, points,_=qrCodeDetector.detectAndDecode(frame)

    if points is not None:
        points = points[0]
        

        center_x = int(np.mean(points[:,0]))
        center_y = int(np.mean(points[:,1]))

        b,g,r = frame[center_y, center_x]

        hex_color = "#{:02X}{:02X}{:02X}".format(r,g,b)

        cv2.circle(frame, (center_x,center_y),5,(0,255,0),-1)
        cv2.putText(frame,hex_color,(center_x + 10,center_y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        
        
        for i in range(len(points)):
            start_point = tuple(map(int,points[i]))
            end_point = tuple(map(int,points[(i + 1) % len(points)]))

            cv2.line(frame, start_point,end_point, (0,255,0),3)
                
       
        print(f"Color:{hex_color} at ({center_x},{center_y})")
        print(f"Decoded: {decodedText}")

       

    cv2.imshow('QR Color Picker',frame )

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
