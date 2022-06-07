import numpy as np
import cv2
# path = 'D:/OneDrive - The Hong Kong Polytechnic University/Postdoc_works/2020-first_half/Research works/TMECH2020/Experiments_2020_Trans/experiment_trans/laparoscopic_phantom/Exp2/'
# cap = cv2.VideoCapture(path + 'experiment_1.avi')


path = 'D:/OneDrive - The Hong Kong Polytechnic University/Postdoc_works/2020-first_half/Research works/TMECH2020/Experiments_2020_Trans/experiment_trans/porcine_pork/Exp11/'
cap = cv2.VideoCapture(path + 'experiment_1.avi')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(path + 'third_view.avi', fourcc, 20.0, (640 * 1,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = frame[0: 480, 640 * 2 : 640 * 3]
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out.write(frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
cap.release()
cv2.destroyAllWindows()