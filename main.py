import cv2
import numpy as np

def FindObjects(outputs, frame):
    height, width, c = frame.shape
    bbox = []
    classIds = []
    confidence = []

    for output in outputs:
        for det in output:
            score = det[5:]
            classId = np.argmax(score)
            conf = score[classId]

            if conf > Threshold_confidence:
                x_, y_, w_, h_ = det[0], det[1], det[2], det[3]
                wbox, hbox = int(w_*width), int(h_*height)
                xbox, ybox = int(x_*width-wbox/2), int(y_*height-hbox/2)

                classIds.append(classId)
                confidence.append(float(conf))
                bbox.append([xbox,ybox,wbox,hbox])

    indices = cv2.dnn.NMSBoxes(bbox, confidence, Threshold_confidence, nms_threshold=0.1)
    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(frame,(x,y),(x+wbox,y+hbox), (0,255,0), 2)
        cv2.putText(frame, f'{classNames[classIds[i]].upper()} {int(confidence[i]*100)}%', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,0),2)




##############################
classFile = 'coconames.txt'
cfg, weights = 'yolo-tiny.cfg', 'yolov2-tiny.weights'
Threshold_confidence = 0.2 #0.5
NMS_confidence = 0.1  #0.3
#############################

cap = cv2.VideoCapture(0)

with open(classFile, 'r') as f:
    classNames = f.read().strip().split('\n')

net = cv2.dnn.readNetFromDarknet(cfg, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

while True:
    _, frame = cap.read()
    #height, width, c = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), [0, 0, 0], crop=False)
    net.setInput(blob)

    output_layers_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers_names)
    FindObjects(outputs, frame)
    
    cv2.imshow('Cámara', frame)

    if cv2.waitKey(10) and 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
