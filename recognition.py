import cv2

thres = 0.45  #intensity of pixel

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as c: #reading coco names
    classNames = c.read().rstrip('\n').split('\n')
    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    while True:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=thres)#threshold of image
        print(classIds, bbox)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                print(classId,"-----------------------------")
                print(classIds,"++++++++++++++++++++++++")
                cv2.rectangle(img, box, color=(0, 0, 255), thickness=2) #detection box

        cv2.imshow('Output', img)
        if (cv2.waitKey(1)) == ord('q'): #to stop camera
            break
    cap.release()

    cv2.destroyAllWindows()