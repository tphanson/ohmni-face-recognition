import time
import cv2 as cv
from facenet import FaceNet
from haar import HAAR


ANCHOR_SHAPE = (96, 96)


def verify(anchor_path):
    camera = cv.VideoCapture(0)
    haar = HAAR()
    facenet = FaceNet()
    img = cv.imread(anchor_path)
    anchor = facenet.normalize_anchor(img, ANCHOR_SHAPE)
    while True:
        start = time.time()
        _, img = camera.read()
        drawed_img = img.copy()
        drawed_img[:ANCHOR_SHAPE[0], :ANCHOR_SHAPE[1], :] = anchor
        faces = haar.predict(img)
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            verified, confidence = facenet.verify(face, anchor)
            color = (0, 255, 0) if verified else (0, 0, 255)
            cv.putText(drawed_img, str(confidence), (x+5, y+15),
                       cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            cv.rectangle(drawed_img, (x, y), (x+w, y+h), color, 2)
        cv.imshow('Camera', drawed_img)
        end = time.time()
        print('Estimated time: {:.4f}'.format(end-start))
        if cv.waitKey(10) & 0xFF == ord('q'):
            break


def find():
    camera = cv.VideoCapture(0)
    haar = HAAR()
    facenet = FaceNet('database')
    while True:
        start = time.time()
        _, img = camera.read()
        drawed_img = img.copy()
        faces = haar.predict(img)
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            existing, name, confidence = facenet.find(face)
            color = (0, 255, 0) if existing else (0, 0, 255)
            cv.rectangle(drawed_img, (x, y), (x+w, y+h), color, 2)
            cv.putText(drawed_img, name, (x+5, y+15),
                       cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            cv.putText(drawed_img, str(confidence), (x+5, y+30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        cv.imshow('Camera', drawed_img)
        end = time.time()
        print('Total estimated time: {:.4f}'.format(end-start))
        if cv.waitKey(10) & 0xFF == ord('q'):
            break


# Main
# verify('database/tuphan/3.jpg')
find()
