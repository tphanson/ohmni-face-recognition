import cv2 as cv


class HAAR:
    def __init__(self):
        self.detector = cv.CascadeClassifier(
            cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect(self, img):
        faces = self.detector.detectMultiScale(img, 1.3, 5)
        return faces

    def align(self):
        pass

    def predict(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return self.detect(img)
