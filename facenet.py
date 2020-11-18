import numpy as np
import cv2 as cv
import tensorflow as tf
import tflite_runtime.interpreter as tflite
import glob
import os
import time
from haar import HAAR

EDGETPU_SHARED_LIB = 'libedgetpu.so.1'
MODEL = 'models/facenet_keras.h5'
TFLITE_MODEL = 'models/facenet_quant_postprocess.tflite'
EDGETPU_TFLITE_MODEL = 'models/facenet_quant_postprocess_edgetpu.tflite'


def convert():
    def representative_dataset_gen():
        for _ in range(1024):
            batch_imgs = np.array(np.random.rand(
                1, 160, 160, 3), dtype=np.float32)
            yield [batch_imgs]
    model = tf.keras.models.load_model(MODEL)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    open(TFLITE_MODEL, 'wb').write(tflite_model)


class FaceNet:
    def __init__(self, database=None):
        self.image_shape = (160, 160)
        self.database = database
        self.interpreter = tflite.Interpreter(
            model_path=EDGETPU_TFLITE_MODEL,
            experimental_delegates=[
                tflite.load_delegate(EDGETPU_SHARED_LIB)
            ])
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.haar = HAAR()
        self.anchors = self.__cache_anchors(self.database)
        self.required_num_images = 3
        self.signle_threshold = 0.75
        self.multiple_threshold = 0.5

    def __normalize(self, img):
        scale, zero_point = self.input_details[0]['quantization']
        img = cv.resize(img, self.image_shape)
        img = np.array(img, dtype=np.float32)
        img = img/255
        img = np.array(img/scale + zero_point, dtype=np.uint8)
        return img

    def __similarity(self, x, y):
        cosine = np.inner(x, y) / (np.linalg.norm(x)*np.linalg.norm(y))
        return cosine

    def __cache_anchors(self, ds):
        anchors = []
        if ds is None:
            return anchors
        names = next(os.walk(ds))[1]
        for name in names:
            features = []
            for filename in glob.glob(ds+'/'+name+'/*.jpg'):
                candidate = cv.imread(filename)
                anchor = self.normalize_anchor(candidate, self.image_shape)
                features.append(self.predict(anchor))
            anchors.append((name, features))
        return anchors

    def normalize_anchor(self, img, shape):
        (x, y, w, h) = self.haar.predict(img)[0]
        return cv.resize(img[y:y+h, x:x+w], shape)

    def predict(self, img):
        img = self.__normalize(img)
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.input_details[0]['index'], [img])
        self.interpreter.invoke()
        feature = self.interpreter.get_tensor(self.output_details[0]['index'])
        scale, zero_point = self.output_details[0]['quantization']
        feature = np.array(feature[0], dtype=np.float32)
        feature = (feature-zero_point)*scale
        return feature

    def verify(self, who, anchor):
        start = time.time()
        who_features = self.predict(who)
        anchor_features = self.predict(anchor)
        confidence = self.__similarity(who_features, anchor_features)
        end = time.time()
        print('Verification estimated time: {:.4f}'.format(end-start))
        if confidence > self.signle_threshold:
            return True, confidence
        else:
            return False, confidence

    def find(self, img):
        if self.database is None:
            return print('Warning: directory of database is not provided!')
        who_feature = self.predict(img)
        existing = False
        max_name = 'unknown'
        max_confidence = 0
        for (name, features) in self.anchors:
            sum_confidence = 0
            for anchor_feature in features:
                sum_confidence += self.__similarity(
                    who_feature, anchor_feature)
            confidence = sum_confidence/self.required_num_images
            existing = existing or (
                True if confidence > self.multiple_threshold else False)
            if existing and confidence > max_confidence:
                max_name = name
                max_confidence = confidence
        return existing, max_name, max_confidence
