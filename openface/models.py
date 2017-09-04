import os

_basedir = os.path.abspath(os.path.dirname(__file__))


def pose_predictor_model_location():
    return os.path.join(_basedir, 'models',
                        'shape_predictor_68_face_landmarks.dat')


def face_recognition_model_location():
    return os.path.join(_basedir, 'models',
                        'dlib_face_recognition_resnet_model_v1.dat')
