import dlib
import numpy as np
import models
import utils

_face_detector = dlib.get_frontal_face_detector()

_predictor_model = models.pose_predictor_model_location()
_post_predictor = dlib.shape_predictor(_predictor_model)

_face_recognition_model = models.face_recognition_model_location()
_face_encoder = dlib.face_recognition_model_v1(_face_recognition_model)


def face_distance(face_encodings, face_to_compare):
    distances = []
    if len(face_encodings) == 0:
        return distances
    for encoding in face_encodings:
        distances.append(np.linalg.norm(encoding - face_to_compare))

    return distances


def _raw_face_locations(img, number_of_times_to_upsample=1):
    return _face_detector(img, number_of_times_to_upsample)


def face_locations(img, number_of_times_to_upsample=1):
    return [utils.trim_bb_to_bounds(utils.rect_to_bb(face), img.shape) for face
            in _raw_face_locations(img, number_of_times_to_upsample=1)]


def _raw_face_landmarks(face_image, face_locations=None):
    if face_locations is None:
        face_locations = _raw_face_locations(face_image)
    else:
        face_locations = [utils.bb_to_rect(face_location) for face_location in
                          face_locations]
    return [_post_predictor(face_image, face_location) for face_location in
            face_locations]


def face_encodings(face_image, known_face_locations=None, num_jitters=1):
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations)

    encodings = []
    for raw_landmark_set in raw_landmarks:
        descriptor = _face_encoder.compute_face_descriptor(face_image,
                                                           raw_landmark_set,
                                                           num_jitters)
        encodings.append(np.array(descriptor))
    return encodings
