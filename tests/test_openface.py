import unittest
import os
import scipy.misc as misc
from openface import api


class OpenFaceTestCase(unittest.TestCase):
    def test_raw_face_locations(self):
        filepath = os.path.join(os.path.dirname(__file__),
                                'images', 'obama.jpg')
        img = misc.imread(filepath)
        detected_faces = api._raw_face_locations(img)

        self.assertEqual(len(detected_faces), 1)
        self.assertEqual(detected_faces[0].top(), 142)
        self.assertEqual(detected_faces[0].bottom(), 409)

    def test_face_locations(self):
        filepath = os.path.join(os.path.dirname(__file__),
                                'images', 'obama.jpg')
        img = misc.imread(filepath)
        detected_faces = api.face_locations(img)

        self.assertEqual(len(detected_faces), 1)
        self.assertEqual(detected_faces[0], (349, 142, 617, 409))

    def test_raw_face_landmarks(self):
        filepath = os.path.join(os.path.dirname(__file__),
                                'images', 'obama.jpg')
        img = misc.imread(filepath)
        face_landmarks = api._raw_face_landmarks(img)
        example_landmark = face_landmarks[0].parts()[10]

        self.assertEqual(len(face_landmarks), 1)
        self.assertEqual(face_landmarks[0].num_parts, 68)
        self.assertEqual((example_landmark.x, example_landmark.y), (552, 399))

    def test_face_encodings(self):
        filepath = os.path.join(os.path.dirname(__file__),
                                'images', 'obama.jpg')
        img = misc.imread(filepath)
        encodings = api.face_encodings(img)

        self.assertEqual(len(encodings), 1)
        self.assertEqual(len(encodings[0]), 128)

    def test_face_distance(self):
        basepath = os.path.join(os.path.dirname(__file__),
                                'images')
        img_a1 = misc.imread(os.path.join(basepath, 'obama.jpg'))
        img_a2 = misc.imread(os.path.join(basepath, 'obama2.jpg'))
        img_a3 = misc.imread(os.path.join(basepath, 'obama3.jpg'))
        img_b1 = misc.imread(os.path.join(basepath, 'biden.jpg'))

        face_encoding_a1 = api.face_encodings(img_a1)[0]
        face_encoding_a2 = api.face_encodings(img_a2)[0]
        face_encoding_a3 = api.face_encodings(img_a3)[0]
        face_encoding_b1 = api.face_encodings(img_b1)[0]

        faces_to_compare = [face_encoding_a2,
                            face_encoding_a3,
                            face_encoding_b1]

        distance_results = api.face_distance(faces_to_compare,
                                             face_encoding_a1)

        self.assertEqual(type(distance_results), list)
        self.assertEqual(len(distance_results), 3)
        self.assertLessEqual(distance_results[0], 0.6)
        self.assertLessEqual(distance_results[1], 0.6)
        self.assertGreaterEqual(distance_results[2], 0.6)


if __name__ == '__main__':
    unittest.main()
