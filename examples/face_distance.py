import cv2
import openface


# Load some images to compare against
known_obama_image = cv2.imread('obama.jpg')
known_biden_image = cv2.imread('biden.jpg')

# Get the face encodings for the known images
obama_face_encoding = openface.face_encodings(known_obama_image)[0]
biden_face_encoding = openface.face_encodings(known_biden_image)[0]

known_encodings = [
    obama_face_encoding,
    biden_face_encoding
]

# Load a test image and get encodings for it
image_to_test = cv2.imread('obama2.jpg')
image_to_test_encoding = openface.face_encodings(image_to_test)[0]

# See how for apart the test image is fro the known faces
face_distances = openface.face_distance(known_encodings,
                                        image_to_test_encoding)


for i, face_distance in enumerate(face_distances):
    print('The test image has a distance of {:.2} from known image #{}'.format(
          face_distance, i))
    print(('- With a normal cutoff of 0.6, would the test image match '
           'the known image? {}').format(face_distance < 0.6))
    print(('- With a very strict cutoff of 0.5, would the test image match the'
           ' known image? {}').format(face_distance < 0.5))
    print('')
