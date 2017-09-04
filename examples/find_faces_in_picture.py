import openface
import cv2


# Load the jpg file into a numpy array
image = cv2.imread('biden.jpg')

# Find all the faces in the image
face_locations = openface.face_locations(image)

if (len(face_locations) == 0):
    print('Found no faces')
else:
    for face_location in face_locations:
        # Print the location of each face in this image
        left, top, right, bottom = face_location
        print(('A face is located at pixel location Left: {}, Top: {}, '
               'Right: {}, Bottom: {}').format(left, top, right, bottom))

        # Draw bounding box
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 1)
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)
