import dlib
import numpy as np


def trim_bb_to_bounds(bounding_box, image_shape):
    return (max(bounding_box[0], 0),
            max(bounding_box[1], 0),
            min(bounding_box[2], image_shape[1]),
            min(bounding_box[3], image_shape[1]))


def rect_to_bb(rectangle):
    # take a bounding predicted by dlib and convert it
    # to the format (left, top, right, bottom)
    return (rectangle.left(), rectangle.top(),
            rectangle.right(), rectangle.bottom())


def bb_to_rect(bounding_box):
    rect = dlib.rectangle(left=bounding_box[0],
                          top=bounding_box[1],
                          right=bounding_box[2],
                          bottom=bounding_box[3])
    return rect


def area_of_bb(bounding_box):
    (left, top, right, bottom) = bounding_box
    width = right - left
    height = bottom - left

    return width * height


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    num_parts = shape.num_parts
    coords = np.zeros((num_parts, 2), dtype=dtype)

    # loop over all landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords
