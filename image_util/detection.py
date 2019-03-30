import numpy as np
import pdb

def sample_rows(num_rows, num_row_samples = 30):
    return[i for i in map(lambda x: int(np.random.random() * num_rows), range(0,num_row_samples))]

def detect_horizontal_shift_monte_carlo(image_array1, image_array2):
    num_rows = len(image_array1)
    sample_row_indices = sample_rows(num_rows)
    return detect_horizontal_shift(image_array1, image_array2, sample_row_indices)

def count_zero_pixels(image_array):
    is_zero_pixels = (np.sum((image_array == 0) + [0,0,0], axis=2) == 3) + 0
    return np.sum(is_zero_pixels)

def detect_horizontal_shift_all_rows(image_array1, image_array2):
    num_rows = len(image_array1)
    sample_row_indices = list(range(0,num_rows))
    return detect_horizontal_shift(image_array1, image_array2, sample_row_indices)

def detect_horizontal_shift(image_array1, image_array2, sample_row_indices, sample_width = 12):
    samples_rows_list = map(lambda img: np.array(list(map(lambda i: img[i, 0:sample_width], sample_row_indices))), [image_array1, image_array2])

    [sample_rows_1, sample_rows_2] = samples_rows_list

    scores_left = list(map(lambda i: count_zero_pixels(sample_rows_1[:, 0:sample_width-i] - sample_rows_2[:, i:]), range(0,sample_width)))
    scores_right = list(map(lambda i: count_zero_pixels(sample_rows_2[:, 0:sample_width-i] - sample_rows_1[:, i:]), range(1,sample_width)))
    scores2 = list(reversed(scores_left)) + scores_right

    index =  np.ndarray.argmax(np.array(scores2))
    direction = index + 1 - sample_width
    return [direction, scores2[index]]
