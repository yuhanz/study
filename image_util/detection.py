import numpy as np
import pdb

def sample_rows(num_rows, num_row_samples = 20):
    return[i for i in map(lambda x: int(np.random.random() * num_rows), range(0,num_row_samples))]

def detect_horizontal_shift_monte_carlo(image_array1, image_array2):
    num_rows = len(image_array1)
    sample_row_indices = sample_rows(num_rows)
    return detect_horizontal_shift(image_array1, image_array2, sample_row_indices)

def detect_horizontal_shift_all_rows(image_array1, image_array2):
    num_rows = len(image_array1)
    sample_row_indices = list(range(0,num_rows))
    return detect_horizontal_shift(image_array1, image_array2, sample_row_indices)

def detect_horizontal_shift(image_array1, img_array2, sample_row_indices, sample_width = 20):
    sample_rows_1 = np.array([x for x in map(lambda i: image_array1[i, 0:sample_width], sample_row_indices)])
    sample_rows_2 = np.array([x for x in map(lambda i: img_array2[i, 0:sample_width], sample_row_indices)])
    pdb.set_trace()
    scores = list(map(lambda i: np.sum(np.ndarray.flatten(np.absolute(sample_rows_1[:, 0:sample_width-i] - sample_rows_2[:, i:]))), range(0,sample_width)))
    scores = list(reversed(scores)) + list(map(lambda i: np.sum(np.ndarray.flatten(np.absolute(sample_rows_2[:, 0:sample_width-i] - sample_rows_1[:, i:]))), range(1,sample_width)))

    print('sample1:', sample_rows_1)
    print('sample2:', sample_rows_2)
    print('scores:', scores)
    index =  np.ndarray.argmin(np.array(scores))
    direction = index + 1 - sample_width
    return [direction, scores[index]]
