import numpy as np # type: ignore
import cv2 # type: ignore

def compute_vector_length(vector):
    len_of_vector = np.linalg.norm(vector)
    return len_of_vector

def compute_dot_product(vector1, vector2):
    result = np.dot(vector1, vector2)
    return result

def matrix_multi_vector(matrix, vector):
    result = np.dot(matrix, vector)
    return result

def matrix_multi_matrix(matrix1, matrix2):
    result = np.dot(matrix1, matrix2)
    return result

def inverse_matrix(matrix):
    result = np.linalg.inv(matrix)
    return result

def compute_eigenvalues_eigenvectors(matrix):
    eigenvalues, eigenvector = np.linalg.eig(matrix)
    return eigenvalues, eigenvector

def compute_cosine(v1, v2):
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cos_sim

def bg_subtraction(file_path1, file_path2, file_path3):
    bg1_image = cv2.imread(file_path1, 1)
    ob_image = cv2.imread(file_path2, 1)
    bg2_image = cv2.imread(file_path3, 1)

    bg1_image = cv2.resize(bg1_image, (678, 381))
    ob_image = cv2.resize(ob_image, (678, 381))
    bg2_image = cv2.resize(bg2_image, (678, 381))

    def compute_difference(bg1_image, ob_image):
        difference_single_channel = cv2.absdiff(bg1_image, ob_image)
        return difference_single_channel

    difference_single_channel = compute_difference(bg1_image, ob_image)

    def compute_binary_mask(difference_single_channel):
        binary_mask = np.where(difference_single_channel > 30, 255, 0)
        return binary_mask

    binary_mask = compute_binary_mask(difference_single_channel)
    output_img = np.where(binary_mask == 255, ob_image, bg2_image)
    cv2.imwrite('data/output.png', output_img)

bg_subtraction('data/GreenBackground.png', 'data/Object.png', 'data/NewBackground.jpg')


