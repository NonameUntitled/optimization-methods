from task3 import gradient_descent, compute_func, compute_grad

import copy
import numpy as np


if __name__ == '__main__':
    x_1 = np.random.randn(1) * 5
    x_2 = np.random.randn(2) * 5
    x_3 = np.random.randn(3) * 5

    b_1 = np.random.randn(1) * 5
    b_2 = np.random.randn(2) * 5
    b_3 = np.random.randn(3) * 5

    c = np.random.randn(1) * 10

    # K = 1
    matrix_1 = np.array([[1]], dtype=np.float64)
    step_count, _, _, _, _ = gradient_descent(
        lambda x: compute_func(matrix_1, b_1, c, x),
        lambda x: compute_grad(matrix_1, b_1, c, x),
        copy.deepcopy(x_1),
        learning_rate=0.1
    )
    print(f'K = 1, N = 1, Step count: {step_count}')

    matrix_2 = np.array([[1, 0], [0, 1]], dtype=np.float64) * 13
    step_count, _, _, _, _ = gradient_descent(
        lambda x: compute_func(matrix_2, b_2, c, x),
        lambda x: compute_grad(matrix_2, b_2, c, x),
        copy.deepcopy(x_2),
        learning_rate=0.1
    )
    print(f'K = 1, N = 2, Step count: {step_count}')

    matrix_3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64) * 4
    step_count, _, _, _, _ = gradient_descent(
        lambda x: compute_func(matrix_3, b_3, c, x),
        lambda x: compute_grad(matrix_3, b_3, c, x),
        copy.deepcopy(x_3),
        learning_rate=0.1
    )
    print(f'K = 1, N = 3, Step count: {step_count}')

    # K = 0.5
    matrix_1 = np.array([[0.5]], dtype=np.float64)
    step_count, _, _, _, _ = gradient_descent(
        lambda x: compute_func(matrix_1, b_1, c, x),
        lambda x: compute_grad(matrix_1, b_1, c, x),
        copy.deepcopy(x_1),
        learning_rate=0.1
    )
    print(f'K = 0.5, N = 1, Step count: {step_count}')

    matrix_2 = np.array([[2, 0], [0, 1]], dtype=np.float64) * 13
    step_count, _, _, _, _ = gradient_descent(
        lambda x: compute_func(matrix_2, b_2, c, x),
        lambda x: compute_grad(matrix_2, b_2, c, x),
        copy.deepcopy(x_2),
        learning_rate=0.1
    )
    print(f'K = 0.5, N = 2, Step count: {step_count}')

    matrix_3 = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=np.float64) * 4
    step_count, _, _, _, _ = gradient_descent(
        lambda x: compute_func(matrix_3, b_3, c, x),
        lambda x: compute_grad(matrix_3, b_3, c, x),
        copy.deepcopy(x_3),
        learning_rate=0.1
    )
    print(f'K = 0.5, N = 3, Step count: {step_count}')

    # K = 0.32
    matrix_1 = np.array([[0.32]], dtype=np.float64)
    step_count, _, _, _, _ = gradient_descent(
        lambda x: compute_func(matrix_1, b_1, c, x),
        lambda x: compute_grad(matrix_1, b_1, c, x),
        copy.deepcopy(x_1),
        learning_rate=0.1
    )
    print(f'K = 0.32, N = 1, Step count: {step_count}')

    matrix_2 = np.array([[2, 1], [1, 5]], dtype=np.float64) * 2
    step_count, _, _, _, _ = gradient_descent(
        lambda x: compute_func(matrix_2, b_2, c, x),
        lambda x: compute_grad(matrix_2, b_2, c, x),
        copy.deepcopy(x_2),
        learning_rate=0.1
    )
    print(f'K = 0.32, N = 2, Step count: {step_count}')

    matrix_3 = np.array([[1.64, 1, 1], [0, 0.7, 0], [0, 0, 2]], dtype=np.float64) * 5
    step_count, _, _, _, _ = gradient_descent(
        lambda x: compute_func(matrix_3, b_3, c, x),
        lambda x: compute_grad(matrix_3, b_3, c, x),
        copy.deepcopy(x_3),
        learning_rate=0.1
    )
    print(f'K = 0.32, N = 3, Step count: {step_count}')

