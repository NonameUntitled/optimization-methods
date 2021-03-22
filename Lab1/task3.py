import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List


def generate_q_matrix(eigenvalues_diff: float) -> np.ndarray:
    basic_matrix = np.array([[2, 3], [4, 1]], dtype=np.float64)
    j_matrix = np.array([[1, 0], [0, 1 * eigenvalues_diff]], dtype=np.float64)
    q = basic_matrix @ j_matrix @ np.linalg.inv(basic_matrix)
    return q


def compute_func(q_matrix: np.ndarray, b_vector: np.ndarray, c_value: float, x: np.ndarray) -> float:
    return np.dot(q_matrix @ x, x) + np.dot(b_vector, x) + c_value


def compute_grad(q_matrix: np.ndarray, b_vector: np.ndarray, c_value: float, x: np.ndarray) -> np.ndarray:
    return q_matrix @ x + q_matrix.T @ x + b_vector


def print_function_view(q_matrix: np.ndarray, b_vector: np.ndarray, c_value: float):
    print(f'Function view = '
          f'{round(q_matrix[0, 0], 3)} * x^2 + '
          f'({round(q_matrix[0, 1] + q_matrix[1, 0], 3)} * x * y) + '
          f'({round(q_matrix[1, 1], 3)} * y^2) + '
          f'({round(b_vector[0], 3)} * x) + '
          f'({round(b_vector[1], 3)} * y) + '
          f'({round(c_value[0], 3)})'
          )


def gradient_descent(
        function: Callable[[np.ndarray], float],
        function_grad: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        learning_rate: float = 1e-3,
        eps: float = 1e-5,
) -> Tuple[int, np.ndarray, float, List[np.ndarray], List[float]]:
    step_count = 0
    args_history = [copy.deepcopy(x)]
    function_history = [function(x)]

    pred_x = None

    while pred_x is None or np.linalg.norm(pred_x - x) > eps:
        grad = function_grad(x)
        pred_x = copy.deepcopy(x)
        x -= learning_rate * grad

        step_count += 1
        args_history.append(copy.deepcopy(x))
        function_history.append(copy.deepcopy(function(x)))

    return step_count, x, function(x), args_history, function_history


if __name__ == '__main__':

    for eigenvalues_diff in [1, 3, 5]:
        q_matrix = generate_q_matrix(eigenvalues_diff)
        b_vector = (np.random.randn(2) - 0.5) * 5
        c_value = np.random.randn(1) * 10
        x_vector = np.random.randn(2) * 5

        step_count, argmin, target_min, args_history, function_history = \
            gradient_descent(
                lambda x: compute_func(q_matrix, b_vector, c_value, x),
                lambda x: compute_grad(q_matrix, b_vector, c_value, x),
                x_vector,
                learning_rate=0.15
            )

        print_function_view(q_matrix, b_vector, c_value)
        print(f'K = {1 / eigenvalues_diff} N = {2}')
        print(f'Steps: {step_count}')
        print(f'Argmin: {argmin}')
        print(f'Min: {target_min}')
        print()

        args_history_x = list(map(lambda pnt: pnt[0], args_history))
        args_history_y = list(map(lambda pnt: pnt[1], args_history))

        plt.plot(args_history_x, args_history_y)
        plt.scatter(args_history_x, args_history_y)
        plt.title(f'K = {1 / eigenvalues_diff} N = {2}. Arguments history')
        plt.grid()
        plt.show()
