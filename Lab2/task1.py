import time
import copy
import numpy as np
from typing import Callable


def compute_function(q_matrix: np.ndarray, b_vector: np.ndarray, c_value: float, x: np.ndarray):
    return 1 / 2 * np.dot(q_matrix @ x, x) + np.dot(b_vector, x) + c_value


def compute_grad(q_matrix: np.ndarray, b_vector: np.ndarray, c_value: float, x: np.ndarray):
    return q_matrix @ x + b_vector


def make_positive_matrix(h: np.ndarray, start_eta: float = 1e-4):
    eta = 0

    while True:
        new_h = np.eye(*h.shape) * eta + h
        is_positive = True

        for i in range(new_h.shape[0]):
            det = np.linalg.det(new_h[:i + 1, :i + 1])
            if det <= 0:
                is_positive = False
                break

        if is_positive:
            break

        eta = eta * 2 if eta != 0 else start_eta

    return new_h


def advanced_gradient_descent(
        function: Callable[[np.ndarray], float],
        function_grad: Callable[[np.ndarray], np.ndarray],
        function_hessian: Callable[[np.ndarray], np.ndarray],
        start_x: np.ndarray,
        eps: float = 1e-8,
):
    step_count = 0
    function_calls_count = 0
    function_grad_calls_count = 0
    basis = []
    start_time = time.time()

    def function_with_counter(x: np.ndarray) -> float:
        nonlocal function_calls_count
        function_calls_count += 1
        return function(x)

    def function_grad_with_counter(x: np.ndarray) -> np.ndarray:
        nonlocal function_grad_calls_count
        function_grad_calls_count += 1
        return function_grad(x)

    curr_x = copy.deepcopy(start_x)
    args_history = [copy.deepcopy(curr_x)]

    while True:
        anti_grad = -function_grad_with_counter(curr_x)
        q_matrix = function_hessian(curr_x)

        if np.linalg.norm(anti_grad) < eps:
            break

        if not basis:
            step_direction = copy.deepcopy(anti_grad)
        else:
            step_direction = anti_grad - np.dot(q_matrix @ step_direction, anti_grad) / \
                             np.dot(q_matrix @ step_direction, step_direction) * step_direction

        step_size = np.dot(step_direction, anti_grad) / np.dot(q_matrix @ step_direction, step_direction)

        if step_size > 1e8:
            step_size = 0

        curr_x = curr_x + step_size * step_direction
        args_history.append(copy.deepcopy(curr_x))

        step_count += 1
        basis.append(step_direction)

    algorithm_converge_time = time.time() - start_time
    min_arg = curr_x
    min_target = function_with_counter(curr_x)

    return \
        step_count, \
        function_calls_count, \
        function_grad_calls_count, \
        min_arg, \
        min_target, \
        algorithm_converge_time, \
        args_history


def newton_method(
        function: Callable[[np.ndarray], float],
        function_grad: Callable[[np.ndarray], np.ndarray],
        function_hessian: Callable[[np.ndarray], np.ndarray],
        start_x: np.ndarray,
        eps: float = 1e-8
):
    step_count = 0
    function_calls_count = 0
    function_grad_calls_count = 0
    function_hessian_calls_count = 0
    start_time = time.time()
    args_history = [copy.deepcopy(start_x)]

    def function_with_counter(x: np.ndarray) -> float:
        nonlocal function_calls_count
        function_calls_count += 1
        return function(x)

    def function_grad_with_counter(x: np.ndarray) -> np.ndarray:
        nonlocal function_grad_calls_count
        function_grad_calls_count += 1
        return function_grad(x)

    def function_hessian_with_counter(x: np.ndarray) -> np.ndarray:
        nonlocal function_hessian_calls_count
        function_hessian_calls_count += 1
        return function_hessian(x)

    curr_x = copy.deepcopy(start_x)

    while True:
        grad = function_grad_with_counter(curr_x)
        if np.linalg.norm(grad) < eps:
            break

        hessian = function_hessian_with_counter(curr_x)

        try:
            inv_hessian = np.linalg.inv(hessian)
        except:
            new_hessian = make_positive_matrix(hessian)
            inv_hessian = np.linalg.inv(new_hessian)

        step_count += 1
        curr_x = curr_x - inv_hessian @ grad
        args_history.append(curr_x)

    algorithm_converge_time = time.time() - start_time
    min_arg = curr_x
    min_target = function_with_counter(curr_x)

    return \
        step_count, \
        function_calls_count, \
        function_grad_calls_count, \
        function_hessian_calls_count, \
        min_arg, \
        min_target, \
        algorithm_converge_time, \
        args_history


if __name__ == '__main__':
    q_matrix = np.array([[3, 4, 0], [4, -3, 0], [0, 0, 5]], dtype=np.float64)
    b_vector = np.array([1, 5, 9], dtype=np.float64)
    c_value = 0.

    advanced_gradient_descent_result = advanced_gradient_descent(
        lambda x: compute_function(q_matrix, b_vector, c_value, x),
        lambda x: compute_grad(q_matrix, b_vector, c_value, x),
        lambda x: q_matrix,
        np.array([12, 1337, 159], dtype=np.float64)
    )

    newton_method_result = newton_method(
        lambda x: compute_function(q_matrix, b_vector, c_value, x),
        lambda x: compute_grad(q_matrix, b_vector, c_value, x),
        lambda x: q_matrix,
        np.array([0, 0, 0], dtype=np.float64)
    )

    advanced_step_count = advanced_gradient_descent_result[0]
    advanced_function_calls_count = advanced_gradient_descent_result[1]
    advanced_function_grad_calls_count = advanced_gradient_descent_result[2]
    advanced_min_arg = advanced_gradient_descent_result[3]
    advanced_min_target = advanced_gradient_descent_result[4]
    advanced_algorithm_converge_time = advanced_gradient_descent_result[5]

    newton_step_count = newton_method_result[0]
    newton_function_calls_count = newton_method_result[1]
    newton_function_grad_calls_count = newton_method_result[2]
    newton_function_hessian_calls_count = newton_method_result[3]
    newton_min_arg = newton_method_result[4]
    newton_min_target = newton_method_result[5]
    newton_algorithm_converge_time = newton_method_result[6]

    print(advanced_step_count, advanced_min_arg, advanced_min_target)
    print(newton_step_count, newton_min_arg, newton_min_target)
