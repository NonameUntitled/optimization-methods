import numpy as np
from typing import Callable, List

from task1 import advanced_gradient_descent, newton_method


def compare_convergence_process(
        function: Callable[[np.ndarray], float],
        function_grad: Callable[[np.ndarray], np.ndarray],
        function_hessian: Callable[[np.ndarray], np.ndarray],
        start_xs: List[np.ndarray],
        show_plots: bool = False
):
    for start_x in start_xs:
        advanced_gradient_descent_result = advanced_gradient_descent(
            function,
            function_grad,
            function_hessian,
            start_x
        )

        newton_method_result = newton_method(
            function,
            function_grad,
            function_hessian,
            start_x
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

        print(
            advanced_step_count, advanced_min_arg,
            round(advanced_min_target, 8),
            round(advanced_algorithm_converge_time, 8)
        )

        print(
            newton_step_count, newton_min_arg,
            round(newton_min_target, 8),
            round(newton_algorithm_converge_time, 8)
        )

        print()


if __name__ == '__main__':
    start_xs = [
        np.array([0, 0], dtype=np.float64),
        np.array([100, 100], dtype=np.float64),
        np.array([-200, -500], dtype=np.float64),
        np.array([1337, 159], dtype=np.float64),
        np.array([-5343, 2323], dtype=np.float64),
    ]

    # Fst function
    fst_function = lambda x: 100 * (x[1] - x[0]) ** 2 + (1 - x[0]) ** 2
    fst_function_grad = lambda x: np.array([-200 * x[1] + 202 * x[0] - 2, 200 * x[1] - 200 * x[0]], dtype=np.float64)
    fst_function_hessian = lambda x: np.array([[202, -200], [-200, 200]], dtype=np.float64)

    compare_convergence_process(
        fst_function,
        fst_function_grad,
        fst_function_hessian,
        start_xs
    )

    # Snd function
    snd_function = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    snd_function_grad = lambda x: np.array(
        [-400 * x[0] * x[1] + 400 * x[0] ** 3 - 2 + 2 * x[0], 200 * x[1] - 200 * x[0] ** 2],
        dtype=np.float64
    )
    snd_function_hessian = lambda x: np.array(
        [
            [-400 * x[1] + 2 + 1200 * x[0] ** 2, -400 * x[0]],
            [-400 * x[0], 200]
        ], dtype=np.float64
    )

    compare_convergence_process(
        snd_function,
        snd_function_grad,
        snd_function_hessian,
        start_xs
    )

    # Third function
    # TODO write maximization

