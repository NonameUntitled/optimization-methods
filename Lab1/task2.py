import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
from math import exp

from task1 import dichotomy, golden_slice, fibonacci


def linear_search(
        function: Callable[[float], float],
        left_arg: float,
        right_arg: float,
        step_size: float,
        debug: bool = False
) -> Tuple[int, float, float, List[float], List[float]]:
    step_count = 0
    arg_min = left_arg
    min_target = function(arg_min)
    arg_history = [arg_min]
    target_history = [min_target]

    while left_arg + step_size * step_count <= right_arg:
        arg = left_arg + step_size * step_count
        target = function(arg)
        if target < min_target:
            arg_min = arg
            min_target = target
            arg_history.append(arg_min)
            target_history.append(min_target)
        step_count += 1

        if debug:
            print(f'Arg: {arg}, Target: {target}, MinArg: {arg_min}, MinTarget: {min_target}')

    return step_count, arg_min, min_target, arg_history, target_history


def gradient_descend(
        function: Callable[[float], float],
        function_diff: Callable[[float], float],
        learning_rate: float,
        start_arg: float,
        eps: float = 1e-8,
        debug: bool = False
) -> Tuple[int, float, float, List[float], List[float]]:
    step_count = 0
    pred_x = None
    arg_min = start_arg
    arg_history = [arg_min]
    target_history = [function(arg_min)]

    while pred_x is None or abs(pred_x - arg_min) > eps:
        if step_count > 1e4 or target_history[-1] > 1e12:
            return_value = 1e8 + learning_rate * 10
            return int(return_value), return_value, return_value, [], []  # Hack to suppress big lr values

        grad = function_diff(arg_min)
        pred_x = arg_min
        arg_min -= learning_rate * function_diff(arg_min)

        step_count += 1
        arg_history.append(arg_min)
        target_history.append(function(arg_min))

        if debug:
            print(grad, function_diff(arg_min))

    minimum_target = function(arg_min)

    return step_count, arg_min, minimum_target, arg_history, target_history


def methods_comparison(
        function: Callable[[float], float],
        function_grad: Callable[[float], float],
        title: str,
        left_arg: float,
        right_arg: float,
        step_size: float = 1,
        learning_rate: float = 1e-4,
        create_plot: bool = False
):
    linear_step_count, linear_arg_min, linear_min_target, linear_arg_history, linear_target_history = \
        linear_search(function, left_arg, right_arg, step_size)
    grad_step_count, grad_arg_min, grad_min_target, grad_arg_history, grad_target_history = \
        gradient_descend(function, function_grad, learning_rate, left_arg)
    print(title)
    print(f'Step count. Linear: {linear_step_count}. Gradient: {grad_step_count}')
    print(f'Arg min. Linear: {linear_arg_min}. Gradient: {grad_arg_min}')
    print(f'Min target. Linear: {linear_min_target}. Gradient: {grad_min_target}')

    if create_plot:
        xs = np.linspace(left_arg, right_arg, 1000)
        ys = np.array(list(map(function, xs)))
        plt.plot(xs, ys, c='r', label='Function')
        plt.plot(grad_arg_history, grad_target_history, label='Gradient', c='orange')
        plt.scatter(grad_arg_history, grad_target_history, c='orange')
        plt.plot(linear_arg_history, linear_target_history, label='Linear', c='b')
        plt.scatter(linear_arg_history, linear_target_history, c='b')
        plt.title(title)
        plt.legend()
        plt.show()


def find_optimal_steps(
        function: Callable[[float], float],
        function_grad: Callable[[float], float],
        left_lr: float,
        right_lr: float,
        left_step_size: float,
        right_step_size: float,
        left_arg: float = -100,
        right_arg: float = 100
):
    grad_target_fun = lambda x: gradient_descend(function, function_grad, x, left_arg)[0]
    linear_target_fun = lambda x: linear_search(function, left_arg, right_arg, x)[2]

    _, _, optimal_learning_rate, _, _ = \
        golden_slice(grad_target_fun, left_lr, right_lr)

    _, _, optimal_step_size, _, _ = \
        golden_slice(linear_target_fun, left_step_size, right_step_size)

    _, _, optimal_learning_rate_2, _, _ = \
        dichotomy(grad_target_fun, left_lr, right_lr)

    _, _, optimal_step_size_2, _, _ = \
        dichotomy(linear_target_fun, left_step_size, right_step_size)

    random_learning_rate = random.random() * (right_lr - left_lr) + left_lr
    random_step_size = random.random() * (right_step_size - left_step_size) + left_step_size
    print(f"golden size {optimal_step_size} : rate {optimal_learning_rate}")
    print(f"dichotomy size {optimal_step_size_2} : rate {optimal_learning_rate_2}")
    print(f"random size {random_step_size} : rate {random_learning_rate}")

    methods_comparison(
        function, function_grad, 'Random',
        left_arg=left_arg,
        right_arg=right_arg,
        step_size=random_step_size,
        learning_rate=random_learning_rate,
        create_plot=True
    )

    methods_comparison(
        function, function_grad, 'Optimal golden',
        left_arg=left_arg,
        right_arg=right_arg,
        step_size=optimal_step_size,
        learning_rate=optimal_learning_rate,
        create_plot=True
    )

    methods_comparison(
        function, function_grad, 'Optimal dichotomy',
        left_arg=left_arg,
        right_arg=right_arg,
        step_size=optimal_step_size,
        learning_rate=optimal_learning_rate_2,
        create_plot=True
    )


if __name__ == '__main__':
    # First function
    fst_function = lambda x: x ** 2 + 10 * x
    fst_function_grad = lambda x: 2 * x + 10

    find_optimal_steps(
        fst_function, fst_function_grad,
        1e-2, 1,
        1e-2, 40,
        -40, 40
    )

    # Second function
    snd_function = lambda x: x ** 2 + 20 * x + 12
    snd_function_grad = lambda x: 2 * x + 20

    find_optimal_steps(
        snd_function, snd_function_grad,
        1e-2, 1,
        1e-2, 40,
        -40, 40
    )

    # Last function
    last_function = lambda x: x ** 2
    last_function_grad = lambda x: 2 * x

    find_optimal_steps(
        last_function, last_function_grad,
        1e-2, 1,
        1e-2, 40,
        -40, 40
    )

