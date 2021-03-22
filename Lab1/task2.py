import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List


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
        eps: float = 1e-8,
        debug: bool = False
) -> Tuple[int, float, float, List[float], List[float]]:
    step_count = 0
    pred_x = None
    arg_min = np.random.random(1)[0]
    arg_history = [arg_min]
    target_history = [function(arg_min)]

    while pred_x is None or abs(pred_x - arg_min) > eps:
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
        eps: float = 1e-8,
):
    linear_step_count, linear_arg_min, linear_min_target, linear_arg_history, linear_target_history = \
        linear_search(function, left_arg, right_arg, step_size)
    grad_step_count, grad_arg_min, grad_min_target, grad_arg_history, grad_target_history = \
        gradient_descend(function, function_grad, learning_rate)
    print(title)
    print(f'Step count. Linear: {linear_step_count}. Gradient: {grad_step_count}')
    print(f'Arg min. Linear: {linear_arg_min}. Gradient: {grad_arg_min}')
    print(f'Min target. Linear: {linear_min_target}. Gradient: {grad_min_target}')
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


if __name__ == '__main__':
    # First function
    fst_function = lambda x: x ** 2 + 10 * x
    fst_function_grad = lambda x: 2 * x + 10

    methods_comparison(
        fst_function,
        fst_function_grad,
        'First function',
        left_arg=-40,
        right_arg=10,
        step_size=1,
        learning_rate=0.003
    )

    # Second function
    snd_function = lambda x: x ** 2 + 20 * x + 12
    snd_function_grad = lambda x: 2 * x + 20

    methods_comparison(
        snd_function,
        snd_function_grad,
        'Second function',
        left_arg=-40,
        right_arg=10,
        step_size=1,
        learning_rate=0.003
    )

    # Last function
    last_function = lambda x: x ** 2
    last_function_grad = lambda x: 2 * x

    methods_comparison(
        snd_function,
        snd_function_grad,
        'Last function',
        left_arg=-40,
        right_arg=10,
        step_size=1,
        learning_rate=0.003
    )
