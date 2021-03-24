from typing import Callable, Tuple, List
import numpy as np
import matplotlib.pyplot as plt


def dichotomy(
        function: Callable[[float], float],
        left_arg: float,
        right_arg: float,
        eps: float = 1e-8,
        delta: float = 1e-9,
        debug: bool = False
) -> Tuple[int, int, float, float, List[float]]:
    step_count = 0
    function_calls = 0
    distance = right_arg - left_arg
    distances_info = [distance]

    def function_with_counter(x):
        nonlocal function_calls
        function_calls += 1
        return function(x)

    while distance > eps:
        step_count += 1

        x1 = (right_arg + left_arg) / 2 - delta
        x2 = (right_arg + left_arg) / 2 + delta

        f1 = function_with_counter(x1)
        f2 = function_with_counter(x2)

        if debug:
            print(f'Left {left_arg}. Right {right_arg}')
            print(f'X1 {x1, f1}. X2 {x2, f2}')
            print()

        if f1 < f2:
            right_arg = x2
        elif f1 > f2:
            left_arg = x1
        else:
            left_arg = x1
            right_arg = x2

        distance = right_arg - left_arg
        distances_info.append(distance)

    left_target = function_with_counter(left_arg)
    right_target = function_with_counter(right_arg)

    if left_target < right_target:
        min_arg = left_arg
        min_target = left_target
    else:
        min_arg = right_arg
        min_target = right_target

    return step_count, function_calls, min_arg, min_target, distances_info


def golden_slice(
        function: Callable[[float], float],
        left_arg: float,
        right_arg: float,
        eps: float = 1e-8,
        debug: bool = False
) -> Tuple[int, int, float, float, List[float]]:
    step_count = 0
    function_calls = 0
    distance = right_arg - left_arg
    distances_info = [distance]

    def function_with_counter(x):
        nonlocal function_calls
        function_calls += 1
        return function(x)

    phi = (np.sqrt(5) - 1) / 2
    delta = distance * phi

    x1 = right_arg - delta
    x2 = left_arg + delta

    f1 = function_with_counter(x1)
    f2 = function_with_counter(x2)

    while distance > eps:
        step_count += 1

        if f1 >= f2:
            left_arg = x1
            distance = right_arg - left_arg
            delta = distance * phi
            x1 = x2
            f1 = f2
            x2 = left_arg + delta
            f2 = function_with_counter(x2)
        else:
            right_arg = x2
            distance = right_arg - left_arg
            delta = distance * phi
            x2 = x1
            f2 = f1
            x1 = right_arg - delta
            f1 = function(x1)
            function_calls += 1

        if debug:
            print(f'Left {left_arg}. Right {right_arg}')
            print(f'X1 {x1, f1}. X2 {x2, f2}')
            print()

        distance = right_arg - left_arg
        distances_info.append(distance)

    left_target = function_with_counter(left_arg)
    right_target = function_with_counter(right_arg)

    if left_target < right_target:
        min_arg = left_arg
        min_target = left_target
    else:
        min_arg = right_arg
        min_target = right_target

    return step_count, function_calls, min_arg, min_target, distances_info


def fibonacci(
        function: Callable[[float], float],
        left_arg: float,
        right_arg: float,
        eps: float = 1e-8,
        n: int = None,
        debug: bool = False
) -> Tuple[int, int, float, float, List[float]]:
    fib_storage = {0: 0, 1: 1}

    def fibonacci_helper(n):
        if n not in fib_storage:
            fib_storage[n] = fibonacci_helper(n - 1) + fibonacci_helper(n - 2)
        return fib_storage[n]

    step_count = 0
    function_calls = 0
    distance = right_arg - left_arg
    distances_info = [distance]

    def function_with_counter(x):
        nonlocal function_calls
        function_calls += 1
        return function(x)

    if n is None:
        n = 2
        while fibonacci_helper(n) < (right_arg - left_arg) / eps:
            n += 1

    x1 = left_arg + fibonacci_helper(n - 2) / fibonacci_helper(n) * distance
    x2 = left_arg + fibonacci_helper(n - 1) / fibonacci_helper(n) * distance
    f1 = function_with_counter(x1)
    f2 = function_with_counter(x2)
    k = 1

    while distance > eps:
        step_count += 1
        if k == n - 1:
            break

        if f1 > f2:
            left_arg = x1
            x1 = x2
            f1 = f2
            x2 = left_arg + fibonacci_helper(n - k - 1) / fibonacci_helper(n - k) * (right_arg - left_arg)
            f2 = function_with_counter(x2)
        else:
            right_arg = x2
            x2 = x1
            f2 = f1
            x1 = left_arg + fibonacci_helper(n - k - 2) / fibonacci_helper(n - k) * (right_arg - left_arg)
            f1 = function_with_counter(x1)

        k += 1

        if debug:
            print(f'Left {left_arg}. Right {right_arg}')
            print(f'X1 {x1, f1}. X2 {x2, f2}')
            print()

        distance = right_arg - left_arg
        distances_info.append(distance)

    distance = right_arg - left_arg
    distances_info.append(distance)

    left_target = function_with_counter(left_arg)
    right_target = function_with_counter(right_arg)

    if left_target < right_target:
        min_arg = left_arg
        min_target = left_target
    else:
        min_arg = right_arg
        min_target = right_target

    return step_count, function_calls, min_arg, min_target, distances_info


if __name__ == '__main__':
    some_function = lambda x: 3 * x ** 2 + 17 * x + 4

    dih_step_count, dih_function_call_count, dih_min_arg, dih_min_target, dih_distances_info = \
        dichotomy(some_function, -5, 10)

    gold_step_count, gold_function_call_count, gold_min_arg, gold_min_target, gold_distances_info = \
        golden_slice(some_function, -5, 10)

    fib_step_count, fib_function_call_count, fib_min_arg, fib_min_target, fib_distances_info = \
        fibonacci(some_function, -5, 10, n=3)

    print(f'Steps. '
          f'Dih: {dih_step_count}. '
          f'Golden: {gold_step_count}. '
          f'Fibonacci: {fib_step_count}')

    print(f'Function calls. '
          f'Dih: {dih_function_call_count}. '
          f'Golden: {gold_function_call_count}. '
          f'Fibonacci: {fib_function_call_count}')

    print(f'Args. '
          f'Dih: {dih_min_arg}. '
          f'Golden: {gold_min_arg}. '
          f'Fibonacci: {fib_min_arg}')

    print(f'Targets. '
          f'Dih: {dih_min_target}. '
          f'Golden: {gold_min_target}. '
          f'Fibonacci: {fib_min_target}')

    plt.plot(dih_distances_info, label='Dichotomy')
    plt.plot(gold_distances_info, label='Golden')
    plt.plot(fib_distances_info, label='Fibonacci')
    plt.title('Distance reduction comparison')
    plt.xlabel('Step count')
    plt.ylabel('right_arg - left_arg')
    plt.grid()
    plt.legend()
    plt.show()
