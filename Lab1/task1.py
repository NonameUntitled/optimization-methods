from typing import Callable, Tuple, List
import numpy as np
import matplotlib.pyplot as plt


def dihotomy_method_easy(
        function: Callable[[float], float],
        left_arg: float,
        right_arg: float,
        eps: float = 1e-8,
        delta: float = 1e-9,
        debug: bool = False
) -> Tuple[int, int, float, float, List[float]]:
    distances_info = []
    step_count = 0
    function_calls = 0
    distance = right_arg - left_arg
    distances_info.append(distance)

    while distance > eps:
        x1 = (right_arg + left_arg) / 2 - delta
        x2 = (right_arg + left_arg) / 2 + delta

        f1 = function(x1)
        f2 = function(x2)
        function_calls += 2

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

        step_count += 1
        distance = right_arg - left_arg
        distances_info.append(distance)

    left_target = function(left_arg)
    right_target = function(right_arg)
    function_calls += 2

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
    distances_info = []
    step_count = 0
    function_calls = 0
    distance = right_arg - left_arg
    distances_info.append(distance)

    phi = (np.sqrt(5) - 1) / 2
    delta = distance * phi

    x_a = right_arg - delta
    x_b = left_arg + delta

    f_a = function(x_a)
    f_b = function(x_b)
    function_calls += 2

    while distance > eps:
        if debug:
            print(left_arg, x_a, f_a, x_b, f_b, right_arg)

        if f_a >= f_b:
            left_arg = x_a

            distance = right_arg - left_arg
            delta = distance * phi

            x_a = x_b
            f_a = f_b
            x_b = left_arg + delta
            f_b = function(x_b)
            function_calls += 1
        else:
            right_arg = x_b

            distance = right_arg - left_arg
            delta = distance * phi

            x_b = x_a
            f_b = f_a
            x_a = right_arg - delta
            f_a = function(x_a)
            function_calls += 1

        step_count += 1
        distance = right_arg - left_arg
        distances_info.append(distance)

    if f_a < f_b:
        return step_count, function_calls, x_a, f_a, distances_info
    else:
        return step_count, function_calls, x_b, f_b, distances_info


def fibonacci_method(
        function: Callable[[float], float],
        left_arg: float,
        right_arg: float,
        eps: float = 1e-8,
        n: int = None,
        debug: bool = False
) -> Tuple[int, int, float, float, List[float]]:
    fib_storage = {}

    def fibonacci(n):
        if n not in fib_storage:
            if n == 0:
                fib_storage[n] = 0
            elif n == 1:
                fib_storage[n] = 1
            else:
                fib_storage[n] = fibonacci(n - 1) + fibonacci(n - 2)
        return fib_storage[n]

    distances_info = []
    step_count = 0
    function_calls = 0
    distance = right_arg - left_arg
    distances_info.append(distance)

    if n is None:
        n = 2
        while fibonacci(n) < (right_arg - left_arg) / eps:
            n += 1

    x1 = left_arg + fibonacci(n - 2) / fibonacci(n) * distance
    x2 = left_arg + fibonacci(n - 1) / fibonacci(n) * distance

    f1 = function(x1)
    f2 = function(x2)
    function_calls += 2
    k = 1

    while distance > eps:
        if debug:
            print(f'Left {left_arg}. Right {right_arg}')
            print(f'X1 {x1, f1}. X2 {x2, f2}')
            print()
        step_count += 1

        if f1 > f2:
            left_arg = x1

            x1 = x2
            f1 = f2
            x2 = left_arg + fibonacci(n - k - 1) / fibonacci(n - k) * (right_arg - left_arg)
            if k != n - 2:
                f2 = function(x2)
                function_calls += 1
                k += 1
            else:
                break
        else:
            right_arg = x2

            x2 = x1
            f2 = f1
            x1 = left_arg + fibonacci(n - k - 2) / fibonacci(n - k) * (right_arg - left_arg)

            if k != n - 2:
                f1 = function(x1)
                function_calls += 1
                k += 1
            else:
                break

        distance = right_arg - left_arg
        distances_info.append(distance)

    distance = right_arg - left_arg
    distances_info.append(distance)

    f1 = function(left_arg)
    f2 = function(right_arg)
    function_calls += 2

    if f1 < f2:
        return step_count, function_calls, left_arg, f1, distances_info
    else:
        return step_count, function_calls, right_arg, f2, distances_info


if __name__ == '__main__':
    dih_step_count, dih_function_call_count, dih_min_arg, dih_min_target, dih_distances_info = \
        dihotomy_method_easy(lambda x: 3 * x ** 2 + 17 * x + 4, -500, 1000)

    gold_step_count, gold_function_call_count, gold_min_arg, gold_min_target, gold_distances_info = \
        golden_slice(lambda x: 3 * x ** 2 + 17 * x + 4, -500, 1000)

    fib_step_count, fib_function_call_count, fib_min_arg, fib_min_target, fib_distances_info = \
        fibonacci_method(lambda x: 3 * x ** 2 + 17 * x + 4, -500, 1000, debug=True, n=10)

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

    plt.plot(dih_distances_info, label='Dihotomy')
    plt.plot(gold_distances_info, label='Golden')
    plt.plot(fib_distances_info, label='Fibonacci')
    plt.title('Distance reduction comparison')
    plt.xlabel('Step count')
    plt.ylabel('right_arg - left_arg')
    plt.grid()
    plt.legend()
    plt.show()
