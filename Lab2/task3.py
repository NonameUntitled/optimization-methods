import numpy as np

from task1 import advanced_gradient_descent, newton_method
from Lab1.task3 import *

if __name__ == '__main__':
    # First function
    fst_function = lambda x: 13 * x[0] ** 2 - 12 * x + 45
    fst_function_grad = lambda x: np.array([26 * x[0] - 12], dtype=np.float64)
    fst_function_hessian = lambda x: np.array([[26]], dtype=np.float64)

    advanced_gradient_descent_result = advanced_gradient_descent(
        fst_function,
        fst_function_grad,
        fst_function_hessian,
        np.array([123], dtype=np.float64)
    )

    gradient_result = gradient_descent(
        fst_function,
        fst_function_grad,
        np.array([100], dtype=np.float64),
        1e-2
    )

    advanced_step_count = advanced_gradient_descent_result[0]
    advanced_function_calls_count = advanced_gradient_descent_result[1]
    advanced_function_grad_calls_count = advanced_gradient_descent_result[2]
    advanced_min_arg = advanced_gradient_descent_result[3]
    advanced_min_target = advanced_gradient_descent_result[4]
    advanced_algorithm_converge_time = advanced_gradient_descent_result[5]
    advanced_algorithm_args_history = advanced_gradient_descent_result[6]

    # Не работает так как нужна двумерная функция
    advanced_algorithm_args_history_x = list(map(lambda pnt: pnt[0], advanced_algorithm_args_history))
    advanced_algorithm_args_history_y = list(map(lambda pnt: pnt[1], advanced_algorithm_args_history))

    newton_step_count = gradient_result[0]
    newton_min_arg = gradient_result[1]
    newton_min_target = gradient_result[2]
    newton_args_history = gradient_result[3]
    newton_function_history = gradient_result[4]

    # Не работает так как нужна двумерная функция
    newton_function_history_x = list(map(lambda pnt: pnt[0], newton_function_history))
    newton_function_history_y = list(map(lambda pnt: pnt[1], newton_function_history))

    plt.title('First function')
    plt.scatter(advanced_algorithm_args_history_x, advanced_algorithm_args_history_y)
    plt.plot(advanced_algorithm_args_history_x, advanced_algorithm_args_history_y)
    plt.scatter(newton_function_history_x, advanced_algorithm_args_history_y)
    plt.plot(newton_function_history_x, advanced_algorithm_args_history_y)
    plt.legend()
    plt.show()

    """
    Предлагаю сделать сравнение по всем параметрам в отчете а в коде показать только самые очевидные 
    кол-во итераций, кол-во подсчётов градиента и гессиана, астрономическое время работы.
    Здесь эе можно и построить графики. See example above
    """

