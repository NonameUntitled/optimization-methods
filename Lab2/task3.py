import numpy as np

from task1 import advanced_gradient_descent, newton_method
from Lab1.task3 import *


# def calc_level_arr(function):
#     x_arr = [ii / 100 for ii in range(-1000, 1000)]
#     y_arr = [ii / 100 for ii in range(-1000, 1000)]
#     f_arr = [[0 for _ in range(len(x_arr))] for _ in range(len(x_arr))]
#     for x_i in range(len(x_arr)):
#         for y_i in range(len(y_arr)):
#             x_v = x_arr[x_i]
#             y_v = y_arr[y_i]
#             f_arr[x_i][y_i] = function(np.array([x_v, y_v], dtype=np.float64))
#     return x_arr, y_arr, f_arr
#
#
# def find_level_point(z, x_arr, y_arr, f_arr, eps = 0.01):
#     x_p = []
#     y_p = []
#     for x_i in range(len(x_arr)):
#         for y_i in range(len(y_arr)):
#             if abs(f_arr[x_i][y_i] - z) < eps:
#                 x_p.append(x_arr[x_i])
#                 y_p.append(y_arr[y_i])
#     return x_p, y_p
#
#
# def find_all_level_points(now_x, now_y, x_arr, y_arr, f_arr, function):
#     x_res = []
#     y_res = []
#     for xx in now_x:
#         for yy in now_y:
#             z = function([xx, yy])
#             x_p, y_p = find_level_point(z, x_arr, y_arr, f_arr)
#             x_res += x_p
#             y_res += y_p
#     return x_res, y_res

if __name__ == '__main__':
    # First function

    # fst_function = lambda x: 100 * (x[1] - x[0]) ** 2 + (1 - x[0]) ** 2
    # fst_function_grad = lambda x: np.array([-200 * x[1] + 202 * x[0] - 2, 200 * x[1] - 200 * x[0]], dtype=np.float64)
    # fst_function_hessian = lambda x: np.array([[202, -200], [-200, 200]], dtype=np.float64)

    # fst_function = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    # fst_function_grad = lambda x: np.array(
    #     [-400 * x[0] * x[1] + 400 * x[0] ** 3 - 2 + 2 * x[0], 200 * x[1] - 200 * x[0] ** 2],
    #     dtype=np.float64
    # )
    # fst_function_hessian = lambda x: np.array(
    #     [
    #         [-400 * x[1] + 2 + 1200 * x[0] ** 2, -400 * x[0]],
    #         [-400 * x[0], 200]
    #     ], dtype=np.float64
    # )

    fst_function = lambda x: 800 * x[0] ** 2 + 12 * x[0] * x[1] + 500 * x[1] ** 2 + 2 * x[0] - 10 * x[1] + 3
    fst_function_grad = lambda x: np.array([12 * x[1] + 1600 * x[0] + 2, 1000 * x[1] + 12 * x[0] - 10], dtype=np.float64)
    fst_function_hessian = lambda x: np.array([[1600, 12], [12, 1000]], dtype=np.float64)


    advanced_gradient_descent_result = advanced_gradient_descent(
        fst_function,
        fst_function_grad,
        fst_function_hessian,
        np.array([5, 3], dtype=np.float64),
        1e-5
    )

    gradient_result = gradient_descent(
        fst_function,
        fst_function_grad,
        np.array([5, 3], dtype=np.float64),
        0.001,
        1e-5
    )

    newton_method_result = newton_method(
        fst_function,
        fst_function_grad,
        fst_function_hessian,
        np.array([5, 3], dtype=np.float64),
        1e-5
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

    gradient_step_count = gradient_result[0]
    gradient_min_arg = gradient_result[1]
    gradient_min_target = gradient_result[2]
    gradient_args_history = gradient_result[3]
    gradient_function_history = gradient_result[4]
    gradient_converge_time = gradient_result[5]

    gradient_arg_history_x = list(map(lambda pnt: pnt[0], gradient_args_history))
    gradient_arg_history_y = list(map(lambda pnt: pnt[1], gradient_args_history))



    newton_step_count = newton_method_result[0]
    newton_function_calls_count = newton_method_result[1]
    newton_function_grad_calls_count = newton_method_result[2]
    newton_function_hessian_calls_count = newton_method_result[3]
    newton_min_arg = newton_method_result[4]
    newton_min_target = newton_method_result[5]
    newton_algorithm_converge_time = newton_method_result[6]
    newton_args_history = newton_method_result[7]

    newton_arg_history_x = list(map(lambda pnt: pnt[0], newton_args_history))
    newton_arg_history_y = list(map(lambda pnt: pnt[1], newton_args_history))

    print(f"steps:\nadv {advanced_step_count}\ngrad {gradient_step_count}\nnewton {newton_step_count}")
    print(f"time:\nadv {advanced_algorithm_converge_time}\ngrad {gradient_converge_time}\nnewton {newton_algorithm_converge_time}")
    print(f"call_func:\nadv {advanced_function_calls_count}\ngrad {gradient_step_count}\nnewton {newton_function_calls_count}")


    plt.title('Third function')
    plt.scatter(advanced_algorithm_args_history_x, advanced_algorithm_args_history_y, s=1, color='green')
    plt.plot(advanced_algorithm_args_history_x, advanced_algorithm_args_history_y, color='green')

    # plt.scatter(gradient_arg_history_x, gradient_arg_history_y, s=1, color='orange')
    # plt.plot(gradient_arg_history_x, gradient_arg_history_y, color='orange')
    #
    # plt.scatter(newton_arg_history_x, newton_arg_history_y, s=1, color='m')
    # plt.plot(newton_arg_history_x, newton_arg_history_y, color='m')

    all_level_x = advanced_algorithm_args_history_x + gradient_arg_history_x + newton_arg_history_x
    all_level_y = advanced_algorithm_args_history_y + gradient_arg_history_y + newton_arg_history_y


    q_matrix = np.array([[800, 6], [6, 500]], dtype=np.float64)
    b_vector = np.array([2, -10], dtype=np.float64)
    c_value = 3

    x_arr, y_arr = calc_all_levels(all_level_x, all_level_y, -3, 7, q_matrix, b_vector, c_value)
    plt.scatter(x_arr, y_arr, s=0.001, color='blue')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    print(advanced_algorithm_args_history_x, advanced_algorithm_args_history_y)
    print(newton_arg_history_x, newton_arg_history_y)


