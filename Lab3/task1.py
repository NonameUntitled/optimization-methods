from dataclasses import dataclass

eps = 1e-10
feasible_origin_eps = 1e15
debug = False


# a - левая часть уравнения/неравенства
# b - правая часть уравнения/неравенства
# sign - знак (-1, если <=; 0, если ==; 1, если >=)
@dataclass
class Constraint:
    a: [float]
    b: float
    sign: int

    def negate(self):
        self.a = [-x for x in self.a]
        self.b = -self.b
        self.sign = -self.sign

    def print(self):
        printed_first = False
        for i in range(0, len(self.a)):
            a_i = self.a[i]
            if a_i != 0:
                if printed_first:
                    if a_i > 0:
                        print(" + ", end='')
                    else:
                        print(" - ", end='')
                abs_a_i = ""
                if printed_first:
                    abs_a_i = str(abs(a_i))
                else:
                    abs_a_i = str(a_i)
                print(f"{abs_a_i}x{i + 1}", end='')
                printed_first = True
        if self.sign == 0:
            print(" = ", end='')
        elif self.sign == -1:
            print(" <= ", end='')
        else:
            print(" >= ", end='')
        print(self.b)


@dataclass
class Result:
    point: [float]
    value: float


def zeros(i: int) -> [float]:
    return [0.0] * i


def sign(i: float) -> float:
    if i >= 0.0:
        return 1
    return -1


# Симплекс метод. На вход принимаем целевую функцию, ограничения и исходное базисное допустимое решение
# Строим симплексную таблицу и итерируемся по алгоритму
def simplex_method(Z: [float], constraints: [Constraint], feasible_origin: [float]) -> Result:
    m = len(Z) + 1
    n = len(constraints)
    # Строим симплексную таблицу
    # Шаг 1. Вычленяем базис (индексы переменных) из исходного базисного допустимого решения
    basis = []
    for i in range(len(feasible_origin)):
        if feasible_origin[i] != 0:
            basis.append(i)
        if feasible_origin[i] == feasible_origin_eps:
            feasible_origin[i] = 0
    constraints.sort(reverse=True, key=lambda c: [c.a[i] for i in basis])
    i = 0
    while True:
        if i >= n:
            break

        zeros_cnt = 0
        for a_i in constraints[i].a:
            if a_i < eps:
                zeros_cnt += 1
        if zeros_cnt == len(constraints[i].a):
            constraints.remove(constraints[i])
            n -= 1
            i -= 1
            continue

        basis_value = constraints[i].a[basis[i]]
        for j in range(len(constraints[i].a)):
            constraints[i].a[j] /= basis_value
        constraints[i].b /= basis_value
        for j in range(n):
            if i == j:
                continue
            d = constraints[j].a[basis[i]]
            for k in range(len(constraints[j].a)):
                constraints[j].a[k] -= d * constraints[i].a[k]
            constraints[j].b -= d * constraints[i].b
            if constraints[j].b < 0:
                constraints[j].negate()
        constraints.sort(reverse=True, key=lambda c: [c.a[i] for i in basis])
        i += 1

    for constraint in constraints:
        for r in [constraint.a[i] for i in basis]:
            if r < 0:
                raise ValueError("Недопустимая база")

    # Шаг 2. Заполняем симплексную таблицу. Первый столбец: b. Последняя строка: -f
    simplex_table = []
    for constraint in constraints:
        simplex_table.append([constraint.b] + constraint.a)
    f = [0] * m
    for i in range(len(Z)):
        f[0] -= Z[i] * feasible_origin[i]
    for i in range(len(Z)):
        if i not in basis:
            f[i + 1] += Z[i]
        else:
            for constraint in constraints:
                if constraint.a[i] == 0:
                    continue
                a = constraint.a
                for j in range(len(a)):
                    if j == i:
                        continue
                    f[j + 1] -= a[j] * Z[i] / a[i]
    simplex_table.append(f)

    # Пока существует положительная симплексная разность, пытаемся найти базис лучше
    while True:
        max_simplex_diff = 0
        lead_element_j = -1
        for j in range(1, m):
            if simplex_table[n][j] > max_simplex_diff:
                max_simplex_diff = simplex_table[n][j]
                lead_element_j = j
        if max_simplex_diff < eps:
            break

        min_fraction = 1e10
        if simplex_table[0][lead_element_j] != 0:
            if sign(simplex_table[0][0]) == sign(simplex_table[0][lead_element_j]):
                min_fraction = simplex_table[0][0] / \
                    simplex_table[0][lead_element_j]
        lead_element_i = 0
        for i in range(n):
            fraction = 1e10
            if simplex_table[i][lead_element_j] != 0:
                fraction = simplex_table[i][0] / \
                    simplex_table[i][lead_element_j]
            if sign(simplex_table[i][0]) == sign(simplex_table[i][lead_element_j]) and fraction < min_fraction:
                min_fraction = fraction
                lead_element_i = i

        lead_element = simplex_table[lead_element_i][lead_element_j]
        for j in range(m):
            simplex_table[lead_element_i][j] /= lead_element

        for i in range(n + 1):
            if i == lead_element_i:
                continue
            d = -simplex_table[i][lead_element_j]
            for j in range(m):
                simplex_table[i][j] += d * simplex_table[lead_element_i][j]

        point = [0] * len(feasible_origin)
        for i in range(n):
            point[basis[i]] = simplex_table[i][0]

        basis[lead_element_i] = lead_element_j - 1

    # Вектор (учитывая искусственные переменные), на котором функция оптимальна, ищем из симплексной таблицы на основе базиса
    point = [0] * len(feasible_origin)
    for i in range(n):
        point[basis[i]] = simplex_table[i][0]

    # Значение функции лежит в первом столбце последней строки
    value = -simplex_table[n][0]

    return Result(point, value)


# Любая ЗЛП сводится к каноничной форме
# Для этого, видя ограничения вида "неравенство", прибавим к левой части новую неотрицательную переменную
def to_canonic(Z: [float], constraints: [Constraint]):
    free_cnt = 0
    # Подсчитаем количество свободных переменных
    # Избавимся от неравенств вида ">=" (ну а вдруг)
    for constraint in constraints:
        if constraint.sign != 0:
            if constraint.sign == 1:
                constraint.negate()
            free_cnt += 1

    Z.extend(zeros(free_cnt))

    current_free_cnt = 0
    # Избавимся от ограничений вида "неравенство", приведя их к ограничению вида "равенство" вышеописанным способом
    for constraint in constraints:
        # Если это уже равенство, ничего делать не нужно, но заполним свободные переменные нулями
        # Если это неравенство, добавим новую переменную с коэффициентом 1
        if constraint.sign == 0:
            constraint.a.extend(zeros(free_cnt))
        else:
            constraint.a.extend(zeros(current_free_cnt))
            constraint.a.append(1)
            constraint.a.extend(zeros(free_cnt - current_free_cnt - 1))
            current_free_cnt += 1
            constraint.sign = 0
        if constraint.b < 0:
            constraint.negate()


# Проверка на каноничность ЗЛП осуществляется по определению
def is_canonic(constraints: [Constraint]) -> bool:
    for row in constraints:
        if row.sign != 0 or row.b < 0:
            return False

    return True


# Мы считаем, что ЗЛП передается в каноничной или стандартной форме
# Если хотим максимизировать функцию, выставляем search_max в True
def optimize(Z: [float], constraints: [Constraint], search_max: bool = True, feasible_origin: [float] = None) -> Result:
    if not is_canonic(constraints):
        to_canonic(Z, constraints)

    if not search_max:
        Z = [-x for x in Z]

    # Нахождение исходного базисного допустимого решения методом искусственных переменных
    # Для этого решаем вспомогательную задачу симплексным методом, её оптимальное решение будет допустимым решением основной задачи
    if not feasible_origin:
        cnt_x = len(constraints[0].a)
        cnt_y = len(constraints)
        Z_y = zeros(cnt_x) + [-1] * cnt_y
        constraints_y = []
        feasible_origin_y = zeros(cnt_x)
        current_free_cnt = 0
        for i in range(len(constraints)):
            constraint = constraints[i]
            a_y = []
            a_y.extend(constraint.a)
            a_y.extend(zeros(current_free_cnt))
            a_y.append(1)
            a_y.extend(zeros(cnt_y - current_free_cnt - 1))
            constraints_y.append(Constraint(a_y, constraint.b, 0))
            b = constraint.b
            if b == 0:
                b += feasible_origin_eps
            feasible_origin_y.append(b)
            current_free_cnt += 1
        solution = optimize(Z_y, constraints_y, True, feasible_origin_y).point
        feasible_origin = solution[:cnt_x]

    return simplex_method(Z, constraints, feasible_origin)


def test(example):
    # Условие задачи
    # Z - функция
    # constraints - ограничения
    Z = example["Z"]
    constraints = example["constraints"]

    # Опционально нам могут дать исходное базисное допустимое решение
    # Если его не дали в задаче - не страшно, найдем его с помощью метода искусственных переменных
    feasible_origin = example["feasible_origin"]

    try:
        solution = optimize(Z, constraints, True, feasible_origin)
        if abs(solution.value - example["expected"]) > eps:
            print("Тест не пройден. Ожидалось:",
                  example["expected"], "; результат:", solution.value)
        else:
            print("Тест пройден")
    except ValueError as e:
        if not example["error"]:
            print("Тест не пройден. Неожиданная ошибка")
        else:
            if example["error"] == str(e):
                print("Тест пройден")
            else:
                print("Тест не пройден. Ожидалось:",
                      example["error"], "; результат:", str(e))


tests = [
    {
        "Z": [-1, -1, 0, -5],
        "constraints": [
            Constraint([1, 1, -1, 3], 1, 0),
            Constraint([1, -2, 3, -1], 1, 0),
            Constraint([5, -4, 7, 3], 5, 0)
        ],
        "feasible_origin": None,
        "expected": -1.0
    },
    {
        "Z": [6, 1, 4, -5],
        "constraints": [
            Constraint([3, 1, -1, 1], 4, 0),
            Constraint([5, 1, 1, -1], 4, 0)
        ],
        "feasible_origin": None,
        "expected": 4.0
    },
    {
        "Z": [6, 1, 4, -5],
        "constraints": [
            Constraint([3, 1, -1, 1], 4, 0),
            Constraint([5, 1, 1, -1], 4, 0)
        ],
        "feasible_origin": [1, 0, 0, 1],
        "expected": 4.0
    },
    {
        "Z": [1, 2, 3, -1],
        "constraints": [
            Constraint([1, -3, -1, -2], -4, 0),
            Constraint([1, -1, 1, 0], 0, 0)
        ],
        "feasible_origin": None,
        "expected": 6.0
    },
    {
        "Z": [1, 2, 3, -1],
        "constraints": [
            Constraint([1, -3, -1, -2], -4, 0),
            Constraint([1, -1, 1, 0], 0, 0)
        ],
        "feasible_origin": [0, 1, 1, 0],
        "expected": 6.0
    },
    {
        "Z": [1, 2, 1, -3, 1],
        "constraints": [
            Constraint([1, 1, 0, 2, 1], 5, 0),
            Constraint([1, 1, 1, 3, 2], 9, 0),
            Constraint([0, 1, 1, 2, 1], 6, 0)
        ],
        "feasible_origin": None,
        "expected": 11.0
    },
    {
        "Z": [1, 2, 1, -3, 1],
        "constraints": [
            Constraint([1, 1, 0, 2, 1], 5, 0),
            Constraint([1, 1, 1, 3, 2], 9, 0),
            Constraint([0, 1, 1, 2, 1], 6, 0)
        ],
        "feasible_origin": [0, 0, 1, 2, 1],
        "expected": 11.0
    },
    {
        "Z": [1, 1, 1, -1, 1],
        "constraints": [
            Constraint([1, 1, 2, 0, 0], 4, 0),
            Constraint([0, -2, -2, 1, -1], -6, 0),
            Constraint([1, -1, 6, 1, 1], 12, 0)
        ],
        "feasible_origin": None,
        "expected": 10.0
    },
    {
        "Z": [1, 1, 1, -1, 1],
        "constraints": [
            Constraint([1, 1, 2, 0, 0], 4, 0),
            Constraint([0, -2, -2, 1, -1], -6, 0),
            Constraint([1, -1, 6, 1, 1], 12, 0)
        ],
        "feasible_origin": [1, 1, 2, 0, 0],
        "error": "Недопустимая база"
    },
    {
        "Z": [1, -4, 3, -10],
        "constraints": [
            Constraint([1, 1, -1, -10], 0, 0),
            Constraint([1, 14, 10, -10], 11, 0)
        ],
        "feasible_origin": None,
        "expected": 4.0
    },
    {
        "Z": [1, -5, -1, 1],
        "constraints": [
            Constraint([1, 3, 3, 1], 3, -1),
            Constraint([2, 0, 3, -1], 4, -1)
        ],
        "feasible_origin": None,
        "expected": 3.0
    },
    {
        "Z": [1, 1, -1, 1, -2],
        "constraints": [
            Constraint([3, 1, 1, 1, -2], 10, 0),
            Constraint([6, 1, 2, 3, -4], 20, 0),
            Constraint([10, 1, 3, 6, -7], 30, 0)
        ],
        "feasible_origin": None,
        "expected": -10.0
    }
]


def run_test(i: int):
    test(tests[i])


def run_all():
    for i in range(len(tests)):
        run_test(i)


run_all()
