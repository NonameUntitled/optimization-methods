1) Общая - ограничения любые (`>=`, `=`, `<=`), стандартная - только `<=` или только `>=`, каноническая - только `=`

2) Методы естественного базиса - ??? (видимо, это когда подходит точка (0, 0, ..., 0) и мы берем ее в качестве исходного допустимого базисного решения). Метод искусственного базиса - вводим искусственные переменные (y_1, ..., y_n), решаем вспомогательную задачу оптимизации функции -y_1 - ... -y_n при ограничении Ax + y = b. Полученное решение является допустимым решением оригинальной задачи

3) -

4) -

5) -

6) 1. `x1`: (1, 0, 0, 0), `x3`: (0, 0, 1, 0), `x1, x3`: (i, 0, j, 0), forall i, j: i + j = 1, `x1, x2, x3, x4`: (i, a, j, b): forall i, j: i + j = 1; forall a, b: a + b = 0

7) x1 = (11 - 9x3) / 7; x2 = (10 - 5x3) / 7

8) Используем это решение как исходное базисное допустимое решение, построим симплексную таблицу и посмотрим, есть ли положительная симплекс разность. Ответ: нет, следовательно, решение оптимально. Также можно было просто решить задачу и убедиться, что ответ совпадает с `f(x)`.