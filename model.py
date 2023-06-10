# requirements
import datetime as dt
import pandas as pd
import pickle
import random
from copy import deepcopy
import numpy as np
from dateutil.relativedelta import relativedelta

with open('./datasets/deposit_predictor.sav', 'rb') as f:
    deposit_predictor = pickle.load(f)
with open('./datasets/loan_predictor.sav', 'rb') as f:
    loan_predictor = pickle.load(f)

from tensorflow import keras
Q_network = keras.models.load_model('datasets/Q_network.h5')


# Специальный метод для датасетов со ставками на депозиты и кредиты. Возвращает массив из четырех ставок на указанный месяц
def rates_on_month(df, string):
    return df[df.Месяц == string][df.columns[1:]].to_numpy().flatten()


# Преобразует дату в строку "Месяц Год" (например, "Сентябрь 2009"). Работает и в обратную сторону (ставится первое число месяца)
def datetime_str_transform(date=None, string=None, str_to_datetime=False):
    # преобразование строки в дату, если str_to_datetime = True. Иначе преобразование даты в строку
    if str_to_datetime:

        # находим индекс пробела в строке (пробел - разделитель между названием месяца и года)
        sep_index = string.find(' ')

        # преобразуем в дату datetime
        return dt.datetime(year=int(string[sep_index + 1:]), month=month_names_to_nums[string[:sep_index]], day=1)

    else:
        return month_names[date.month] + ' ' + str(date.year)


# Поиск значения по любой дате в пределах датасета
# Метод только для датасетов заданной структуры: столбец с датой или месяцем идет первым, со значениями - вторым, всего их два
def value_on_date(df, date: dt.datetime, monthly=False):
    # определяем имена столбцов с датами и значениями
    date_col, value_col = df.columns[0], df.columns[1]

    # параметр monthly определяет, что находится в первом столбце: дата или месяц (False или True соответственно)
    if monthly:

        value = df[df[date_col] == datetime_str_transform(date=date)][value_col].to_numpy()[0]

    else:

        # если искомой даты нет в датасете - считаем среднее между значениями на соседние присутствующие даты
        try:

            value = df[df[date_col] == date][value_col].to_numpy()[0]

        except IndexError:

            next_value = df[df[date_col] > date].tail(1)[value_col].to_numpy()[0]
            prev_value = df[df[date_col] < date].head(1)[value_col].to_numpy()[0]
            value = (next_value + prev_value) / 2

    return value


# Возвращает массив с нормализованными данными одного параметра (min-max)
def normalize_x(arr, min_max):
    return np.array([(x - min_max[0]) / (min_max[1] - min_max[0]) for x in arr], dtype='float32')


# возвращает исходное значение параметра по нормализованному
def denormalize_x(norm_x, min_max):
    return (min_max[1] - min_max[0]) * norm_x + min_max[0]


# количество дней в каждом месяце
days_in_month = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 31, 7: 30, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}


# Функция возвращает тип срока, к которому относится срок term
def term_category(term):
    categories = ((1, 3), (4, 6), (7, 12), (13, 36))

    for i in range(len(categories)):

        if categories[i][0] <= term <= categories[i][1]:
            break

    return i


# Даты первых платежей
def first_payment_dates(df):
    dates = df.Дата
    payment_days = df.День_платежа

    return [dt.datetime(year=dates[i].year,
                        month=dates[i].month,
                        day=payment_days[i]) + relativedelta(months=1) for i in range(len(df))]


# Даты последних платежей, они же даты закрытия (этот метод будет использоваться и на других этапах работы)
def close_dates(df):
    first_payment_dates_ = first_payment_dates(df)
    terms = df.Срок

    return np.array([first_payment_dates_[i] + relativedelta(months=terms[i] - 1) for i in range(len(df))])


class Env:

    # Конструктор класса. Задается датой, диапазоном суммы элемента портфеля, структурой портфелей и прогнозными моделями
    def __init__(self, date: dt.datetime, min_V, max_V, deposits, loans, deposit_predictor, loan_predictor):

        # текущая и предыдущая дата
        self.date = date
        self.prev_date = date - dt.timedelta(days=1)

        # ключи для получения минимальных и максимальных значений параметров среды и ставок (совпадают по индексам)
        self.deposit_mm_keys = [key for key in list(min_maxs.keys())[:4]]
        self.deposit_mm_keys.extend([key for key in list(min_maxs.keys())[8:12]])
        self.loan_mm_keys = [key for key in list(min_maxs.keys())[:4]]
        self.loan_mm_keys.extend([key for key in list(min_maxs.keys())[4:8]])

        # параметры среды из датасетов
        self.key_rate = value_on_date(key_rates, date)
        self.dollar = value_on_date(dollar, date)
        self.euro = value_on_date(euro, date)
        self.inflation = inflation[inflation.Год == date.year][month_names[date.month]].to_numpy()[0]
        self.deposit_rates = rates_on_month(deposit_rates, datetime_str_transform(date))
        self.loan_rates = rates_on_month(loan_rates, datetime_str_transform(date))

        # минимальное и максимальное значение суммы одного депозита или кредита
        self.min_V, self.max_V = min_V, max_V

        # текущая структура портфелей депозитов и кредитов
        self.deposits = deposits
        self.loans = loans

        # модели, отражающие зависимость дневных объемов привлечений и кредитов в зависимости от ставок и параметров среды
        self.deposit_predictor = deposit_predictor
        self.loan_predictor = loan_predictor

        # текущее значение Q-функции и награда
        self.Q = 0
        self.reward = 0

    # Возвращает вектор параметров среды (удобно использовать для быстрого формирования входных параметров моделей)
    def get_state(self):
        return [self.inflation, self.key_rate, self.dollar, self.euro]

    # Возвращает ожидаемые дневные объемы депозитов и кредитов по выбранной стратегии (новые ставки по депозитам)
    def predict_V(self, action):

        # формируем входные ветора для прогнозных моделей (для кредитов всегда применяются средневзвешенные ставки)
        deposit_params, loan_params = self.get_state(), self.get_state()
        deposit_params.extend(action)
        loan_params.extend(self.loan_rates)

        # нормализуем входные вектора
        deposit_params = [[normalize_x([deposit_params[i]],
                                       min_maxs[self.deposit_mm_keys[i]])[0] for i in range(len(deposit_params))]]

        loan_params = [
            [normalize_x([loan_params[i]], min_maxs[self.loan_mm_keys[i]])[0] for i in range(len(loan_params))]]

        # получаем прогнозы из моделей, денормализуем и делим их на кол-во дней в месяце, получая ожидаемые дневные объемы
        deposit_V = denormalize_x(self.deposit_predictor.predict(deposit_params)[0],
                                  min_maxs['Объем_депозитов']) / days_in_month[self.date.month]

        loan_V = denormalize_x(self.loan_predictor.predict(loan_params)[0],
                               min_maxs['Объем_кредитов']) / days_in_month[self.date.month]

        return deposit_V, loan_V

    # Генерирует данные для одного случайного депозита или кредита на указанную дату date
    def random_item(self, date, amount, rates, payment_day):

        # ставка берется по сроку в соответствии с текущими ставками rates
        term = np.random.randint(1, 37)
        rate = rates[term_category(term)]

        return date, amount, rate, term, payment_day

    # Генерирует обновление для портфеля (случайное число элементов общей суммой в required_V +- self.min_V)
    def update_portfolio(self, rates, required_V):

        # payment_day - число, на которое каждый месяц будет совершаться платеж по кредиту либо день даты закрытия депозита
        payment_day = min(self.date.day, 28)
        columns = self.deposits.columns

        items = {column: [] for column in columns}
        items_V = 0

        while items_V < required_V:

            # сумма нового элемента портфеля генерируется в пределах диапазона [self.min_V, self.max_V] и кратна 100 000
            amount = round(np.random.randint(self.min_V, self.max_V), -5)

            # если добавление amount к общей сумме превысит нужный объем портфеля
            if (items_V + amount) > required_V:

                # если разность между нужной суммой и общей больше минимальной суммы элемента, то amount равняется этой разности
                if required_V - items_V > self.min_V:
                    amount = round(required_V - items_V, -5)

                # иначе в соответствии со случайной вероятностью либо берется минимальная сумма, либо генерация заканчивается
                else:

                    if random.getrandbits(1):
                        amount = self.min_V
                    else:
                        break

            # генерируем данные для нового элемента портфеля
            new_item_data = self.random_item(self.date, amount, rates, payment_day)

            # передаем данные в словарь
            for i in range(len(columns)):
                items[columns[i]].append(new_item_data[i])

            # прибавляем значение суммы к значению общей суммы
            items_V += new_item_data[1]

        return pd.DataFrame(data=items)

    # Метод для обновления состояния среды. На входе новые ставки по депозитам (выбранная стратегия на данный момент)
    def update_state(self, action):

        # устанавливаем новые ставки на депозиты
        self.deposit_rates = action

        # переключаем текущую дату на день вперед, обновляем значение предыдущей даты
        self.prev_date = self.date
        self.date += dt.timedelta(days=1)

        # обновляем параметры среды
        self.key_rate = value_on_date(key_rates, self.date)
        self.dollar = value_on_date(dollar, self.date)
        self.euro = value_on_date(euro, self.date)

        # эти параметры обновляются раз в месяц (на первое число каждого месяца)
        if self.prev_date.month != self.date.month:
            self.inflation = inflation[inflation.Год == self.date.year][month_names[self.date.month]].to_numpy()[0]
            self.loan_rates = rates_on_month(loan_rates, datetime_str_transform(self.date))

        # прогнозируем объем привлечений и кредитов на грядущий день
        deposit_V, loan_V = self.predict_V(action)

        # обновляем портфели
        self.deposits = self.deposits._append(self.update_portfolio(action, deposit_V))
        self.loans = self.loans._append(self.update_portfolio(self.loan_rates, loan_V))
        self.deposits.reset_index(drop=True, inplace=True)
        self.loans.reset_index(drop=True, inplace=True)

        # отбрасываем депозиты и кредиты, даты закрытия которых раньше новой даты
        self.deposits = self.deposits[close_dates(self.deposits) >= self.date]
        self.loans = self.loans[close_dates(self.loans) >= self.date]
        self.deposits.reset_index(drop=True, inplace=True)
        self.loans.reset_index(drop=True, inplace=True)

        # обновляем значения награды и Q-функции
        prev_Q = self.Q
        self.Q = sum(self.loans.Сумма * (self.loans.Ставка / 100 / 12 + 1) ** self.loans.Срок) - \
                 sum(self.deposits.Сумма * (self.deposits.Ставка / 100 / 12 + 1) ** self.deposits.Срок)
        self.reward = self.Q - prev_Q

    # Вывод данных по текущим параметрам среды
    def summary(self):

        print(f'Текущая дата: {self.date:%d.%m.%Y}\n\n'
              f'Темп инфляции: {self.inflation}%\n'
              f'Ключевая ставка ЦБ: {self.key_rate}%\n'
              f'Курс доллара: {round(self.dollar, 2)} руб.\n'
              f'Курс евро: {round(self.euro, 2)} руб.\n\n'
              f'Средневзвешенные ставки по кредитам в РФ:\n'
              f'от 31 до 90 дней: {self.loan_rates[0]}\n'
              f'от 91 до 180 дней: {self.loan_rates[1]}\n'
              f'от 181 дня до 1 года: {self.loan_rates[2]}\n'
              f'от 1 года до 3 лет: {self.loan_rates[3]}'
              )


# берем данные начиная с 01.10.2013 (в сентябре была введена в действие ключевая ставка ЦБ)
start_date = dt.datetime(year=2013, month=10, day=1)

# обрезаем датасеты до 01.02.2023
end_date = dt.datetime(year=2023, month=2, day=1)

# название месяца по его номеру
month_names = {1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель', 5: 'Май', 6: 'Июнь',
               7: 'Июль', 8: 'Август', 9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'}

# номер месяца по названию
month_names_to_nums = {value: key for key, value in month_names.items()}

min_maxs = {'Темп_инфляции': (-0.54, 7.61),
            'Ключевая_ставка': (4.25, 20.0),
            'Курс_доллара': (31.6618, 120.3785),
            'Курс_евро': (43.5123, 132.9581),
            'КрСтавка1_3': (9.84, 30.35),
            'КрСтавка4_6': (11.0, 30.39),
            'КрСтавка7_12': (11.0, 33.55),
            'КрСтавка13_36': (11.1, 29.69),
            'ДепСтавка1_3': (3.76, 19.31),
            'ДепСтавка4_6': (3.99, 19.95),
            'ДепСтавка7_12': (4.14, 16.04),
            'ДепСтавка13_36': (4.12, 16.14),
            'Объем_депозитов': (2496421238.5099177, 45581985298.44098),
            'Объем_кредитов': (4372109257.0473, 61481732430.80357)}

inflation = pd.read_csv('datasets/inflation.csv')

# убираем пробелы из названий столбцов
old_names = inflation.columns
new_names = {old_names[i]: old_names[i].strip() for i in range(len(old_names))}
inflation.rename(columns=new_names, inplace=True)

# фильтруем по дате
inflation = inflation[(inflation.Год >= start_date.year) & (inflation.Год <= end_date.year)]

# ставим индексы с нуля
inflation.reset_index(drop=True, inplace=True)

key_rates = pd.read_excel('datasets/key_rates.xlsx', parse_dates=['Дата'])

# фильтруем по дате
key_rates = key_rates[(key_rates.Дата >= start_date) & (key_rates.Дата <= end_date)]

# сортируем по дате по возрастанию
key_rates = key_rates.sort_values(by='Дата')

# ставим индексы с нуля
key_rates.reset_index(drop=True, inplace=True)

dollar = pd.read_excel('datasets/dollar.xlsx', usecols=['data', 'curs'])

# переименуем столбцы для удобства
dollar.rename(columns={'data': 'Дата', 'curs': 'Курс'}, inplace=True)

# фильтруем по дате
dollar = dollar[(dollar.Дата >= start_date) & (dollar.Дата <= end_date)]

# сортируем по дате по возрастанию
dollar = dollar.sort_values(by='Дата')

# ставим индексы с нуля
dollar.reset_index(drop=True, inplace=True)

euro = pd.read_excel('datasets/euro.xlsx', usecols=['data', 'curs'])

# переименуем столбцы для удобства
euro.rename(columns={'data': 'Дата', 'curs': 'Курс'}, inplace=True)

# фильтруем по дате
euro = euro[(euro.Дата >= start_date) & (euro.Дата <= end_date)]

# сортируем по дате по возрастанию
euro = euro.sort_values(by='Дата')

# ставим индексы с нуля
euro.reset_index(drop=True, inplace=True)

deposit_rates = pd.read_excel('datasets/deposit_rates.xlsx')

# фильтруем по дате
start_index = deposit_rates.loc[deposit_rates.Месяц == datetime_str_transform(date=start_date)].index[0]
end_index = deposit_rates.loc[deposit_rates.Месяц == datetime_str_transform(date=end_date)].index[0]
deposit_rates = deposit_rates.iloc[start_index:end_index + 1]

# ставим индексы с нуля
deposit_rates.reset_index(drop=True, inplace=True)

loan_rates = pd.read_excel('datasets/loan_rates.xlsx')

# фильтруем по дате
start_index = loan_rates.loc[loan_rates.Месяц == datetime_str_transform(date=start_date)].index[0]
end_index = loan_rates.loc[loan_rates.Месяц == datetime_str_transform(date=end_date)].index[0]
loan_rates = loan_rates.iloc[start_index:end_index + 1]

# ставим индексы с нуля
loan_rates.reset_index(drop=True, inplace=True)

# #количество нейронов в слоях, из которых будет сложен входной слой сети
rnn_size = 4
dense_size = 4

# #количество нейронов выходного слоя (равняется количеству возможных действий)
possible_actions = [round(i, 1) for i in np.arange(-1.0, 1.1, 0.1)]

# начинаем с этой даты
start_date = dt.datetime(year=2013, month=10, day=1)

# создаем параметры для передачи в конструктор класса среды
columns = ['Дата', 'Сумма', 'Ставка', 'Срок', 'День_платежа']
deposits, loans = pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)
min_V, max_V = 100_000, 100_000_000

# создание среды
env = Env(start_date, min_V, max_V, deposits, loans, deposit_predictor, loan_predictor)


# Функция для подготовки входных данных
def input_data(env):
    # данные для входного слоя LSTM
    lstm_input_data = env.get_state()
    lstm_input_data = np.array(lstm_input_data).reshape((1, rnn_size))

    # данные для входного слоя Dense
    dense_input_data = env.deposit_rates
    dense_input_data = np.array(dense_input_data).reshape((1, dense_size))

    return lstm_input_data, dense_input_data


def get_result(days: int, env: object):
    # создаем две копии среды, чтобы сравнить результаты со средневзвешенными ставками и теми, что предлагает нейросеть
    q_env = deepcopy(env)
    prediction = []
    actions = []
    # собираем данные за days дней
    for _ in range(days):
        # считаем Q-значение при применении ставок, рекомендуемых нейросетью
        lstm_input_data, dense_input_data = input_data(q_env)
        action = np.array(q_env.deposit_rates) + \
                 possible_actions[np.argmax(Q_network.predict({'LSTM': lstm_input_data, 'Dense': dense_input_data}))]
        q_env.update_state(action)
        prediction.append(q_env.Q)
        actions.append([round(i, 2) for i in action])
    return {'prediction': prediction, 'actions': actions}


get_result(days=30, env=env)

if __name__ == '__main__':
    print(get_result(days=30, env=env))