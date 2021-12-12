import pandas
from sklearn.model_selection import train_test_split
import Levenshtein
from main import show_plot

if __name__ == '__main__':
    # Загружаем tfidf, матрицу, и датафрейм (данные)
    data = pandas.read_csv("data.csv")
    # Список авторов
    authors_classes = data["authors"].unique()
    # разделение данных на обучающую 90% и тестовую 10%
    X_train, X_test, y_train, y_test = train_test_split(data["segment_texts"].values, data["authors"], test_size=0.10,
                                                        random_state=27)
    data_test = pandas.DataFrame({'authors': y_train, 'segment_texts': X_train})

    # предсказывание на 10% и вывод диаграммы
    y_pred = []
    # Список открывков
    for text_test, author_test in zip(X_test, y_test):
        diff = 99999999999999
        author_temp = ""
        for text_train, author_train in zip(X_train, y_train):
            _diff = Levenshtein.distance(text_test, text_train)
            if _diff < diff:
                diff = _diff
                author_temp = author_train

        y_pred.append(author_temp)

    show_plot(y_pred, y_test, authors_classes)

    # предсказывание на тестовых данных и вывод диаграммы

    data_test = pandas.read_csv("data_test.csv")
    X_test = data_test["segment_texts"]
    y_test = data_test["authors"]
    y_pred = []

    for text_test, author_test in zip(X_test, y_test):
        diff = 99999999999999
        author_temp = ""
        for text_train, author_train in zip(X_train, y_train):
            _diff = Levenshtein.distance(text_test, text_train)
            if _diff < diff:
                diff = _diff
                author_temp = author_train

        y_pred.append(author_temp)

    show_plot(y_pred, y_test, authors_classes)