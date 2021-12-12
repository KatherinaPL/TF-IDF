# импортируем библиотеки
import numpy
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split

from data_parser import load_pickle


def show_plot(pred, y_true, labels):
    # строим тепловую карту
    import matplotlib.pyplot as plt
    # строим матрицу ошибок
    cm = confusion_matrix(y_true, pred)

    fig, ax = plt.subplots()
    im = ax.imshow(cm)

    # We want to show all ticks...
    ax.set_xticks(numpy.arange(len(labels)))
    ax.set_yticks(numpy.arange(len(labels)))
    # добавляем классов на каждую ось
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    #  Добавляем подписи к осям:
    ax.set_xlabel('Предсказанные классы')
    ax.set_ylabel('Истинные классы')

    # поворачиваем подписи на 45 градусов
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # загружаем данные и устанавливаем настройки в ячейки
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    fig.tight_layout()
    # выводим график и сохраняем
    plt.show()
    plt.savefig("img.png")


if __name__ == '__main__':
    # Загружаем tfidf, матрицу, и датафрейм (данные)
    vectorizer = load_pickle("tfidf.pickle")
    X = load_pickle("matrix.pickle")
    data = pandas.read_csv("data.csv")
    # Список авторов
    authors_classes = data["authors"].unique()

    # проверка косинусного расстояния на тестовых данных

    data_test = pandas.read_csv("data_test.csv")
    segment_texts = data_test["segment_texts"]
    authors = data_test["authors"]

    # истинные классы и предсказанные авторов
    y_true, y_pred = [], []

    for i, text in enumerate(segment_texts):
        # получаем вектор для входных данных
        new_entry = vectorizer.transform([text])
        # косинусное сравнение (массив) (.flatten() -> преобразует матрицу в массив одномерный)
        cosine_similarities = linear_kernel(new_entry, X).flatten()

        #делаем копию оригинальных данных (для изменения)
        copy_data = data.copy()
        # добавляем результаты косинусного сравнения
        copy_data['cos_similarities'] = cosine_similarities
        # print(cosine_similarities)

        # #Сортируем датафрейм по убыванию косинусного совпадения
        copy_data = copy_data.sort_values(by=['cos_similarities'], ascending=[0])

        # извлекаем автора с самым большим косинусным схождением
        index = copy_data.index[0]
        # сохраняем истинного автора и предсказанного
        y_true.append(authors[i])
        y_pred.append(copy_data["authors"][index])

    show_plot(y_pred, y_true, authors_classes)

    # проверка косинусного расстояния на 10%

    # разделение данных на обучающую 90% и тестовую 10%
    X_train, X_test, y_train, y_test = train_test_split(data["segment_texts"].values, data["authors"], test_size=0.10, random_state=27)
    data_test = pandas.DataFrame({'authors': y_train, 'segment_texts': X_train})
    y_pred = []
    # Список открывков
    corpus = data["segment_texts"].values
    # Векторизатор тф идф
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    # Получаем матрицу

    X = vectorizer.fit_transform(X_train)
    for i, text in enumerate(X_test):
        # получаем вектор для входных данных
        new_entry = vectorizer.transform([text])
        # косинусное сравнение (массив) (.flatten() -> преобразует матрицу в массив одномерный)
        cosine_similarities = linear_kernel(new_entry, X).flatten()

        #делаем копию оригинальных данных (для изменения)
        copy_data = data_test.copy()
        # добавляем результаты косинусного сравнения
        copy_data['cos_similarities'] = cosine_similarities
        # print(cosine_similarities)

        # #Сортируем датафрейм по убыванию косинусного совпадения
        copy_data = copy_data.sort_values(by=['cos_similarities'], ascending=[0])

        # извлекаем автора с самым большим косинусным схождением
        index = copy_data.index[0]
        # сохраняем истинного автора и предсказанного
        y_pred.append(copy_data["authors"][index])

    show_plot(y_pred, y_test, authors_classes)