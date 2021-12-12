import os
import random
import re
import nltk
import pandas as pandas
import pymorphy2
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# паттерн для регулярного выражения (удаления знаков)
punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“„—…–«»*IVXL0123456789"""
punctuation_pattern = re.compile('[%s]' % re.escape(punctuation))

# Морфологический анализатор
morph = pymorphy2.MorphAnalyzer()

# Стоп слова
nltk.download("stopwords")
stopwords = stopwords.words("russian")
stopwords = list(stopwords)
stopwords.append("это")
stopwords.append("глава")


def read_file(path, encoding="windows-1251"):
    with open(path, encoding=encoding) as f:
        return f.read()


def read_date(root_path="data"):

    # Считываем авторов
    class_names = os.listdir(root_path)
    texts = {}

    for author in class_names:

        # добавляем автора в словарь, если его нет
        if texts.get(author) is None:
            texts.update({author: []})

        # Считываем имена файлов произведения автора
        name_path = root_path + "/" + author
        file_names = os.listdir(name_path)
        # бегаем по именам файлов произведения
        for file_name in file_names:
            # Загружаем произведение в кодировке windows-1251
            try:
                text = read_file(name_path + "/" + file_name)
            except Exception as e:
                # загружаем произведение в кодировке utf-8 если не получилось в предыдущей
                text = read_file(name_path + "/" + file_name, encoding="utf-8")
            # добавляем в словарь произведение к данному автору
            texts[author].append(text)

            # if texts.get(name).get(file_name) is None:
            #     texts[name].update({file_name: text})

    return texts


def data_split(texts, count=500):
    """ Разделяет произведения авторов на отрезки в количестве count"""

    authors = [] #массив авторов
    classes = [] # массив классов авторов
    new_texts = [] #массив отрывков

    # Обработка текстов авторов (удаление знаков препинания, лемматизация, удаление стоп слов)
    for author_class, author in enumerate(texts.keys()):
        for text in texts[author]:

            # удаление знаков препинания и \n
            text = punctuation_pattern.sub(' ', text)
            text = text.replace("\n", " ")

            words = []

            # приведение слов к нормальной форме и проверка на стоп-слова,
            # если не стоп слово, то добавляем в список слов
            for word in text.lower().split():
                if word != "":
                    word = morph.parse(word)[0].normal_form
                    if word not in stopwords:
                        words.append(word)

            # определяем количество слов в отрезке
            word_count = len(words) // count
            segment_list = []
            for i in range(count):
                # извлекаем из списка слов word_count слов и объединяем в отрывок и так count раз
                segment_list.append(" ".join(words[i * word_count: (i + 1) * word_count]))
                authors.append(author)
                classes.append(author_class)

            new_texts.extend(segment_list)

    # Создаем датафрейм где первый столбец авторы, второй столбец открывок автора
    return pandas.DataFrame({'authors': authors, "classes": classes, 'segment_texts': new_texts})


def get_test_text(texts):
    """ Делит произведения автора на 500 открывков, так с каждым автором"""

    authors = []  # массив авторов
    new_texts = []  # массив отрывков

    # Обработка текстов авторов (удаление знаков препинания, лемматизация, удаление стоп слов)
    for author in texts.keys():
        text_all = ""
        count_text = 0
        for text in texts[author]:
            text_all += " " + text
            count_text += 1

        # удаление знаков препинания и \n
        text_all = punctuation_pattern.sub(' ', text_all)
        text_all = text_all.replace("\n", " ")
        words = []

        # слова в виде массива
        text_words = text_all.lower().split()
        for word in text_words:
            if word != "":
                # нормальная форма слова
                word = morph.parse(word)[0].normal_form
                if word not in stopwords:
                    words.append(word)

        # извлекаем из списка слов word_count слов и объединяем в отрывок и так count раз
        segment_list = []
        l = len(words) // 500
        for i in range(500):
            segment_list.append(" ".join(words[i * l: (i + 1) * l]))
            authors.append(author)

        new_texts.extend(segment_list)

    # Создаем датафрейм где первый столбец авторы, второй столбец открывок автора
    return pandas.DataFrame({'authors': authors, 'segment_texts': new_texts})


def save_pickle(path, data):
    """ Сериaлизация объектов в файл (тфидв, матрица)"""
    pickle.dump(data, open(path, "wb"))


def load_pickle(path=""):
    """ Десериaлизиация файлов в объекты"""
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    # Получаем данные в виде таблицы
    data = data_split(read_date())
    # Список открывков
    corpus = data["segment_texts"].values
    # Векторизатор тф идф
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    # Получаем матрицу
    X = vectorizer.fit_transform(corpus)

    # сохраняем матрицу,тфидв и данные
    save_pickle("matrix.pickle", X)
    save_pickle("tfidf.pickle", vectorizer)
    data.to_csv("data.csv", encoding='utf-8')


    # #test
    # # Получаем данные в виде таблицы
    # data = get_test_text(read_date(root_path="input_data"))
    # # Список открывков
    # corpus = data["segment_texts"].values
    # # Векторизатор тф идф
    # vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    # # Получаем матрицу
    # X = vectorizer.fit_transform(corpus)
    #
    # # сохраняем матрицу,тфидв и данные
    # save_pickle("matrix_test.pickle", X)
    # save_pickle("tfidf_test.pickle", vectorizer)
    # data.to_csv("data_test.csv", encoding='utf-8')
