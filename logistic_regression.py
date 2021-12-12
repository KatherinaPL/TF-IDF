import pandas
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from data_parser import load_pickle
from main import show_plot

vectorizer = load_pickle("tfidf.pickle")
X = load_pickle("matrix.pickle")
data = pandas.read_csv("data.csv")
authors_classes = data["authors"].unique()

le = preprocessing.LabelEncoder()


# Переводим авторов в числовые метки Гоголь - 0 и тд
l = len(set(list(data["authors"])))
le.fit(data["authors"])
data["labels"] = le.transform(data["authors"])

# разделение данных на обучающую 90% и тестовую 10%
X_train, X_test, y_train, y_test = train_test_split(X, data["labels"], test_size=0.10, random_state=27)

classifier = LogisticRegression(random_state=0, verbose=1, n_jobs=6, max_iter=10)
classifier.fit(X_train, y_train)

prediction = classifier.predict(X_test)

# Оценка точности — простейший вариант оценки работы классификатора
print(accuracy_score(prediction, y_test))
print(confusion_matrix(y_test, prediction))
# Но матрица неточности и отчёт о классификации дадут больше информации о производительности
print(classification_report(prediction, y_test))
show_plot(prediction, y_test, authors_classes)


# Проверка на тестовых данных
data = pandas.read_csv("data_test.csv")
authors_classes = data["authors"].unique()

le = preprocessing.LabelEncoder()


# Переводим авторов в числовые метки Гоголь - 0 и тд
l = len(set(list(data["authors"])))
le.fit(data["authors"])
data["labels"] = le.transform(data["authors"])
y_test = data["labels"]
segment_texts = data["segment_texts"]
prediction = []

# Собираем массив предсказаний от классификатора
for i, text in enumerate(segment_texts):
    new_entry = vectorizer.transform([text])
    pred = classifier.predict(new_entry)
    prediction.append(pred)

# Оценка точности — простейший вариант оценки работы классификатора
print(accuracy_score(prediction, y_test))
print(confusion_matrix(y_test, prediction))
# Но матрица неточности и отчёт о классификации дадут больше информации о производительности
print(classification_report(prediction, y_test))
show_plot(prediction, y_test, authors_classes)