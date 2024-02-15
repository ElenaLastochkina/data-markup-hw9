import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
 
# Загрузка размеченного набора данных
data = pd.read_csv("labeled_dataset.csv")
 
# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)
 
# Преобразование текстовых данных в векторы признаков
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
 
# Создание и обучение модели машинного обучения
model = LogisticRegression()
model.fit(X_train, y_train)
 
# Оценка точности модели
accuracy = model.score(X_test, y_test)
print("Точность модели:", accuracy)

# Загрузка нового набора данных, который не содержит меток
new_data = pd.read_csv("new_dataset.csv")
 
# Преобразование текстовых данных в векторы признаков
X_new = vectorizer.transform(new_data["text"])
 
# Применение обученной модели к новым данным
predictions = model.predict(X_new)
 
# Вывод предсказанных меток
for prediction in predictions:
    print(prediction)
 