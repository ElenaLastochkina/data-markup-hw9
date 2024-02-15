import pandas as pd
 
def rule_based_labeling(text):
    positive_keywords = ["хороший", "прекрасный", "отличный", "восхитительный"]
    negative_keywords = ["плохой", "ужасный", "отвратительный", "скучный"]
 
    for keyword in positive_keywords:
        if keyword in text:
            return "положительно"
 
    for keyword in negative_keywords:
        if keyword in text:
            return "отрицательно"
 
    return "неопределено"
 
# Загрузка датасета IMDB Dataset.csv
data = pd.read_csv("IMDB Dataset.csv")
 
# Применение правил к данным и добавление меток в новый столбец "метка"
data["метка"] = data["review"].apply(rule_based_labeling)
 
# Сохранение изменений в новом файле "labeled_dataset.csv"
data.to_csv("labeled_dataset.csv", index=False)