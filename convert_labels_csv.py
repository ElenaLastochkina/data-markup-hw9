import json
import pandas as pd
 
# Загрузка размеченных данных из файла JSON
with open('labeled_dataset.json', 'r') as file:
    labeled_data = json.load(file)
 
# Преобразование данных в формат DataFrame
data = pd.DataFrame(labeled_data)
 
# Сохранение данных в новом файле "labeled_dataset.csv"
data.to_csv('labeled_dataset.csv', index=False)