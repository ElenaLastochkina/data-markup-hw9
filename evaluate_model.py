from sklearn.metrics import accuracy_score
 
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
 
    # Расчет метрик
    accuracy = accuracy_score(y_test, y_pred)
 
    return accuracy
 
loaded_model = load_model("trained_model.pkl")  
accuracy = evaluate_model(loaded_model, X_test, y_test)