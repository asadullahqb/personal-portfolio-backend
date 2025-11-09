from app.utils.model_loader import load_model_a

def predict(input_features):
    model = load_model_a()
    pred = model.predict([input_features])
    return {"prediction": int(pred[0])}
