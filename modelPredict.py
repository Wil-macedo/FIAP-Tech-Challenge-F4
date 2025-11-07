from tensorflow.keras.models import load_model as keras_load_model  # type: ignore
from processData import model_files, get_scaler
import numpy as np
import os

# Modelo de dado para predição (exemplo)
x_data = [
    23.3007,
    23.3082,
    23.1652,
    22.9243,
    23.0824,
    22.7887,
    22.6382,
    22.8490,
    23.3082,
    23.2329,
    23.4136,
    23.0297,
    22.5930,
    21.8025,
    22.0735,
    22.2090,
    22.3370,
    21.9531,
    21.2153,
    21.3884,
    21.4261,
    21.5541,
    20.9593,
    21.0948,
    20.8690,
    21.0873,
    21.0722,
    21.1701,
    21.0271,
    21.4431,
    21.6246,
    21.9120,
    21.7608,
    21.7305,
    21.4280,
    21.6549,
    21.6322,
    21.6851,
    21.9498,
    21.5263,
    21.5263,
    21.6549,
    21.6246,
    21.6549,
    21.7834,
    21.9120,
    22.0709,
    22.1389,
    22.1541,
    22.2146,
    22.4112,
    22.3961,
    22.3810,
    22.3885,
    22.6003,
    22.4264,
    22.6986,
    22.4339,
    22.3810,
    22.5171
]

model = None


def load_model():
    """Carrega o modelo treinado do disco."""
    global model
    model = keras_load_model(os.path.join(model_files, "my_model.keras"))
    model.summary()


def model_predict(x_data: list) -> float:
    """
    Realiza predição com o modelo LSTM.

    Args:
        x_data: Lista com 60 preços de fechamento.

    Returns:
        Preço previsto para o próximo dia.
    """
    global model

    if len(x_data) != 60:
        raise ValueError("É necessário fornecer exatamente 60 valores para predição.")

    if model is None:
        load_model()

    # Preparar dados para predição
    x_array = np.array(x_data).reshape(-1, 1)
    x_scaled = get_scaler().transform(x_array)
    x_reshaped = np.array(x_scaled).reshape(1, 60, 1)

    # Realizar predição
    prediction = model.predict(x_reshaped)
    prediction_value = get_scaler().inverse_transform(prediction)[0][0]

    return round(float(prediction_value), 2)