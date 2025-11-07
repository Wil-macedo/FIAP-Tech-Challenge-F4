from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from sklearn.model_selection import train_test_split
from processData import download_data, get_scaler, create_dataset
import mlflow
import math
import os

# Configuração do experimento MLflow
mlflow.set_experiment("Tech Challenge F4")

# Carregamento e pré-processamento dos dados
print("Baixando dados históricos...")
data = download_data()

if data is None:
    print("Erro: Não foi possível baixar os dados.")
    quit()

print("Normalizando dados...")
scaled_data = get_scaler().fit_transform(data)

print("Criando dataset...")
x, y = create_dataset(scaled_data, 60)
x = x.reshape(x.shape[0], x.shape[1], 1)

print("Dividindo em treino e teste...")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)


print("Iniciando treinamento do modelo LSTM...")
with mlflow.start_run():
    # Ativa o AutoLogging do MLflow
    mlflow.tensorflow.autolog()

    # Construção do modelo LSTM
    print("Construindo arquitetura do modelo...")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])

    # Compilação do modelo
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

    # Treinamento
    print("Treinando modelo...")
    model.fit(x_train, y_train, epochs=15, batch_size=32, validation_data=(x_test, y_test))

    # Salvamento do modelo
    model_path = os.path.join("modelFiles", "my_model.keras")
    model.save(model_path)
    print(f"Modelo salvo em: {model_path}")

    # Avaliação do modelo
    print("Avaliando modelo...")
    predictions = model.predict(x_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    # Log das métricas
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    print(f"\nMétricas de Avaliação:")
    print(f"  - MSE: {mse:.4f}")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - R² Score: {r2:.4f}")
    print("\nTreinamento concluído com sucesso!")