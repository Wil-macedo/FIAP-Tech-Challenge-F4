from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os

# Constantes
scaler_file: str = os.path.join("modelFiles", "scaler.pkl")
model_files: str = "modelFiles"


def download_data(symbol: str = 'MSFT', start: str = '2010-01-01', end: str = '2025-01-01') -> pd.DataFrame:
    """
    Baixa dados históricos de ações do Yahoo Finance.

    Args:
        symbol: Símbolo da ação (padrão: MSFT).
        start: Data de início (formato: YYYY-MM-DD).
        end: Data de fim (formato: YYYY-MM-DD).

    Returns:
        DataFrame com preços de fechamento.
    """
    data = yf.download(symbol, start=start, end=end)
    data = data[['Close']]
    return data


def get_scaler() -> MinMaxScaler:
    """
    Carrega ou cria um scaler MinMaxScaler.

    Returns:
        Objeto MinMaxScaler configurado.
    """
    if os.path.exists(scaler_file):
        scaler = joblib.load(scaler_file)
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        joblib.dump(scaler, scaler_file)

    return scaler


def create_dataset(data: np.ndarray, time_step: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cria dataset para treinamento de LSTM.

    Args:
        data: Array numpy com dados normalizados.
        time_step: Número de passos temporais (padrão: 60).

    Returns:
        Tupla (X, y) com dados de entrada e saída.
    """
    x, y = [], []
    for i in range(len(data) - time_step - 1):
        x.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])

    return np.array(x), np.array(y)