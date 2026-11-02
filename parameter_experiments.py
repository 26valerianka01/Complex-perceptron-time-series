# Эксперименты с гиперпараметрами сложного перцептрона
# Запускает несколько конфигураций и строит графики восстановления / ошибки.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Загрузка данных
file_path = 'TATN.xlsx'
df = pd.read_excel(file_path)
numeric_cols = df.select_dtypes(include=[np.number]).columns
series = df[numeric_cols[0]].dropna().to_numpy(dtype=float)

# Функции для нормализации и построения датасета
def normalize(data: np.ndarray, delta: float = 0.05):
    x_min = data.min()
    x_max = data.max()
    z = (data - x_min) / (x_max - x_min)
    z = delta + z * (1.0 - 2.0 * delta)
    return z, x_min, x_max

def denormalize(z: np.ndarray, x_min: float, x_max: float, delta: float = 0.05):
    return (z - delta) / (1.0 - 2.0 * delta) * (x_max - x_min) + x_min

def build_dataset(series: np.ndarray, n_in: int, n_out: int, n_cross: int):
    total = n_in + n_out - n_cross
    X_list, T_list = [], []
    k = 0
    while k + total <= len(series):
        x_vec = series[k:k + n_in]
        t_start = k + n_in - n_cross
        t_vec = series[t_start:t_start + n_out]
        X_list.append(x_vec)
        T_list.append(t_vec)
        k += 1
    return np.stack(X_list), np.stack(T_list)

def reconstructed_series(series_len: int, X_out: np.ndarray, n_in: int, n_out: int, n_cross: int):
    rec = np.zeros(series_len, dtype=float)
    cnt = np.zeros(series_len)
    n_samples = X_out.shape[0]
    for d in range(n_samples):
        t_start = d + n_in - n_cross
        for j in range(n_out):
            idx = t_start + j
            if idx < series_len:
                rec[idx] += X_out[d, j]
                cnt[idx] += 1
    for i in range(series_len):
        if cnt[i] > 0:
            rec[i] /= cnt[i]
        else:
            rec[i] = np.nan
    return rec

# Класс нейронной сети
class ComplexPerceptron:
    def __init__(self, n_in: int, n_hidden: int, n_out: int, lr: float = 0.01):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.lr = lr
        rng = np.random.default_rng(seed=42)
        self.W1 = rng.uniform(-0.5, 0.5, size=(n_in + 1, n_hidden))
        self.W2 = rng.uniform(-0.5, 0.5, size=(n_hidden + 1, n_out))

    def _add_bias(self, X: np.ndarray):
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def forward(self, X: np.ndarray):
        X_tilde = self._add_bias(X)
        S_hidden = X_tilde @ self.W1
        X_hidden = 1.0 / (1.0 + np.exp(-S_hidden))
        X_hidden_tilde = self._add_bias(X_hidden)
        S_out = X_hidden_tilde @ self.W2
        X_out = 1.0 / (1.0 + np.exp(-S_out))
        return X_out, X_hidden_tilde, X_tilde, S_out

    def loss(self, X_out: np.ndarray, T: np.ndarray):
        return float(np.sum((X_out - T) ** 2))

    def train(self, X: np.ndarray, T: np.ndarray, epochs: int = 1000, batch_size: int = 64, verbose: bool = True,
              X_val: np.ndarray = None, T_val: np.ndarray = None, patience: int = 5):
        history_train, history_val = [], []
        n_samples = X.shape[0]
        best_val = np.inf
        best_epoch = 0
        epochs_without_improve = 0

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            T_shuffled = T[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                T_batch = T_shuffled[i:i + batch_size]

                X_out, X_hidden_tilde, X_tilde, S_out = self.forward(X_batch)

                Delta_out = (X_out - T_batch) * X_out * (1.0 - X_out)
                Delta_hidden = (Delta_out @ self.W2[1:, :].T) * X_hidden_tilde[:, 1:] * (1.0 - X_hidden_tilde[:, 1:])

                grad_W2 = X_hidden_tilde.T @ Delta_out
                grad_W1 = X_tilde.T @ Delta_hidden

                self.W2 -= self.lr * grad_W2
                self.W1 -= self.lr * grad_W1

            X_out_full, _, _, _ = self.forward(X)
            E_train = self.loss(X_out_full, T)
            history_train.append(E_train)

            if X_val is not None and T_val is not None:
                X_out_val, _, _, _ = self.forward(X_val)
                E_val = self.loss(X_out_val, T_val)
                history_val.append(E_val)

                if verbose:
                    print(f"Эпоха {epoch + 1:4d}/{epochs}, E_train = {E_train:.6f}, E_test = {E_val:.6f}")

                if E_val < best_val:
                    best_val = E_val
                    best_epoch = epoch + 1
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1

                if epochs_without_improve >= patience:
                    if verbose:
                        print(f"Ранняя остановка: ошибка на тесте не улучшается {patience} эпох подряд.")
                    break
            else:
                if verbose:
                    print(f"Эпоха {epoch + 1:4d}/{epochs}, E_train = {E_train:.6f}")

        if best_epoch == 0:
            best_epoch = len(history_train)

        return history_train, history_val, best_epoch

    def predict(self, X: np.ndarray):
        X_out, _, _, _ = self.forward(X)
        return X_out

# Функция для запуска тестов
def run_test(N_IN, N_HIDDEN, N_OUT, N_CROSS, DELTA, LR, EPOCHS, BATCH_SIZE, TRAIN_RATIO, FUTURE_STEPS, SPLIT_PERCENT):
    series_norm, x_min, x_max = normalize(series, delta=DELTA)
    X, T = build_dataset(series_norm, N_IN, N_OUT, N_CROSS)

    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    n_train = int(n_samples * TRAIN_RATIO)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, T_train = X[train_idx], T[train_idx]
    X_test, T_test = X[test_idx], T[test_idx] if len(test_idx) > 0 else (None, None)

    model = ComplexPerceptron(N_IN, N_HIDDEN, N_OUT, LR)
    history_train, history_test, best_epoch = model.train(
        X_train, T_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=True,
        X_val=X_test, T_val=T_test, patience=5
    )

    X_out_all = model.predict(X)
    rec_norm_full = reconstructed_series(len(series_norm), X_out_all, N_IN, N_OUT, N_CROSS)
    rec_full = denormalize(rec_norm_full, x_min, x_max, delta=DELTA)

    # Построение графиков
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    t_idx = np.arange(len(series))

    axes[0].plot(t_idx, series, "bx", markersize=5, label="Исходные данные", alpha=0.9)
    valid = ~np.isnan(rec_full)
    axes[0].plot(t_idx[valid], rec_full[valid], "r-", label="Восстановление (весь датасет)", alpha=0.9)
    axes[0].set_title(f"Исходные данные и восстановленный ряд (весь датасет)\nПараметры: N_HIDDEN={N_HIDDEN}, EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, LR={LR}")
    axes[0].set_xlabel("Время")
    axes[0].legend()
    axes[0].grid(True)

    split = int(X.shape[0] * SPLIT_PERCENT / 100)
    model_part = ComplexPerceptron(N_IN, N_HIDDEN, N_OUT, LR)
    model_part.train(X[:split], T[:split], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=False)
    X_out_part = model_part.predict(X)
    rec_norm_part = reconstructed_series(len(series_norm), X_out_part, N_IN, N_OUT, N_CROSS)
    rec_part = denormalize(rec_norm_part, x_min, x_max, delta=DELTA)

    valid1 = ~np.isnan(rec_part)
    axes[1].plot(t_idx, series, "bx", markersize=5, label="Исходные данные", alpha=0.9)
    axes[1].plot(t_idx[valid1], rec_part[valid1], "r-", label="Восстановление (часть датасета)", alpha=0.9)
    boundary_idx = split + N_IN - N_CROSS - 1
    axes[1].axvline(x=boundary_idx, color="gray", linestyle="--", label="Граница обучающей выборки")
    axes[1].set_title(f"Исходные данные и восстановленный ряд (часть датасета)\nПараметры: N_HIDDEN={N_HIDDEN}, EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, LR={LR}")
    axes[1].set_xlabel("Время")
    axes[1].legend()
    axes[1].grid(True)

    series_ext = series_norm.tolist()
    future_norm_list = []
    for _ in range(FUTURE_STEPS):
        last_input = np.array(series_ext[-N_IN:]).reshape(1, -1)
        pred_segment = model.predict(last_input)[0]
        next_point = float(pred_segment[-1])
        future_norm_list.append(next_point)
        series_ext.append(next_point)

    future_norm = np.array(future_norm_list)
    future_real = denormalize(future_norm, x_min, x_max, delta=DELTA)
    future_idx = np.arange(len(series), len(series) + FUTURE_STEPS)

    valid2 = ~np.isnan(rec_full)
    axes[2].plot(t_idx, series, "bx", markersize=5, label="Исходные данные", alpha=0.9)
    axes[2].plot(t_idx[valid2], rec_full[valid2], "r-", label="Модель (весь датасет)", alpha=0.9)
    axes[2].plot(future_idx, future_real, "r--", label="Экстраполяция (прогноз)", alpha=0.9)
    axes[2].set_title(f"Исходные данные и экстраполяция в будущее\nПараметры: N_HIDDEN={N_HIDDEN}, EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, LR={LR}")
    axes[2].set_xlabel("Время")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    fig_err, ax_err = plt.subplots(1, 1, figsize=(8, 4), sharex=True)
    epochs_axis = np.arange(1, len(history_train) + 1)
    ax_err.plot(epochs_axis, history_train, "b-", label="Ошибка обучения")
    if history_test:
        ax_err.plot(epochs_axis[:len(history_test)], history_test, "r-", label="Ошибка теста")
        ax_err.axvline(x=best_epoch, color="gray", linestyle="--", label=f"Лучшая эпоха ({best_epoch})")
    ax_err.set_title(f"Ошибки обучения и тестирования\nПараметры: N_HIDDEN={N_HIDDEN}, EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, LR={LR}")
    ax_err.set_xlabel("Эпоха")
    ax_err.set_ylabel("Ошибка")
    ax_err.grid(True)
    ax_err.legend()
    plt.tight_layout()
    plt.show()

# Запуск тестов
print("Тест 1: Стандартные параметры")
run_test(N_IN=10, N_HIDDEN=5, N_OUT=5, N_CROSS=3, DELTA=0.05, LR=0.01, EPOCHS=1000, BATCH_SIZE=16, TRAIN_RATIO=0.8, FUTURE_STEPS=10, SPLIT_PERCENT=80)

print("Тест 2: Увеличено количество нейронов в скрытом слое")
run_test(N_IN=10, N_HIDDEN=10, N_OUT=5, N_CROSS=3, DELTA=0.05, LR=0.01, EPOCHS=1000, BATCH_SIZE=16, TRAIN_RATIO=0.8, FUTURE_STEPS=10, SPLIT_PERCENT=80)

print("Тест 3: Уменьшено количество эпох")
run_test(N_IN=10, N_HIDDEN=5, N_OUT=5, N_CROSS=3, DELTA=0.05, LR=0.01, EPOCHS=500, BATCH_SIZE=16, TRAIN_RATIO=0.8, FUTURE_STEPS=10, SPLIT_PERCENT=80)

print("Тест 4: Увеличен размер минибатча")
run_test(N_IN=10, N_HIDDEN=5, N_OUT=5, N_CROSS=3, DELTA=0.05, LR=0.01, EPOCHS=1000, BATCH_SIZE=32, TRAIN_RATIO=0.8, FUTURE_STEPS=10, SPLIT_PERCENT=80)

print("Тест 5: Увеличен шаг градиентного спуска")
run_test(N_IN=10, N_HIDDEN=5, N_OUT=5, N_CROSS=3, DELTA=0.05, LR=0.05, EPOCHS=1000, BATCH_SIZE=16, TRAIN_RATIO=0.8, FUTURE_STEPS=10, SPLIT_PERCENT=80)
