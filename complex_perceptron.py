import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# =============================================================================
# НАСТРОЙКИ
# =============================================================================
EXCEL_NAME   = "TATN.xlsx"   # файл с котировками
COLUMN_NAME  = None          # столбец с данными (None = первый числовой)

N_IN    = 10    # количество входных точек
N_HIDDEN = 5    # количество нейронов в скрытом слое
N_OUT   = 5     # количество выходных точек
N_CROSS = 3     # количество перекрёстных точек

DELTA   = 0.05  # нормировка ряда в [DELTA, 1-DELTA] для сигмоиды
LR      = 0.01  # шаг градиентного спуска λ
EPOCHS  = 1000  # количество эпох обучения
BATCH_SIZE = 16 # размер минибатча

SPLIT_PERCENT = 80       # процент окон для обучения на части выборки
TRAIN_RATIO   = 0.8      # доля данных для обучения (остальное - тест)
FUTURE_STEPS  = 10       # сколько точек экстраполировать в будущее
FIG_SIZE      = (12, 10) # размер окна с графиками
VERBOSE       = True     # выводить прогресс по эпохам
SAVE_FIGURES = True      # сохранять графики в assets/results
SHOW_FIGURES = True      # показывать интерактивные окна matplotlib

# =============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# =============================================================================

def find_excel(name: str) -> str:
    base, ext = os.path.splitext(name)
    for folder in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        for variant in (name, base + ".XLSX", base + ".xlsx"):
            path = os.path.join(folder, variant)
            if os.path.isfile(path):
                return path
    return name

def load_series(path: str, column: str | None = None) -> np.ndarray:
    df = pd.read_excel(path)
    if column is not None:
        series = df[column]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("В файле нет числовых столбцов.")
        series = df[numeric_cols[0]]
    return series.dropna().to_numpy(dtype=float)

# =============================================================================
# 2. НОРМИРОВКА
# =============================================================================

def normalize(data: np.ndarray, delta: float = 0.05) -> tuple[np.ndarray, float, float]:
    x_min = data.min()
    x_max = data.max()
    z = (data - x_min) / (x_max - x_min)
    z = delta + z * (1.0 - 2.0 * delta)
    return z, x_min, x_max

def denormalize(z: np.ndarray, x_min: float, x_max: float, delta: float = 0.05) -> np.ndarray:
    return (z - delta) / (1.0 - 2.0 * delta) * (x_max - x_min) + x_min

# =============================================================================
# 3. ПОСТРОЕНИЕ ДАТАСЕТА
# =============================================================================

def build_dataset(series: np.ndarray, n_in: int, n_out: int, n_cross: int) -> tuple[np.ndarray, np.ndarray]:
    total = n_in + n_out - n_cross
    X_list, T_list = [], []
    k = 0
    while k + total <= len(series):
        x_vec = series[k : k + n_in]
        t_start = k + n_in - n_cross
        t_vec = series[t_start : t_start + n_out]
        X_list.append(x_vec)
        T_list.append(t_vec)
        k += 1
    return np.stack(X_list), np.stack(T_list)

def reconstructed_series(series_len: int, X_out: np.ndarray, n_in: int, n_out: int, n_cross: int) -> np.ndarray:
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

# =============================================================================
# 4. ФУНКЦИЯ АКТИВАЦИИ - СИГМОИДА
# =============================================================================

def sigmoid(S: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-S))

def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    return x * (1.0 - x)

# =============================================================================
# 5. СЛОЖНЫЙ ПЕРЦЕПТРОН С ОДНИМ СКРЫТЫМ СЛОЕМ (с минибатчами)
# =============================================================================

class ComplexPerceptron:
    def __init__(self, n_in: int, n_hidden: int, n_out: int, lr: float = 0.01) -> None:
        self.n_in    = n_in
        self.n_hidden = n_hidden
        self.n_out   = n_out
        self.lr      = lr

        # Инициализация весов: W1 (вход → скрытый), W2 (скрытый → выход)
        rng = np.random.default_rng(seed=42)
        self.W1 = rng.uniform(-0.5, 0.5, size=(n_in + 1, n_hidden))
        self.W2 = rng.uniform(-0.5, 0.5, size=(n_hidden + 1, n_out))

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Прямой проход: вход → скрытый → выход
        X_tilde = self._add_bias(X)
        S_hidden = X_tilde @ self.W1
        X_hidden = sigmoid(S_hidden)
        X_hidden_tilde = self._add_bias(X_hidden)
        S_out = X_hidden_tilde @ self.W2
        X_out = sigmoid(S_out)
        return X_out, X_hidden_tilde, X_tilde, S_out

    def loss(self, X_out: np.ndarray, T: np.ndarray) -> float:
        return float(np.sum((X_out - T) ** 2))

    def train(
        self,
        X: np.ndarray,
        T: np.ndarray,
        epochs: int = 1000,
        batch_size: int = 64,
        verbose: bool = True,
        X_val: np.ndarray | None = None,
        T_val: np.ndarray | None = None,
        patience: int = 5,
    ) -> tuple[list[float], list[float], int]:
        """
        Обучение с минибатчами.
        Возвращает:
          - history_train: значения ошибки на обучающей выборке по эпохам,
          - history_val:   значения ошибки на тестовой (валидационной) выборке по эпохам,
          - best_epoch:    номер эпохи с наименьшей ошибкой на тестовой выборке.

        Если X_val/T_val не заданы, history_val будет нулевым списком,
        а best_epoch совпадёт с количеством реально пройденных эпох.
        """
        history_train: list[float] = []
        history_val: list[float] = []

        n_samples = X.shape[0]
        best_val = np.inf
        best_epoch = 0
        epochs_without_improve = 0

        for epoch in range(epochs):
            # Перемешиваем индексы
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            T_shuffled = T[indices]

            # Разбиваем на минибатчи
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                T_batch = T_shuffled[i:i + batch_size]

                # Прямой проход
                X_out, X_hidden_tilde, X_tilde, S_out = self.forward(X_batch)

                # Обратное распространение ошибки
                Delta_out = (X_out - T_batch) * sigmoid_grad(X_out)
                Delta_hidden = (Delta_out @ self.W2[1:, :].T) * sigmoid_grad(X_hidden_tilde[:, 1:])

                # Обновление весов
                grad_W2 = X_hidden_tilde.T @ Delta_out
                grad_W1 = X_tilde.T @ Delta_hidden

                self.W2 -= self.lr * grad_W2
                self.W1 -= self.lr * grad_W1

            # Ошибка на обучающей выборке
            X_out_full, _, _, _ = self.forward(X)
            E_train = self.loss(X_out_full, T)
            history_train.append(E_train)

            # Ошибка на тестовой (валидационной) выборке
            if X_val is not None and T_val is not None:
                X_out_val, _, _, _ = self.forward(X_val)
                E_val = self.loss(X_out_val, T_val)
                history_val.append(E_val)

                if verbose:
                    print(f"Epoch {epoch + 1:4d}/{epochs},  E_train = {E_train:.6f},  E_test = {E_val:.6f}")

                # Ранняя остановка по переобучению
                if E_val < best_val:
                    best_val = E_val
                    best_epoch = epoch + 1
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1

                if epochs_without_improve >= patience:
                    if verbose:
                        print(f"Остановка обучения: ошибка на тесте не улучшается {patience} эпох подряд.")
                    break
            else:
                if verbose:
                    print(f"Epoch {epoch + 1:4d}/{epochs},  E_train = {E_train:.6f}")

        # Если валидационной выборки нет, best_epoch = количество пройденных эпох
        if best_epoch == 0:
            best_epoch = len(history_train)

        return history_train, history_val, best_epoch

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_out, _, _, _ = self.forward(X)
        return X_out

# =============================================================================
# 6. ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================

def _results_dir() -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "results")
    os.makedirs(path, exist_ok=True)
    return path


def main() -> None:
    excel_path = find_excel(EXCEL_NAME)
    if not os.path.isfile(excel_path):
        print(f"Файл {excel_path} не найден.")
        return
    print(f"Файл данных: {excel_path}")
    series = load_series(excel_path, column=COLUMN_NAME)
    print(f"Загружено точек: {len(series)}")

    series_norm, x_min, x_max = normalize(series, delta=DELTA)
    X, T = build_dataset(series_norm, n_in=N_IN, n_out=N_OUT, n_cross=N_CROSS)
    print(f"Датасет: {X.shape[0]} окон  |  X: {X.shape}  |  T: {T.shape}")

    # Явное разбиение на обучающую и тестовую выборки для контроля переобучения
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    n_train = int(n_samples * TRAIN_RATIO)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, T_train = X[train_idx], T[train_idx]
    X_test, T_test = X[test_idx], T[test_idx] if len(test_idx) > 0 else (None, None)

    print(f"Обучающая выборка: {X_train.shape[0]} окон, тестовая: {0 if X_test is None else X_test.shape[0]} окон")

    model = ComplexPerceptron(n_in=N_IN, n_hidden=N_HIDDEN, n_out=N_OUT, lr=LR)
    if X_test is not None and T_test is not None:
        history_train, history_test, best_epoch = model.train(
            X_train,
            T_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=VERBOSE,
            X_val=X_test,
            T_val=T_test,
            patience=5,
        )
    else:
        history_train, history_test, best_epoch = model.train(
            X_train,
            T_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=VERBOSE,
            X_val=None,
            T_val=None,
            patience=5,
        )

    print(f"\nФинальная ошибка на обучающей выборке E_train = {history_train[-1]:.6f}")
    if history_test:
        print(f"Лучшее значение ошибки на тесте E_test = {min(history_test):.6f} на эпохе {best_epoch}")

    # Восстановление ряда по всей выборке (для графиков 1 и 2)
    X_out_all = model.predict(X)
    rec_norm_full = reconstructed_series(len(series_norm), X_out_all, N_IN, N_OUT, N_CROSS)
    rec_full = denormalize(rec_norm_full, x_min, x_max, delta=DELTA)

    split = int(X.shape[0] * SPLIT_PERCENT / 100)
    model_part = ComplexPerceptron(n_in=N_IN, n_hidden=N_HIDDEN, n_out=N_OUT, lr=LR)
    model_part.train(X[:split], T[:split], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)
    X_out_part = model_part.predict(X)
    rec_norm_part = reconstructed_series(len(series_norm), X_out_part, N_IN, N_OUT, N_CROSS)
    rec_part = denormalize(rec_norm_part, x_min, x_max, delta=DELTA)

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

    # Окно 1: временные ряды (восстановление и экстраполяция)
    fig, axes = plt.subplots(3, 1, figsize=FIG_SIZE, sharex=False)
    t_idx = np.arange(len(series))

    axes[0].plot(t_idx, series, "bx", markersize=5, label="исходные данные", alpha=0.9)
    valid = ~np.isnan(rec_full)
    axes[0].plot(t_idx[valid], rec_full[valid], "r-", label="восстановление (вся выборка)", alpha=0.9)
    axes[0].set_title("1. Исходные данные и восстановление ряда (обучение на всём датасете)")
    axes[0].set_xlabel("Время ")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t_idx, series, "bx", markersize=5, label="исходные данные", alpha=0.9)
    valid1 = ~np.isnan(rec_part)
    axes[1].plot(t_idx[valid1], rec_part[valid1], "r-", label="восстановление (обучение на части)", alpha=0.9)
    boundary_idx = split + N_IN - N_CROSS - 1
    axes[1].axvline(x=boundary_idx, color="gray", linestyle="--", label="граница обучающей выборки")
    axes[1].set_title("2. Исходные данные и восстановление (обучение на части выборки)")
    axes[1].set_xlabel("Время ")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t_idx, series, "bx", markersize=5, label="исходные данные", alpha=0.9)
    valid2 = ~np.isnan(rec_full)
    axes[2].plot(t_idx[valid2], rec_full[valid2], "r-", label="модель на всём датасете", alpha=0.9)
    axes[2].plot(future_idx, future_real, "r--", label="экстраполяция (прогноз)", alpha=0.9)
    axes[2].set_title("3. Исходные данные и экстраполяция в будущее")
    axes[2].set_xlabel("Время ")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    if SAVE_FIGURES:
        out = _results_dir()
        fig.savefig(os.path.join(out, "series_overview.png"), dpi=150, bbox_inches="tight")
        print(f"Сохранено: {os.path.join(out, 'series_overview.png')}")

    # Окно 2: график ошибок train vs test по эпохам
    if history_train:
        fig_err, ax_err = plt.subplots(1, 1, figsize=(8, 4), sharex=True)
        epochs_axis = np.arange(1, len(history_train) + 1)
        ax_err.plot(epochs_axis, history_train, "b-", label="E_train")
        if history_test:
            ax_err.plot(epochs_axis[:len(history_test)], history_test, "r-", label="E_test")
            ax_err.axvline(x=best_epoch, color="gray", linestyle="--", label=f"лучшая эпоха ({best_epoch})")
        ax_err.set_title("Ошибка на обучающей и тестовой выборках")
        ax_err.set_xlabel("Эпоха")
        ax_err.set_ylabel("E")
        ax_err.grid(True)
        ax_err.legend()
        plt.tight_layout()
        if SAVE_FIGURES:
            out = _results_dir()
            fig_err.savefig(os.path.join(out, "loss_curves.png"), dpi=150, bbox_inches="tight")
            print(f"Сохранено: {os.path.join(out, 'loss_curves.png')}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")

if __name__ == "__main__":
    main()
