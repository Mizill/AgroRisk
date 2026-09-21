"""
pinn_model.py
=============
Загрузка и ИНФЕРЕНС нейросети `pinn_model.pt`.

В прошлой версии сеть только загружалась, а в расчёт риска шло
`degradation_index = None`, то есть модель фактически не использовалась.
Здесь сеть работает по-настоящему: на каждую декаду считается индекс
деградации/эрозионной уязвимости почвы, который входит в композитный индекс
риска и попадает в отчёт.

Архитектура (считана напрямую из ZIP/pickle-контейнера чекпойнта,
без исполнения чужого кода) — 13 → 64 → 128 → 64 → 32 → 1:

    network.0  Linear(13, 64)     network.8   Linear(128, 64)
    network.1  ReLU               network.9   ReLU
    network.2  BatchNorm1d(64)    network.10  BatchNorm1d(64)
    network.3  ReLU               network.11  ReLU
    network.4  Linear(64, 128)    network.12  Linear(64, 32)
    network.5  ReLU               network.13  ReLU
    network.6  BatchNorm1d(128)   network.14  BatchNorm1d(32)
    network.7  ReLU               network.15  ReLU
                                  network.16  Linear(32, 1)

Кроме весов в чекпойнте лежат физические константы RUSLE:
K_factor=0.3, P_factor=0.8, LS_factor=1.1907, max_rusle=erosion_scale=4.2866.

ДВА БЭКЕНДА
-----------
* torch   — если пакет установлен (нужен и для дообучения);
* numpy   — чистая реализация того же прямого прохода (Linear/ReLU/BatchNorm
            в режиме eval). Нужна, чтобы программа работала без тяжёлой
            зависимости и чтобы оффлайн-тест гонял настоящие веса.
Оба бэкенда дают одинаковый результат; выбор — параметром `backend`.

ВАЖНО ПРО КАЛИБРОВКУ
--------------------
В чекпойнте НЕ сохранён scaler входных признаков, поэтому точный масштаб
обучающих данных восстановить нельзя. Подробности и следствия описаны
в README, раздел «Как именно используется сеть». Препроцессинг вынесен
в `features.py`, а перевод сырого выхода сети в индекс 0..1 — в
`calibrate_outputs()` ниже.
"""
from __future__ import annotations

import io
import pickle
import zipfile
from typing import Optional, Sequence

import numpy as np

# --- порядок признаков менять нельзя: он зашит в первый слой сети ---------
INPUT_FEATURES = [
    "R_factor",           # 0  эрозионный потенциал осадков за декаду
    "K_factor",           # 1  эродируемость почвы (константа чекпойнта)
    "LS_factor",          # 2  длина/крутизна склона (константа чекпойнта)
    "C_factor",           # 3  фактор растительного покрова (RUSLE C)
    "P_factor",           # 4  агротехнические практики (константа чекпойнта)
    "precip_mm_decade",   # 5  сумма осадков за декаду, мм
    "precip_norm_ratio",  # 6  отношение к климатической норме
    "temp_mean_c",        # 7  средняя температура декады, °C
    "temp_anom_c",        # 8  аномалия температуры, °C
    "wind_mean_ms",       # 9  средняя скорость ветра, м/с
    "wind_max_ms",        # 10 максимальный порыв, м/с
    "rel_humidity_pct",   # 11 средняя относительная влажность, %
    "soil_moisture_frac", # 12 объёмная влажность почвы 0-7 см, м³/м³
]
INPUT_DIM = len(INPUT_FEATURES)  # 13

PHYSICS_KEYS = ("K_factor", "P_factor", "LS_factor", "max_rusle", "erosion_scale")
BN_EPS = 1e-5

try:                       # torch не обязателен для инференса
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:          # pragma: no cover
    torch = None
    nn = None
    TORCH_AVAILABLE = False


# ===========================================================================
# 1. Чтение чекпойнта без torch
# ===========================================================================
class _OrderedDictStub(dict):
    """dict с __dict__ — pickle кладёт в OrderedDict атрибут _metadata."""


class _Ignored:
    def __init__(self, *args, **kwargs):
        pass


_NUMPY_DTYPES = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64,
    "HalfStorage": np.float16, "LongStorage": np.int64,
    "IntStorage": np.int32, "ShortStorage": np.int16,
    "CharStorage": np.int8, "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


def _rebuild_tensor_v2(storage, storage_offset, size, stride,
                       requires_grad=False, backward_hooks=None, *extra):
    flat, dtype = storage
    size = tuple(size)
    count = int(np.prod(size)) if size else 1
    chunk = np.array(flat[storage_offset:storage_offset + count], dtype=dtype)
    return chunk.reshape(size) if size else chunk.reshape(())


def _rebuild_parameter(data, requires_grad=False, backward_hooks=None, *extra):
    return data


class _SafeUnpickler(pickle.Unpickler):
    """
    Разбирает `data.pkl` чекпойнта, подменяя типы torch на numpy.
    Никакой код из чекпойнта не исполняется: всё незнакомое заменяется
    безвредной заглушкой.
    """

    def __init__(self, file, zf: zipfile.ZipFile, prefix: str, byteorder: str):
        super().__init__(file)
        self._zf, self._prefix, self._byteorder = zf, prefix, byteorder

    def find_class(self, module, name):
        if module == "torch._utils":
            if name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if name == "_rebuild_parameter":
                return _rebuild_parameter
        if module == "torch" and name in _NUMPY_DTYPES:
            return _NUMPY_DTYPES[name]
        if module == "collections" and name == "OrderedDict":
            return _OrderedDictStub
        return _Ignored

    def persistent_load(self, pid):
        storage_type, key = pid[1], pid[2]
        dtype = storage_type if isinstance(storage_type, type) else np.float32
        raw = self._zf.read(f"{self._prefix}data/{key}")
        arr = np.frombuffer(raw, dtype=dtype)
        if self._byteorder == "big":
            arr = arr.byteswap().view(arr.dtype.newbyteorder())
        return (arr, dtype)


def load_raw_checkpoint(path: str) -> dict:
    """
    Возвращает содержимое `pinn_model.pt` как обычный словарь:
    {"model_state": {имя: np.ndarray}, "K_factor": ..., "max_rusle": ...}.
    Работает без torch.
    """
    if not zipfile.is_zipfile(path):
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                f"{path} сохранён в старом (не ZIP) формате torch — "
                "для его чтения нужен установленный пакет torch.")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        state = {k: v.detach().cpu().numpy() for k, v in ck["model_state"].items()}
        out = {k: ck.get(k) for k in PHYSICS_KEYS}
        out["model_state"] = state
        return out

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        pkl = next((n for n in names if n.endswith("data.pkl")), None)
        if pkl is None:
            raise RuntimeError(f"В {path} не найден data.pkl — это не чекпойнт torch.")
        prefix = pkl[:-len("data.pkl")]
        byteorder = "little"
        if f"{prefix}byteorder" in names:
            byteorder = zf.read(f"{prefix}byteorder").decode().strip()
        ck = _SafeUnpickler(io.BytesIO(zf.read(pkl)), zf, prefix, byteorder).load()

    if "model_state" not in ck:
        raise RuntimeError("В чекпойнте нет ключа 'model_state'.")
    return dict(ck)


def extract_physics(ck: dict) -> dict:
    """Физические константы RUSLE из чекпойнта, с безопасными значениями по умолчанию."""
    defaults = {"K_factor": 0.3, "P_factor": 0.8, "LS_factor": 1.0,
                "max_rusle": 1.0, "erosion_scale": 1.0}
    out = {}
    for key in PHYSICS_KEYS:
        value = ck.get(key)
        out[key] = float(value) if value is not None else defaults[key]
    if out["max_rusle"] <= 0:
        out["max_rusle"] = 1.0
    return out


# ===========================================================================
# 2. Бэкенд numpy
# ===========================================================================
class NumpyAgroPINN:
    """Прямой проход сети в numpy (режим eval: BatchNorm по running-статистикам)."""

    LAYERS = [
        ("linear", "network.0"), ("relu", None), ("bn", "network.2"), ("relu", None),
        ("linear", "network.4"), ("relu", None), ("bn", "network.6"), ("relu", None),
        ("linear", "network.8"), ("relu", None), ("bn", "network.10"), ("relu", None),
        ("linear", "network.12"), ("relu", None), ("bn", "network.14"), ("relu", None),
        ("linear", "network.16"),
    ]

    def __init__(self, state: dict):
        self.state = {k: np.asarray(v, dtype=np.float64) for k, v in state.items()}
        w0 = self.state.get("network.0.weight")
        if w0 is None or w0.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Ожидался вход {INPUT_DIM} признаков, в чекпойнте "
                f"{None if w0 is None else w0.shape}.")

    def forward(self, X: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(X, dtype=np.float64))
        for kind, name in self.LAYERS:
            if kind == "relu":
                x = np.maximum(x, 0.0)
            elif kind == "linear":
                x = x @ self.state[f"{name}.weight"].T + self.state[f"{name}.bias"]
            else:  # batchnorm, eval
                mean = self.state[f"{name}.running_mean"]
                var = self.state[f"{name}.running_var"]
                x = (x - mean) / np.sqrt(var + BN_EPS)
                x = x * self.state[f"{name}.weight"] + self.state[f"{name}.bias"]
        return x.reshape(-1)

    __call__ = forward


# ===========================================================================
# 3. Бэкенд torch
# ===========================================================================
if TORCH_AVAILABLE:

    class AgroPINN(nn.Module):
        """Точная копия архитектуры, сохранённой в pinn_model.pt."""

        def __init__(self, input_dim: int = INPUT_DIM):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Linear(64, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.network(x)

else:                                                   # pragma: no cover
    AgroPINN = None


class TorchAgroPINN:
    """Обёртка вокруг AgroPINN с тем же интерфейсом, что у NumpyAgroPINN."""

    def __init__(self, state: dict, device: str = "cpu"):
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch не установлен.")
        self.device = device
        self.module = AgroPINN()
        self.module.load_state_dict(
            {k: torch.as_tensor(np.asarray(v)) for k, v in state.items()})
        self.module.to(device).eval()

    def forward(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(np.atleast_2d(np.asarray(X)), dtype=torch.float32,
                                device=self.device)
            return self.module(t).cpu().numpy().reshape(-1)

    __call__ = forward


def load_model(path: str, backend: str = "auto"):
    """
    Загружает чекпойнт и возвращает `(model, physics)`.

    backend: "auto" (torch, если есть, иначе numpy) | "torch" | "numpy".
    У модели есть метод `forward(X) -> np.ndarray` для матрицы [N, 13].
    """
    ck = load_raw_checkpoint(path)
    physics = extract_physics(ck)
    state = ck["model_state"]

    if backend not in ("auto", "torch", "numpy"):
        raise ValueError("backend должен быть 'auto', 'torch' или 'numpy'.")
    if backend == "torch" and not TORCH_AVAILABLE:
        raise RuntimeError("Запрошен backend='torch', но пакет torch не установлен.")
    use_torch = TORCH_AVAILABLE if backend == "auto" else (backend == "torch")

    model = TorchAgroPINN(state) if use_torch else NumpyAgroPINN(state)
    physics["backend"] = "torch" if use_torch else "numpy"
    return model, physics


# ===========================================================================
# 4. Физика RUSLE и калибровка выхода
# ===========================================================================
def rusle_A(R, K, LS, C, P):
    """RUSLE: A = R * K * LS * C * P — потенциальный смыв почвы."""
    return np.asarray(R) * K * LS * np.asarray(C) * P


def rusle_A_norm(R, C, physics: dict) -> np.ndarray:
    """A, нормированное на max_rusle из чекпойнта → диапазон 0..1."""
    A = rusle_A(R, physics["K_factor"], physics["LS_factor"], C, physics["P_factor"])
    return np.clip(A / physics["max_rusle"], 0.0, 1.0)


def calibrate_outputs(raw: np.ndarray,
                      reference: Optional[np.ndarray] = None,
                      mode: str = "climatology") -> np.ndarray:
    """
    Переводит сырой выход сети в индекс деградации 0..1.

    mode="climatology" (по умолчанию) — перцентиль декады внутри собственной
        климатической выборки этой же точки: 0 = самая спокойная декада
        выборки, 1 = самая эрозионно-напряжённая. Ранжирование целиком
        делает сеть; перцентиль лишь убирает неизвестный сдвиг/масштаб
        выхода (тот же приём, что в SPI для осадков).
    mode="sigmoid" — логистическое сжатие сырого выхода.
    mode="raw"     — выход сети без изменений (для отладки).
    """
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    if mode == "raw":
        return raw
    if mode == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -60, 60)))
    if mode != "climatology":
        raise ValueError("mode: 'climatology', 'sigmoid' или 'raw'.")

    ref = raw if reference is None else np.asarray(reference, dtype=np.float64).reshape(-1)
    ref = ref[np.isfinite(ref)]
    if ref.size < 5:                       # выборки не хватает — честный запасной путь
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -60, 60)))
    if np.allclose(ref.max(), ref.min()):
        return np.full(raw.shape, 0.5)
    ranks = np.searchsorted(np.sort(ref), raw, side="right") / ref.size
    return np.clip(ranks, 0.0, 1.0)


def predict_degradation_index(model, X: np.ndarray,
                              reference_X: Optional[np.ndarray] = None,
                              mode: str = "climatology") -> tuple:
    """
    Считает индекс деградации почвы для матрицы признаков `X` [N, 13].

    reference_X — климатическая выборка декад той же точки, по которой
    калибруется выход. Возвращает `(index 0..1, raw_output)`.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if X.shape[1] != INPUT_DIM:
        raise ValueError(f"Ожидалось {INPUT_DIM} признаков, получено {X.shape[1]}.")
    raw = np.asarray(model.forward(X), dtype=np.float64).reshape(-1)

    ref_raw = None
    if reference_X is not None and len(reference_X) > 0:
        ref = np.atleast_2d(np.asarray(reference_X, dtype=np.float64))
        ref_raw = np.asarray(model.forward(ref), dtype=np.float64).reshape(-1)

    return calibrate_outputs(raw, ref_raw, mode=mode), raw


# ===========================================================================
# 5. Дообучение (нужен torch)
# ===========================================================================
def physics_informed_loss(pred, batch_features, physics, lambda_phys: float = 0.1):
    """
    Физический residual: расхождение выхода сети с RUSLE-оценкой,
    нормированной на max_rusle (A = R*K*LS*C*P).
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("Для physics_informed_loss нужен torch.")
    R, K = batch_features[:, 0], batch_features[:, 1]
    LS, C, P = batch_features[:, 2], batch_features[:, 3], batch_features[:, 4]
    A_norm = torch.clamp(R * K * LS * C * P / physics["max_rusle"], 0, 1)
    return lambda_phys * torch.mean((pred.squeeze(-1) - A_norm) ** 2)


def finetune(model, physics, X, y, epochs: int = 200, lr: float = 1e-3,
             lambda_phys: float = 0.1, batch_size: int = 32):
    """
    Дообучение на фактических данных хозяйства (нужен torch).

    X: [N, 13] признаки в порядке INPUT_FEATURES,
    y: [N, 1] целевой индекс деградации 0..1 (зафиксированная эрозия и т.п.).
    Возвращает `(model, история_значений_loss)`.
    BatchNorm требует батч больше одного примера — это учтено.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("Для дообучения нужен пакет torch: pip install torch")
    module = model.module if isinstance(model, TorchAgroPINN) else model
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)
    if X.shape[0] < 2:
        raise ValueError("Для дообучения нужно минимум 2 примера (BatchNorm).")

    module.train()
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []
    for _ in range(epochs):
        perm = torch.randperm(X.shape[0])
        epoch_loss = 0.0
        for start in range(0, X.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            if len(idx) < 2:
                continue
            xb, yb = X[idx], y[idx]
            opt.zero_grad()
            pred = module(xb)
            loss = loss_fn(pred, yb) + physics_informed_loss(pred, xb, physics, lambda_phys)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
        history.append(epoch_loss)
    module.eval()
    return model, history


def save_checkpoint(model, physics: dict, path: str):
    """Сохраняет дообученную модель в формате исходного pinn_model.pt."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("Для сохранения чекпойнта нужен torch.")
    module = model.module if isinstance(model, TorchAgroPINN) else model
    payload = {"model_state": module.state_dict()}
    for key in PHYSICS_KEYS:
        payload[key] = physics[key]
    torch.save(payload, path)
    return path
