"""미적분 모듈: 수치 미분(중심차분법)과 Gradient 시각화.

여러 변수 함수는 좌표를 "첫 번째 축"에 담은 배열 p를 받도록 쓴다.
    def f(p):
        return p[0] ** 2 + p[1] ** 2      # f(x, y) = x² + y²
  - p shape (2,)        → 점 하나의 값 (스칼라)
  - p shape (2, N)      → 점 N개의 값 (N,)
  - p shape (2, H, W)   → 격자 전체의 값 (H, W)   ← 등고선 그릴 때
이렇게 하면 같은 함수를 점 하나에도, 격자 전체에도 반복문 없이 쓸 수 있다.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# 1. 수치 미분
# ============================================================
def numerical_derivative(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """중심차분법으로 1변수 함수의 미분값 f'(x)를 구한다.

    수식:  f'(x) ≈ ( f(x + h) − f(x − h) ) / (2h)

    왜 양옆을 보나: 테일러 전개
        f(x + h) = f(x) + f'(x)·h + f''(x)·h²/2 + f'''(x)·h³/6 + …
        f(x − h) = f(x) − f'(x)·h + f''(x)·h²/2 − f'''(x)·h³/6 + …
    두 식을 빼면 짝수 차수(f, f'')가 지워져서
        ( f(x + h) − f(x − h) ) / (2h) = f'(x) + f'''(x)·h²/6 + …
    → 오차가 h²에 비례한다 (h를 10배 줄이면 오차는 100배 준다).
       f(x) = x²는 f''' = 0이라 이론상 오차가 0이다.

    h를 너무 작게 하면 f(x + h)와 f(x − h)가 거의 같은 수가 되어, 컴퓨터의 반올림 오차
    (약 2.2e-16)가 2h로 나뉘며 커진다. 그래서 1e-5 정도가 적당하다 (실험상 최적 약 1.7e-5).

    Args:
        f: 미분할 함수 (실수 → 실수).
        x: 미분할 위치.
        h: 양옆으로 움직이는 거리.

    Returns:
        f'(x)의 근삿값.
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def forward_derivative(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """전방차분법 (비교용): 앞쪽 한 점만 본다.

    수식:  f'(x) ≈ ( f(x + h) − f(x) ) / h  =  f'(x) + f''(x)·h/2 + …
    → 오차가 h에 비례한다 (중심차분보다 한 차수 나쁘다).

    Args:
        f: 미분할 함수.
        x: 미분할 위치.
        h: 앞으로 움직이는 거리.

    Returns:
        f'(x)의 근삿값.
    """
    return (f(x + h) - f(x)) / h


def numerical_gradient(f: Callable[[np.ndarray], np.ndarray], point: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """여러 변수 함수의 Gradient ∇f를 중심차분으로 구한다.

    수식:  ∇f(p) = ( ∂f/∂x₁, ∂f/∂x₂, … )
           ∂f/∂xᵢ ≈ ( f(p + h·eᵢ) − f(p − h·eᵢ) ) / (2h)      eᵢ = i번 좌표만 1인 벡터
    "다른 좌표는 고정하고 i번 좌표만 살짝 움직였을 때의 기울기"를 좌표마다 구해 모은 것.

    Args:
        f: 다변수 함수. 좌표를 첫 번째 축에 담은 배열을 받는다 (모듈 설명 참고).
        point: shape (n,) 점 하나, 또는 shape (n, N, …) 여러 점.
        h: 좌표마다 움직이는 거리.

    Returns:
        point와 같은 shape. i번 줄이 ∂f/∂xᵢ.
    """
    p = np.asarray(point, dtype=float)
    grad = np.empty_like(p)
    for i in range(p.shape[0]):          # 좌표 개수만큼만 반복 (2변수면 2번)
        step = np.zeros_like(p)
        step[i] = h                      # i번 좌표만 h (여러 점이면 그 점들 모두의 i번 좌표)
        grad[i] = (f(p + step) - f(p - step)) / (2 * h)
    return grad


# ============================================================
# 2. 등고선 접선 (Gradient 없이 직접 찾기)
# ============================================================
def contour_tangent(f: Callable[[np.ndarray], np.ndarray], point: np.ndarray,
                    radius: float = 1e-3, n_samples: int = 3600) -> np.ndarray:
    """점을 지나는 등고선의 접선 방향을 Gradient를 쓰지 않고 직접 찾는다 (2변수 전용).

    방법: 점 p를 중심으로 아주 작은 원(반지름 radius)을 그리고, 그 원 위에서
          f 값이 f(p)와 같아지는 두 곳 q₁, q₂(= 등고선이 작은 원을 뚫고 지나가는 곳)를 찾는다.
          q₂ − q₁ 방향이 등고선의 접선이다.
    이렇게 구한 접선과 Gradient가 직각인지 재면, "Gradient ⟂ 등고선"을
    공식에 기대지 않고 숫자로 확인할 수 있다.

    Args:
        f: 2변수 함수.
        point: shape (2,) 점.
        radius: 작은 원의 반지름.
        n_samples: 작은 원 위에서 값을 재 볼 점 개수.

    Returns:
        shape (2,) 단위 접선 벡터.

    Raises:
        ValueError: 등고선과 만나는 곳을 두 곳 찾지 못할 때 (최솟값 점처럼 등고선이 없는 곳).
    """
    p = np.asarray(point, dtype=float)
    phi = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    ring = p[:, None] + radius * np.vstack([np.cos(phi), np.sin(phi)])   # 작은 원 위의 점들 (2, n)
    g = f(ring) - f(p)                                                    # 0이 되는 곳 = 등고선 위
    s = np.sign(g)
    crossings = np.where(s != np.roll(s, -1))[0]                          # i와 i+1 사이에서 부호가 바뀜
    if len(crossings) < 2:
        raise ValueError(f"점 {p}에서 등고선을 찾지 못했습니다.")
    found = []
    for i in crossings[:2]:
        j = (i + 1) % n_samples
        t = g[i] / (g[i] - g[j])                                          # 직선 보간: g가 0이 되는 비율
        found.append(ring[:, i] + t * (ring[:, j] - ring[:, i]))
    tangent = found[1] - found[0]
    return tangent / np.linalg.norm(tangent)


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """두 벡터 사이의 각도(도)를 구한다.

    수식:  cos θ = (u · v) / (‖u‖ ‖v‖)

    Args:
        u, v: 같은 길이의 벡터.

    Returns:
        0° ~ 180° 사이 각도.
    """
    cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


# ============================================================
# 3. 시각화
# ============================================================
def plot_derivative_error(x0: float = 3.0, h_used: float = 1e-5, tol: float = 1e-4) -> plt.Figure:
    """h를 바꿔 가며 전방차분·중심차분의 오차를 잰 그림 (로그-로그).

    왼쪽 f(x) = x²   (f' = 2x):  중심차분은 이론상 오차 0 → 반올림 오차만 보인다.
    오른쪽 f(x) = x³ (f' = 3x²): 중심차분 오차 = h² (정확히), 전방차분 오차 ≈ 9h.
    두 그림 모두 h가 너무 작으면 반올림 오차 때문에 다시 커진다.

    Args:
        x0: 미분할 위치.
        h_used: 실제로 쓰는 h (세로 점선으로 표시).
        tol: 과제 통과 기준 (가로 점선으로 표시).

    Returns:
        Figure.
    """
    from matplotlib.ticker import FuncFormatter

    hs = np.logspace(-12, -1, 221)
    cases = [
        ("f(x) = x²,  f'(3) = 6", lambda x: x ** 2, 2 * x0, None),
        ("f(x) = x³,  f'(3) = 27", lambda x: x ** 3, 3 * x0 ** 2, lambda h: h ** 2),
    ]
    fmt = FuncFormatter(lambda v, _: f"1e{int(round(np.log10(v)))}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
    for ax, (title, f, exact, theory) in zip(axes, cases):
        err_c = np.abs(numerical_derivative(f, x0, hs) - exact)
        err_f = np.abs(forward_derivative(f, x0, hs) - exact)
        ax.loglog(hs, np.maximum(err_f, 1e-17), color="#2F6DB5", lw=1.5, label="전방차분 (f(x+h) − f(x)) / h")
        ax.loglog(hs, np.maximum(err_c, 1e-17), color="#C8423A", lw=1.5, label="중심차분 (f(x+h) − f(x−h)) / 2h")
        if theory is not None:
            ax.loglog(hs, theory(hs), ":", color="#C8423A", lw=1.3, label="이론: 중심차분 오차 = h²")
        ax.axvline(h_used, color="#15232B", lw=0.9, ls="--")
        ax.text(h_used, 5, f" 사용한 h = {h_used:g}", fontsize=9, va="top")
        ax.axhline(tol, color="#9A6514", lw=0.9, ls="--")
        ax.text(2e-8, tol * 1.4, f"과제 기준 {tol:g}", fontsize=9, color="#9A6514", va="bottom")
        ax.text(1.3e-12, 5, "← h가 너무 작으면 반올림 오차가 커짐", fontsize=9, color="#56676F", va="top")
        ax.set_ylim(1e-17, 10)
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("h (양옆으로 움직이는 거리, 로그 눈금)")
        ax.set_ylabel("|수치 미분 − 정답| (로그 눈금)")
        ax.grid(True, which="major", color="#D9E2E6", lw=0.6)
        ax.legend(fontsize=8.5, loc="lower right")
    fig.suptitle(f"수치 미분 오차: x = {x0:g}에서 h를 바꿔 가며", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


def plot_gradient_field(f: Callable[[np.ndarray], np.ndarray], points: np.ndarray, title: str,
                        xlim: tuple[float, float], ylim: tuple[float, float],
                        arrow_len: float = 0.6, show_to_center: bool = False,
                        ax: plt.Axes | None = None) -> tuple[plt.Figure, dict]:
    """등고선 위 여러 점에서 Gradient 방향 화살표와 등고선 접선을 함께 그린다.

    그리는 것:
      - 등고선: 각 점의 f 값 높이로 그려서, 화살표가 모두 등고선 "위에서" 출발하게 한다.
      - 빨간 화살표: Gradient 방향 (길이는 같게 맞춤. 방향만 비교하기 위해)
      - 검은 짧은 선: 그 점의 등고선 접선 (contour_tangent로 직접 찾은 것)
      - 작은 직각 표시: 화살표와 접선이 만나는 모서리
      - (선택) 회색 점선: 점에서 최솟값(원점)으로 가는 방향 — 내리막(−∇f)과 비교용
    x·y 눈금 간격을 같게(aspect equal) 해야 직각이 직각으로 보인다.

    Args:
        f: 2변수 함수.
        points: shape (2, N) 화살표를 그릴 점들.
        title: 그림 제목.
        xlim, ylim: 그릴 범위.
        arrow_len: 화살표 길이 (데이터 단위).
        show_to_center: 원점으로 가는 점선을 그릴지.
        ax: 그릴 위치. None이면 새 Figure.

    Returns:
        (Figure, dict) — dict: points, gradients(수치), tangents, angles(도), max_angle_error(도)
    """
    points = np.asarray(points, dtype=float)
    if ax is None:
        width = 7.0
        height = width * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]) + 1.3
        fig, ax = plt.subplots(figsize=(width, height))
    else:
        fig = ax.figure

    # (1) 배경 등고선: 격자 전체에서 f 값을 한 번에 계산
    X, Y = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
    Z = f(np.stack([X, Y]))                                     # (2, 400, 400) → (400, 400)
    levels = np.unique(np.round(f(points), 6))                  # 점들이 놓인 높이
    ax.contourf(X, Y, Z, levels=30, cmap="Blues", alpha=0.35)
    cs = ax.contour(X, Y, Z, levels=levels, colors="#2F6DB5", linewidths=1.1)
    ax.clabel(cs, fmt=lambda v: f"f = {v:g}", fontsize=8.5, inline=True)

    # (2) 점마다: Gradient(수치) · 접선(직접 찾기) · 각도
    grads = numerical_gradient(f, points)                       # (2, N)
    tangents, angles = [], []
    for p, g in zip(points.T, grads.T):
        d = g / np.linalg.norm(g)                               # Gradient 방향 (길이 1)
        t = contour_tangent(f, p)                               # 등고선 접선 (길이 1)
        tangents.append(t)
        angles.append(angle_between(d, t))
        if show_to_center:
            ax.plot([p[0], 0], [p[1], 0], color="#7A858B", lw=0.8, ls=":", zorder=2)
        ax.plot(*np.column_stack([p - 0.28 * t, p + 0.28 * t]), color="#15232B", lw=1.6, zorder=4)
        s = 0.11                                                # 직각 표시 크기
        corner = np.column_stack([p + s * t, p + s * t + s * d, p + s * d])
        ax.plot(*corner, color="#15232B", lw=0.9, zorder=4)
        ax.annotate("", xy=p + arrow_len * d, xytext=p,
                    arrowprops=dict(arrowstyle="-|>", color="#C8423A", lw=1.8), zorder=5)
    ax.scatter(*points, s=14, color="#15232B", zorder=6)

    angles = np.array(angles)
    max_err = float(np.max(np.abs(angles - 90)))
    ax.text(0.02, 0.02,
            f"화살표와 접선 사이 각도 ({len(angles)}개 점)\n"
            f"평균 {angles.mean():.4f}°, 90°와 최대 차이 {max_err:.1e}°",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#B8C2C8"), zorder=7)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    handles =[Line2D([], [], color="#2F6DB5", lw=1.1, label="등고선 (f 값이 같은 점들)"),
               Line2D([], [], color="#C8423A", lw=1.8, label="Gradient ∇f 방향 (가장 가파른 오르막)"),
               Line2D([], [], color="#15232B", lw=1.6, label="등고선 접선 (직접 찾음) + 직각 표시")]
    if show_to_center:
        handles.append(Line2D([], [], color="#7A858B", lw=0.8, ls=":", label="최솟값(원점)으로 가는 방향"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
              fontsize=9, frameon=False)
    return fig, {"points": points, "gradients": grads, "tangents": np.array(tangents),
                 "angles": angles, "max_angle_error": max_err}


def points_on_levels(levels: list[float], counts: list[int], b: float = 1.0, offset_deg: float = 0.0) -> np.ndarray:
    """f(x, y) = x² + b·y² 의 등고선 f = c 위에 고르게 점을 놓는다.

    수식:  x = √c · cos θ,   y = √(c / b) · sin θ     →   x² + b·y² = c

    Args:
        levels: 등고선 높이 c들.
        counts: 높이마다 놓을 점 개수.
        b: y² 앞의 계수 (원이면 1, 타원 x² + 10y²면 10).
        offset_deg: 시작 각도 (높이마다 조금씩 돌려 화살표 겹침을 줄인다).

    Returns:
        shape (2, 점 개수 합) 배열.
    """
    pts = []
    for idx, (c, n) in enumerate(zip(levels, counts)):
        theta = np.radians(offset_deg * idx) + np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts.append(np.vstack([np.sqrt(c) * np.cos(theta), np.sqrt(c / b) * np.sin(theta)]))
    return np.hstack(pts)
