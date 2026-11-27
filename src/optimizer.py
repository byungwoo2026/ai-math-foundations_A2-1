"""최적화 모듈: 경사하강법(Vanilla GD)과 Momentum, 수렴 경로 시각화.

역할 나누기
  - 옵티마이저 클래스: "지금 위치와 기울기를 받아 다음 위치를 돌려주는" 한 걸음(step)만 맡는다.
  - run_optimizer: 한 걸음을 n번 반복하며 지나간 위치를 모두 기록한다.
  - plot_*: 기록한 경로를 등고선 위에 점선으로, 또는 "원점까지 거리" 그래프로 그린다.

모양 약속 (다른 모듈과 같음): 점 하나는 shape (2,), 경로는 shape (2, 걸음 수 + 1).
→ path[:, k]가 k번째 걸음 뒤의 위치이고, ax.plot(*path)로 바로 그릴 수 있다.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

GD_COLOR = "#2F6DB5"        # Vanilla GD (파랑)
MOMENTUM_COLOR = "#C2571A"  # Momentum (주황)
INK = "#15232B"
MUTED = "#56676F"
GRID = "#D9E2E6"
OK_COLOR = "#2F7A46"        # 목표 반경 원

Function = Callable[[np.ndarray], np.ndarray]


# ============================================================
# 1. 시험용 함수 (최솟값은 둘 다 원점 (0, 0))
# ============================================================
def bowl(p: np.ndarray) -> np.ndarray:
    """둥근 그릇 f(x, y) = x² + y². 좌표를 첫 번째 축에 담은 배열을 받는다."""
    return p[0] ** 2 + p[1] ** 2


def bowl_grad(p: np.ndarray) -> np.ndarray:
    """∇f = (2x, 2y). 두 방향의 휘어짐(곡률)이 같다."""
    return np.stack([2 * p[0], 2 * p[1]])


def ellipse_bowl(p: np.ndarray) -> np.ndarray:
    """길쭉한 그릇 f(x, y) = x² + 10y². y 방향이 x 방향보다 10배 가파르다."""
    return p[0] ** 2 + 10 * p[1] ** 2


def ellipse_grad(p: np.ndarray) -> np.ndarray:
    """∇f = (2x, 20y)."""
    return np.stack([2 * p[0], 20 * p[1]])


# ============================================================
# 2. 옵티마이저
# ============================================================
class VanillaGD:
    """경사하강법: 기울기의 반대 방향으로 lr만큼 걷는다.

    수식:  x ← x − lr · ∇f(x)

    f = x² + y²이면 ∇f = 2x라서 x ← (1 − 2·lr) · x, 즉 매 걸음 같은 배율을 곱한다.
      |1 − 2·lr| < 1 (0 < lr < 1)이면 수렴, lr = 1이면 제자리 왕복, lr > 1이면 발산.
    f = x² + 10y²의 y 방향은 배율이 (1 − 20·lr)이라 lr < 0.1이어야 수렴한다.

    Attributes:
        lr: 학습률 (한 걸음의 크기).
    """

    name = "Vanilla GD"

    def __init__(self, lr: float = 0.1) -> None:
        self.lr = lr

    def reset(self) -> None:
        """기억하는 상태가 없으므로 할 일이 없다 (Momentum과 같은 모양으로 쓰기 위한 함수)."""

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """한 걸음 이동한 새 위치를 돌려준다.

        Args:
            x: 지금 위치, shape (2,).
            grad: 지금 위치의 기울기 ∇f(x), shape (2,).

        Returns:
            다음 위치, shape (2,).
        """
        return x - self.lr * grad


class Momentum:
    """Momentum: 지난 기울기들을 속도 v에 쌓아 두고, 관성으로 움직인다.

    수식 (PyTorch 방식):
        v ← β · v + ∇f(x)
        x ← x − lr · v

    같은 방향 기울기가 계속되면 v가 커져 가속되고 (기울기가 일정하면 최대 1 / (1 − β) = 10배),
    방향이 번갈아 바뀌면 서로 상쇄된다. 그래서 lr을 작게 잡아야 하는 평평한 방향에서 빨라진다.
    대신 관성 때문에 최솟값을 지나쳤다가 돌아오며 출렁인다.

    다른 교재의 표기 v ← β·v − lr·∇f,  x ← x + v 와는 v의 크기만 (−lr)배 다를 뿐 경로가 똑같다.

    Attributes:
        lr: 학습률.
        beta: 관성 계수 β (0이면 Vanilla GD와 같다).
        v: 속도. 첫 step 전에는 None.
    """

    name = "Momentum"

    def __init__(self, lr: float = 0.1, beta: float = 0.9) -> None:
        self.lr = lr
        self.beta = beta
        self.v: np.ndarray | None = None

    def reset(self) -> None:
        """속도를 지운다 (새로 출발할 때 이전 실행의 관성이 남지 않게)."""
        self.v = None

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """속도를 갱신하고 한 걸음 이동한 새 위치를 돌려준다.

        Args:
            x: 지금 위치, shape (2,).
            grad: 지금 위치의 기울기 ∇f(x), shape (2,).

        Returns:
            다음 위치, shape (2,).
        """
        if self.v is None:
            self.v = np.zeros_like(x)               # 첫 걸음: 정지 상태(속도 0)에서 출발
        self.v = self.beta * self.v + grad          # 이전 속도의 β배 + 이번 기울기
        return x - self.lr * self.v


# ============================================================
# 3. 반복 실행과 기록
# ============================================================
def run_optimizer(optimizer: VanillaGD | Momentum, grad_f: Function, start: tuple[float, float],
                  n_steps: int = 100, max_norm: float = 1e6, stop_below: float = 0.0) -> np.ndarray:
    """옵티마이저로 n_steps번 걸으며 지나간 위치를 모두 기록한다.

    Args:
        optimizer: VanillaGD 또는 Momentum 객체 (시작할 때 reset한다).
        grad_f: 기울기 함수. shape (2,) 위치를 받아 shape (2,) 기울기를 돌려준다.
        start: 출발점 (x, y).
        n_steps: 최대 걸음 수.
        max_norm: 원점에서 이보다 멀어지면 발산으로 보고 멈춘다 (계속 돌리면 숫자가 넘쳐 의미가 없다).
        stop_below: 원점까지 거리가 이보다 작아지면 다 왔다고 보고 멈춘다 (기본 0 = 끝까지 걷는다).
            lr을 수백 번 바꿔 가며 돌릴 때 시간을 줄이려고 쓴다.

    Returns:
        경로, shape (2, 실제 걸음 수 + 1). path[:, 0]이 출발점.
    """
    optimizer.reset()
    x = np.asarray(start, dtype=float)
    path = [x]
    for _ in range(n_steps):
        x = optimizer.step(x, grad_f(x))
        path.append(x)
        dist = np.linalg.norm(x)
        if dist > max_norm or dist < stop_below:
            break
    return np.array(path).T


def distances(path: np.ndarray) -> np.ndarray:
    """경로의 각 위치에서 최솟값(원점)까지의 거리, shape (걸음 수 + 1,)."""
    return np.linalg.norm(path, axis=0)


def steps_to_stay(path: np.ndarray, radius: float = 0.1) -> int | None:
    """원점 반경 radius 안에 들어가서 "끝까지 머무르기 시작한" 걸음 번호.

    Momentum은 관성 때문에 원을 그냥 통과해 지나가기도 해서, "처음 들어간 걸음"으로 세면
    실제보다 빨라 보인다. 그래서 그 뒤로 다시 나가지 않는 첫 걸음을 센다.

    Args:
        path: run_optimizer의 결과.
        radius: 목표 반경.

    Returns:
        걸음 번호. 마지막 위치가 반경 밖이면(아직 도착 못 함 또는 발산) None.
    """
    d = distances(path)
    if d[-1] > radius:
        return None
    outside = np.nonzero(d > radius)[0]
    return int(outside[-1]) + 1 if outside.size else 0


# ============================================================
# 4. 그림
# ============================================================
def _log_formatter() -> FuncFormatter:
    """로그 눈금을 "1e-6" 모양으로 (한글 폰트에서 마이너스가 깨지지 않게)."""
    return FuncFormatter(lambda v, _: f"1e{int(round(np.log10(v)))}")


def _minus(text: str) -> str:
    """숫자의 하이픈(-)을 수학 기호 마이너스(−)로 바꾼다 (범례·제목용)."""
    return text.replace("-", "−")


def _contours(ax: plt.Axes, f: Function, xlim: tuple[float, float], ylim: tuple[float, float],
              n_levels: int = 10) -> None:
    """회색 등고선 배경. 높이를 제곱 간격으로 잡아 등고선이 고른 간격으로 보이게 한다."""
    X, Y = np.meshgrid(np.linspace(*xlim, 300), np.linspace(*ylim, 300))
    Z = f(np.stack([X, Y]))
    top = np.sqrt(Z.max())
    levels = np.linspace(0, top, n_levels + 1)[1:] ** 2      # f = c 의 √c 가 같은 간격
    ax.contourf(X, Y, Z, levels=30, cmap="Greys", alpha=0.22)
    ax.contour(X, Y, Z, levels=levels, colors="#8A979E", linewidths=0.7)


def _limits(paths: list[np.ndarray], pad: float = 0.1) -> tuple[tuple[float, float], tuple[float, float]]:
    """경로들과 원점이 모두 들어가는 그림 범위."""
    pts = np.hstack(paths + [np.zeros((2, 1))])
    lo, hi = pts.min(axis=1), pts.max(axis=1)
    span = np.maximum(hi - lo, 1e-9)
    lo, hi = lo - pad * span, hi + pad * span
    return (lo[0], hi[0]), (lo[1], hi[1])


def _draw_path(ax: plt.Axes, path: np.ndarray, color: str, label: str, ms: float = 3.5) -> None:
    """경로를 점선 + 걸음마다 점으로 그린다."""
    ax.plot(*path, ls="--", lw=1.2, color=color, marker="o", ms=ms, label=label, zorder=3)


def plot_gd_path(f: Function, grad_f: Function, f_label: str, start: tuple[float, float] = (5.0, 5.0),
                 lr: float = 0.1, n_steps: int = 100, radius: float = 0.1) -> plt.Figure:
    """Vanilla GD 경로를 등고선 위에 점선으로 그린다 (왼쪽: 전체, 오른쪽: 원점 근처 확대).

    Args:
        f, grad_f: 함수와 기울기 함수.
        f_label: 제목에 쓸 함수 이름 (예: "f = x² + y²").
        start: 출발점.
        lr: 학습률.
        n_steps: 걸음 수.
        radius: 목표 반경 (확대 그림에 원으로 표시).

    Returns:
        Figure.
    """
    path = run_optimizer(VanillaGD(lr), grad_f, start, n_steps)
    d = distances(path)
    k_in = steps_to_stay(path, radius)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.3))

    # 왼쪽: 전체 모습
    lim = 1.3 * float(np.max(np.abs(start)))
    _contours(ax1, f, (-lim, lim), (-lim, lim))
    _draw_path(ax1, path, GD_COLOR, "Vanilla GD 경로 (점 하나 = 한 걸음)")
    ax1.scatter(*path[:, 0], marker="s", s=70, color=INK, zorder=5, label=f"출발점 {tuple(float(v) for v in start)}")
    for k in range(4):                                       # 처음 몇 걸음 번호
        ax1.annotate(f"{k}", path[:, k], xytext=(7, -12), textcoords="offset points", fontsize=9, color=INK)
    box = 0.13
    ax1.add_patch(plt.Rectangle((-box, -box), 2 * box, 2 * box, fill=False, ec=OK_COLOR, lw=1.2, zorder=4))
    ax1.annotate("오른쪽에서 확대", (box, -box), xytext=(0.9, -2.4), fontsize=9, color=OK_COLOR,
                 arrowprops=dict(arrowstyle="->", color=OK_COLOR, lw=1))
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_aspect("equal")
    ax1.set_title(f"{f_label}, lr = {lr:g}, {n_steps}회: 등고선을 직각으로 가로지르며 원점으로", fontsize=11)
    ax1.legend(loc="lower left", fontsize=9)

    # 오른쪽: 원점 근처 확대
    _contours(ax2, f, (-box, box), (-box, box))
    _draw_path(ax2, path, GD_COLOR, "Vanilla GD 경로", ms=5)
    ax2.add_patch(plt.Circle((0, 0), radius, fill=False, ec=OK_COLOR, lw=1.6, ls="-", zorder=4))
    ax2.text(-0.07, -0.106, f"목표: 원점 반경 {radius:g}", color=OK_COLOR, fontsize=9.5)
    for k in range(len(d)):
        if 0.045 < d[k] < box * 1.15:                         # 확대 그림 안에서 겹치지 않는 걸음만 번호
            ax2.annotate(f"{k}번째", path[:, k], xytext=(8, -4), textcoords="offset points", fontsize=9, color=INK)
    x_end, y_end = path[:, -1]
    ax2.text(-0.122, 0.118,
             f"{k_in}번째 걸음부터 반경 {radius:g} 안\n"
             f"{n_steps}회 후 위치 ({x_end:.1e}, {y_end:.1e})\n"
             f"원점까지 거리 {d[-1]:.1e}  → 과제 기준 통과",
             fontsize=9.5, va="top", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#B8C2C8"), zorder=6)
    ax2.set_xlim(-box, box)
    ax2.set_ylim(-box, box)
    ax2.set_aspect("equal")
    ax2.set_title("원점 근처 확대: 걸음이 점점 짧아진다 (매번 0.8배)", fontsize=11)
    for ax in (ax1, ax2):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.tight_layout()
    return fig


def plot_learning_rates(start: tuple[float, float] = (5.0, 5.0), n_steps: int = 40,
                        zigzag_start: tuple[float, float] = (5.0, 1.0)) -> plt.Figure:
    """학습률(lr)이 크면 발산한다는 것을 세 칸으로 보인다.

    (a) x² + y²: lr별 원점까지 거리. 한 걸음 배율 1 − 2·lr → 경계 lr = 1.
    (b) x² + 10y²: 가파른 y 방향 배율 1 − 20·lr → 경계 lr = 0.1 (lr = 0.5는 곧바로 발산).
    (c) x² + 10y² 경로: 경계 바로 아래(0.09)는 지그재그하며 수렴, 바로 위(0.11)는 지그재그가 커지며 발산.

    Args:
        start: (a), (b)의 출발점.
        n_steps: (a), (b)의 걸음 수.
        zigzag_start: (c)의 출발점 (지그재그가 한눈에 보이도록 y를 작게).

    Returns:
        Figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.9), gridspec_kw={"width_ratios": [1, 1, 0.85]})
    y_low, y_high = 1e-4, 1e5
    cases = [
        (axes[0], bowl_grad, 2, "배율",
         [(0.1, "#2F6DB5", "-"), (0.5, OK_COLOR, "-"), (0.9, "#6FA8DC", "--"),
          (1.0, MUTED, "-"), (1.05, "#E08A3C", "-"), (1.1, "#C8423A", "-")],
         "(a) f = x² + y²: 한 걸음 배율 (1 − 2·lr)\n→ |배율| < 1이면 수렴, 경계는 lr = 1"),
        (axes[1], ellipse_grad, 20, "y 배율",
         [(0.05, OK_COLOR, "-"), (0.09, "#2F6DB5", "-"), (0.1, MUTED, "-"),
          (0.11, "#E08A3C", "-"), (0.5, "#C8423A", "-")],
         "(b) f = x² + 10y²: 가파른 y 방향 배율 (1 − 20·lr)\n→ 경계가 lr = 0.1로 내려간다"),
    ]
    for ax, grad, curvature, word, settings, title in cases:
        for lr, color, ls in settings:
            path = run_optimizer(VanillaGD(lr), grad, start, n_steps)
            d = distances(path)
            factor = round(1 - curvature * lr, 3) + 0.0
            label = _minus(f"lr = {lr:g}  ({word} {factor:g})")
            zero = np.nonzero(d == 0)[0]
            if zero.size:                                     # 거리가 정확히 0 → 로그 눈금에 못 그림
                k0 = int(zero[0])
                ax.plot(np.arange(k0), d[:k0], ls=ls, color=color, lw=2, label=label + ": 1번 만에 원점")
                ax.plot([k0 - 1, k0], [d[k0 - 1], y_low * 1.6], ls=":", color=color, lw=2)
                ax.scatter([k0], [y_low * 1.6], marker="v", s=60, color=color, zorder=5)
            else:
                ax.plot(np.arange(d.size), d, ls=ls, color=color, lw=2, label=label)
        ax.set_yscale("log")
        ax.set_ylim(y_low, y_high)
        ax.set_xlim(0, n_steps)
        ax.yaxis.set_major_formatter(_log_formatter())
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_xlabel("반복 횟수")
        ax.set_ylabel("원점까지 거리 (로그 눈금)")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8.8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    axes[0].text(12, 2e-4, "0.1과 0.9는 |배율| = 0.8로 같아서 선이 겹친다\n(0.9는 원점을 사이에 두고 좌우로 튀며 수렴)",
                 fontsize=9, color=MUTED)
    axes[0].text(22, 2.2, "lr = 1: (5, 5) ↔ (−5, −5) 영원히 왕복", fontsize=9, color=MUTED)
    axes[1].text(9, 2e3, "lr = 0.5는 y가 5 → −45 → 405 → …", fontsize=9, color="#C8423A")

    # (c) 경로
    ax = axes[2]
    zig = {lr: run_optimizer(VanillaGD(lr), ellipse_grad, zigzag_start, 8) for lr in (0.09, 0.11)}
    xlim, ylim = (-0.4, 5.6), (-4.8, 4.8)
    _contours(ax, ellipse_bowl, xlim, ylim, n_levels=12)
    _draw_path(ax, zig[0.09], "#2F6DB5", "lr = 0.09: 지그재그하며 수렴", ms=4.5)
    _draw_path(ax, zig[0.11], "#E08A3C", "lr = 0.11: 지그재그가 커지며 발산", ms=4.5)
    ax.scatter(*zig[0.09][:, 0], marker="s", s=60, color=INK, zorder=5,
               label=f"출발점 {tuple(float(v) for v in zigzag_start)}")
    end = zig[0.11][:, -1]
    ax.annotate("8걸음 뒤", end, xytext=(1.3, 4.3), fontsize=9, color="#E08A3C",
                arrowprops=dict(arrowstyle="->", color="#E08A3C", lw=1))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("(c) f = x² + 10y² 경로 8걸음\n경계(lr = 0.1) 바로 아래와 바로 위", fontsize=11)
    ax.legend(fontsize=8.8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, frameon=False)
    fig.tight_layout()
    return fig


def plot_comparison(f: Function, grad_f: Function, f_label: str, start: tuple[float, float],
                    lrs: list[float], n_steps: int, beta: float = 0.9, radius: float = 0.1,
                    notes: list[str] | None = None) -> tuple[plt.Figure, dict]:
    """Vanilla GD와 Momentum을 같은 lr로 돌려 한 Figure에 겹쳐 그린다. lr 하나당 한 줄.

    왼쪽 칸: 등고선 위 두 경로 (점선). 오른쪽 칸: 원점까지 거리 (로그 눈금) + 반경 안에 머물기 시작한 걸음.

    Args:
        f, grad_f: 함수와 기울기 함수.
        f_label: 제목에 쓸 함수 이름.
        start: 출발점.
        lrs: 비교할 학습률들 (줄 수).
        n_steps: 걸음 수.
        beta: Momentum의 β.
        radius: 목표 반경.
        notes: 줄마다 제목 뒤에 붙일 설명 (없으면 생략).

    Returns:
        (Figure, dict) — dict[lr] = {"gd_stay", "mom_stay", "gd_end", "mom_end"} (end = 마지막 거리).
    """
    rows = len(lrs)
    fig, axes = plt.subplots(rows, 2, figsize=(13.5, 5.8 * rows), squeeze=False,
                             gridspec_kw={"width_ratios": [1.05, 1]})
    info = {}
    for r, lr in enumerate(lrs):
        gd = run_optimizer(VanillaGD(lr), grad_f, start, n_steps)
        mom = run_optimizer(Momentum(lr, beta), grad_f, start, n_steps)
        gd_stay, mom_stay = steps_to_stay(gd, radius), steps_to_stay(mom, radius)
        info[lr] = {"gd_stay": gd_stay, "mom_stay": mom_stay,
                    "gd_end": float(distances(gd)[-1]), "mom_end": float(distances(mom)[-1])}

        def stay_text(k: int | None) -> str:
            return f"{k}번째부터 반경 {radius:g} 안" if k is not None else f"{n_steps}회 안에 못 들어감"

        note = f" — {notes[r]}" if notes else ""
        ax_p, ax_d = axes[r]
        xlim, ylim = _limits([gd, mom], pad=0.08)
        _contours(ax_p, f, xlim, ylim, n_levels=12)
        _draw_path(ax_p, gd, GD_COLOR, f"Vanilla GD: {stay_text(gd_stay)}", ms=3)
        _draw_path(ax_p, mom, MOMENTUM_COLOR, f"Momentum (β = {beta:g}): {stay_text(mom_stay)}", ms=3)
        ax_p.scatter(*gd[:, 0], marker="s", s=60, color=INK, zorder=5, label=f"출발점 {tuple(float(v) for v in start)}")
        ax_p.scatter([0], [0], marker="+", s=120, color=INK, zorder=5)
        ax_p.set_xlim(*xlim)
        ax_p.set_ylim(*ylim)
        ax_p.set_aspect("equal")
        ax_p.set_xlabel("x")
        ax_p.set_ylabel("y")
        ax_p.set_title(f"{f_label}, lr = {lr:g}: 경로 {n_steps}회{note}", fontsize=11)
        ax_p.legend(fontsize=8.8, loc="best")

        ax_d.axhline(radius, color=OK_COLOR, lw=1.2, ls="--", label=f"목표 반경 {radius:g}")
        for path, color, name, k, height in ((gd, GD_COLOR, "Vanilla GD", gd_stay, 0.03),
                                             (mom, MOMENTUM_COLOR, "Momentum", mom_stay, 0.10)):
            d = distances(path)
            ax_d.plot(np.arange(d.size), np.maximum(d, 1e-300), color=color, lw=1.8, label=name)
            if k is not None:
                ax_d.axvline(k, color=color, lw=1, ls=":")
                right_side = k > 0.75 * n_steps                  # 오른쪽 끝이면 글자를 점선 왼쪽에
                ax_d.text(k + (-1 if right_side else 1), height, f"{name} {k}회",
                          transform=ax_d.get_xaxis_transform(),   # x는 데이터 값, y는 그림 아래에서부터 비율
                          ha="right" if right_side else "left", fontsize=9.5, color=color, fontweight="bold")
        ax_d.set_yscale("log")
        ax_d.yaxis.set_major_formatter(_log_formatter())
        ax_d.set_xlim(0, n_steps)
        ax_d.grid(True, color=GRID, lw=0.6)
        ax_d.set_xlabel("반복 횟수")
        ax_d.set_ylabel("원점까지 거리 (로그 눈금)")
        ax_d.set_title("원점까지 거리: 점선 = 반경 안에 머물기 시작한 걸음", fontsize=11)
        ax_d.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    return fig, info


def plot_lr_sweep(grad_f: Function, f_label: str, start: tuple[float, float], lrs: np.ndarray,
                  beta: float = 0.9, radius: float = 0.1, n_steps: int = 2000,
                  marks: tuple[float, ...] = ()) -> tuple[plt.Figure, dict]:
    """학습률을 바꿔 가며 "반경 안에 머물기 시작한 걸음 수"를 두 옵티마이저로 비교한다.

    Args:
        grad_f: 기울기 함수.
        f_label: 제목에 쓸 함수 이름.
        start: 출발점.
        lrs: 시험할 학습률들 (오름차순).
        beta: Momentum의 β.
        radius: 목표 반경.
        n_steps: lr마다 최대 걸음 수.
        marks: 세로 점선으로 표시할 lr (다른 그림에서 쓴 lr).

    Returns:
        (Figure, dict) — dict: lrs, gd_steps, mom_steps (도착 못 하면 NaN),
        gd_limit / mom_limit (처음으로 도착 못 한 lr = 발산 경계 근처),
        crossover (GD가 Momentum보다 빨라지기 시작하는 lr), theory (Momentum 이론 걸음 수).
    """
    lrs = np.asarray(lrs, dtype=float)
    gd_steps = np.full(lrs.size, np.nan)
    mom_steps = np.full(lrs.size, np.nan)
    for i, lr in enumerate(lrs):
        # 거리 1e-6 미만까지 오면 다시 반경 0.1 밖으로 나갈 일이 없으므로 거기서 멈춰 시간을 줄인다
        for steps, optimizer in ((gd_steps, VanillaGD(lr)), (mom_steps, Momentum(lr, beta))):
            k = steps_to_stay(run_optimizer(optimizer, grad_f, start, n_steps, stop_below=1e-6), radius)
            steps[i] = np.nan if k is None else k

    def first_fail(steps: np.ndarray) -> float:
        bad = np.nonzero(np.isnan(steps))[0]
        return float(lrs[bad[0]]) if bad.size else float("nan")

    gd_limit, mom_limit = first_fail(gd_steps), first_fail(mom_steps)
    faster = np.nonzero(gd_steps <= mom_steps)[0]            # NaN 비교는 False
    crossover = float(lrs[faster[0]]) if faster.size else float("nan")
    # Momentum이 출렁이며 줄어들 때 한 걸음 배율의 크기는 √β (lr과 무관) → 필요한 걸음 수
    d0 = float(np.linalg.norm(start))
    theory = np.log(radius / d0) / np.log(np.sqrt(beta))

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.plot(lrs, gd_steps, color=GD_COLOR, lw=2.2, label="Vanilla GD")
    ax.plot(lrs, mom_steps, color=MOMENTUM_COLOR, lw=2.2, label=f"Momentum (β = {beta:g})")
    ax.axhline(theory, color=MOMENTUM_COLOR, lw=1, ls=":")
    ax.text(lrs[-1], theory * 0.86, f"이론: 매 걸음 √{beta:g} = {np.sqrt(beta):.3f}배 → 약 {theory:.0f}회",
            color=MOMENTUM_COLOR, fontsize=9, ha="right", va="top")
    x_max = float(lrs[-1]) + 0.005
    ax.axvspan(gd_limit, x_max, color=GD_COLOR, alpha=0.07)
    ax.text(gd_limit + 0.002, 1500, f"GD 발산\n(lr ≥ {gd_limit:g})", color=GD_COLOR, fontsize=9.5, va="top")
    if np.isfinite(mom_limit):
        ax.axvspan(mom_limit, x_max, color=MOMENTUM_COLOR, alpha=0.12)
        ax.text(mom_limit - 0.002, 1500, f"Momentum은 lr < {mom_limit:g}까지\n수렴 → 발산 →", color=MOMENTUM_COLOR,
                fontsize=9.5, ha="right", va="top")
    ax.axvline(crossover, color=INK, lw=1, ls="--")
    ax.text(crossover + 0.002, 1500, f"← lr < {crossover:g}이면\n    Momentum이 빠름", color=INK, fontsize=9.5,
            ha="left", va="top")
    for m in marks:
        ax.axvline(m, color=MUTED, lw=0.9, ls=":")
        ax.text(m, 11, f" lr = {m:g}", color=MUTED, fontsize=8.5, rotation=90, va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(10, 2000)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(0, x_max)
    ax.grid(True, color=GRID, lw=0.6, which="both")
    ax.set_xlabel("학습률 lr")
    ax.set_ylabel(f"반경 {radius:g} 안에 머물기 시작한 걸음 수 (로그 눈금)")
    ax.set_title(f"{f_label}, 출발점 {tuple(float(v) for v in start)}: lr에 따라 누가 빠른가", fontsize=11.5)
    ax.legend(loc="lower right", fontsize=9.5)
    fig.tight_layout()
    return fig, {"lrs": lrs, "gd_steps": gd_steps, "mom_steps": mom_steps, "gd_limit": gd_limit,
                 "mom_limit": mom_limit, "crossover": crossover, "theory": float(theory)}
