"""선형대수 모듈: 행렬 변환 시각화 · Power Iteration · SVD 이미지 압축.

이 모듈에서 점 집합은 항상 shape (2, N) 배열로 다룬다.
  - 0번 행 = x 좌표, 1번 행 = y 좌표, 열 하나 = 점 하나
  - 그래서 N개의 점을 한꺼번에 변환하는 것이 행렬 곱 한 번이다.
        (2, 2) @ (2, N) → (2, N)

핵심 관찰: 행렬 A의 열은 기저벡터가 옮겨 가는 곳이다.
        A @ [1, 0] = A의 0번 열,   A @ [0, 1] = A의 1번 열
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# 1. 도형(점 집합) 만들기
# ============================================================
def unit_circle(n_points: int = 100) -> np.ndarray:
    """단위 원 위의 점 n_points개를 만든다.

    수식:  (x, y) = (cos θ, sin θ),   θ ∈ [0, 2π]

    np.linspace(0, 2π, 100)은 처음(θ=0)과 끝(θ=2π)이 같은 점이므로
    점 100개로 실제로는 변이 99개인 다각형(99각형)이 된다.

    Args:
        n_points: 점 개수. 과제 가이드대로 기본 100개.

    Returns:
        shape (2, n_points) 배열. 0번 행 x, 1번 행 y.
    """
    theta = np.linspace(0, 2 * np.pi, n_points)       # 각도 θ: (n_points,)
    return np.vstack([np.cos(theta), np.sin(theta)])  # 위아래로 쌓기 → (2, n_points)


def grid_lines(lim: float = 1.5, n_lines: int = 7, n_samples: int = 60) -> list[np.ndarray]:
    """np.meshgrid로 격자선(세로선 + 가로선) 점 집합을 만든다.

    변환 전후 격자를 같이 그리면 "공간 전체가 어떻게 휘는지"가 보인다.

    Args:
        lim: 격자 범위. x, y 모두 [-lim, lim].
        n_lines: 방향마다 그을 선 개수.
        n_samples: 선 하나를 이루는 점 개수 (변환 후에도 선이 매끄럽게 보이도록).

    Returns:
        선 하나당 shape (2, n_samples) 배열인 리스트 (세로선 n_lines개 + 가로선 n_lines개).
    """
    ticks = np.linspace(-lim, lim, n_lines)    # 선이 지나는 위치: (n_lines,)
    t = np.linspace(-lim, lim, n_samples)      # 선 위의 촘촘한 점: (n_samples,)
    X, T = np.meshgrid(ticks, t)               # 둘 다 (n_samples, n_lines). X의 j번 열 = ticks[j]로 고정
    vertical = [np.vstack([X[:, j], T[:, j]]) for j in range(n_lines)]    # x 고정 → 세로선
    horizontal = [np.vstack([T[:, j], X[:, j]]) for j in range(n_lines)]  # y 고정 → 가로선
    return vertical + horizontal


# ============================================================
# 2. 변환 행렬
# ============================================================
def rotation_matrix(theta: float) -> np.ndarray:
    """회전 행렬 R(θ): 원점을 중심으로 반시계 방향 θ(라디안)만큼 돌린다.

    수식:  R(θ) = [[cos θ, −sin θ],
                   [sin θ,  cos θ]]
           det R = cos²θ + sin²θ = 1   → 넓이가 변하지 않는다.

    Args:
        theta: 회전 각도 (라디안). 30°는 np.pi / 6.

    Returns:
        shape (2, 2) 배열.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])


def scaling_matrix(sx: float, sy: float) -> np.ndarray:
    """스케일링 행렬 S(sx, sy): x 방향으로 sx배, y 방향으로 sy배 늘인다.

    수식:  S(sx, sy) = [[sx, 0 ],
                        [0,  sy]]
           det S = sx · sy   → 넓이가 sx·sy배.  S(2, 0.5)면 2 × 0.5 = 1.

    Args:
        sx: x 방향 배율.
        sy: y 방향 배율.

    Returns:
        shape (2, 2) 배열.
    """
    return np.array([[sx, 0.0],
                     [0.0, sy]])


def shear_matrix(k: float) -> np.ndarray:
    """전단(shear) 행렬 Sh(k): 높이(y)에 비례해 x 방향으로 민다.

    수식:  Sh(k) = [[1, k],
                    [0, 1]]       (x, y) → (x + k·y, y)
           det Sh = 1·1 − k·0 = 1   → 기울어지지만 넓이는 그대로.
           (책 더미를 옆으로 밀어도 책 부피는 그대로인 것과 같다)

    Args:
        k: 미는 정도. y = 1인 점이 오른쪽으로 k만큼 이동한다.

    Returns:
        shape (2, 2) 배열.
    """
    return np.array([[1.0, k],
                     [0.0, 1.0]])


# ============================================================
# 3. 변환 적용과 넓이 계산
# ============================================================
def apply_transform(A: np.ndarray, points: np.ndarray) -> np.ndarray:
    """모든 점에 행렬 A를 곱한다 (행렬 곱 한 번에 N개 점을 동시에 변환).

    수식:  P' = A · P        shape: (2, 2) @ (2, N) → (2, N)

    Args:
        A: shape (2, 2) 변환 행렬.
        points: shape (2, N) 점 집합.

    Returns:
        변환된 점 집합, shape (2, N).

    Raises:
        ValueError: shape이 맞지 않을 때.
    """
    A = np.asarray(A, dtype=float)
    points = np.asarray(points, dtype=float)
    if A.shape != (2, 2) or points.ndim != 2 or points.shape[0] != 2:
        raise ValueError(f"A는 (2, 2), points는 (2, N)이어야 합니다. 받은 값: {A.shape}, {points.shape}")
    return A @ points


def det_2x2(A: np.ndarray) -> float:
    """2×2 행렬식(determinant)을 공식으로 직접 계산한다.

    수식:  A = [[a, b],
                [c, d]]   →   det A = a·d − b·c

    의미: 단위 정사각형(넓이 1)이 A로 옮겨진 평행사변형의 넓이(부호 포함).
          |det| = 넓이 배율, det < 0 이면 도형이 뒤집힌다(거울상).

    Args:
        A: shape (2, 2) 행렬.

    Returns:
        행렬식 값.
    """
    (a, b), (c, d) = np.asarray(A, dtype=float)
    return float(a * d - b * c)


def polygon_area(points: np.ndarray, signed: bool = False) -> float:
    """다각형 꼭짓점들로 넓이를 구한다 (신발끈 공식, shoelace formula).

    수식:  넓이 = ½ · Σᵢ (xᵢ · yᵢ₊₁ − xᵢ₊₁ · yᵢ)      (마지막 점 다음은 첫 점)

    꼭짓점이 반시계 방향으로 돌면 +, 시계 방향이면 − 가 나온다.
    그래서 부호까지 보면 "도형이 뒤집혔는지"도 알 수 있다.

    Args:
        points: shape (2, N) 꼭짓점 (순서대로 이어진 것).
        signed: True면 부호 있는 넓이, False면 절댓값.

    Returns:
        넓이.
    """
    x, y = np.asarray(points, dtype=float)
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)   # 한 칸씩 당김: i번째 자리에 i+1번째 점
    area = 0.5 * np.sum(x * y_next - x_next * y)
    return float(area if signed else abs(area))


def area_ratio_check(A: np.ndarray, points: np.ndarray | None = None, tol_pct: float = 1.0) -> dict:
    """변환 전후 넓이 비율을 재서 det(A)와 비교한다.

    수식:  넓이 비율 = 넓이(A·P) / 넓이(P)   ≟   det(A)
           오차(%)   = |넓이 비율 − det(A)| / |det(A)| × 100

    부호 있는 넓이로 비율을 구하므로 det가 음수(뒤집기)인 경우도 그대로 비교된다.
    참고로 원래 도형이 진짜 원이었다면 넓이는 π이므로, "변환 후 넓이 / π"도 함께 돌려준다.
    (점 100개 다각형은 원보다 약간 작아서 이 값은 det와 0.07%쯤 차이 난다)

    Args:
        A: shape (2, 2) 변환 행렬.
        points: 변환할 도형. None이면 단위 원(점 100개).
        tol_pct: 통과 기준 오차(%). 과제 기준 1%.

    Returns:
        det, area_before, area_after, ratio, error_pct, ratio_vs_pi, error_vs_pi_pct, passed를 담은 dict.
    """
    if points is None:
        points = unit_circle()
    det = det_2x2(A)
    area_before = polygon_area(points, signed=True)
    area_after = polygon_area(apply_transform(A, points), signed=True)
    ratio = area_after / area_before
    scale = abs(det) if det != 0 else 1.0                        # det가 0이면 나눌 수 없으니 절대오차로
    error_pct = abs(ratio - det) / scale * 100
    ratio_vs_pi = area_after / np.pi
    error_vs_pi_pct = abs(ratio_vs_pi - det) / scale * 100
    return {
        "det": det,
        "area_before": area_before,
        "area_after": area_after,
        "ratio": ratio,
        "error_pct": error_pct,
        "ratio_vs_pi": ratio_vs_pi,
        "error_vs_pi_pct": error_vs_pi_pct,
        "passed": bool(error_pct <= tol_pct),
    }


# ============================================================
# 4. 시각화
# ============================================================
def plot_transformation(A: np.ndarray, title: str, n_points: int = 100,
                        show_grid: bool = True, ax: plt.Axes | None = None) -> tuple[plt.Figure, dict]:
    """단위 원에 행렬 A를 적용하고 변환 전(파랑)·후(빨강)를 한 그림에 겹쳐 그린다.

    함께 그리는 것:
      - 격자: 회색(전) → 연한 빨강(후). 공간 전체가 어떻게 휘는지 보인다.
      - 기저벡터 화살표: e₁=[1,0] → A의 0번 열, e₂=[0,1] → A의 1번 열.
      - 표시점 8개: 원 위 8개 점이 어디로 옮겨 갔는지 (원을 회전하면 모양은 같아서
        표시점이 없으면 회전했는지 알 수 없다).
      - 오른쪽 아래 글상자: det, 넓이 비율, 오차.

    Args:
        A: shape (2, 2) 변환 행렬.
        title: 그림 제목.
        n_points: 원 위의 점 개수.
        show_grid: 격자를 그릴지.
        ax: 그릴 위치. None이면 새 Figure를 만든다.

    Returns:
        (Figure, area_ratio_check 결과 dict).
    """
    A = np.asarray(A, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 6.2))
    else:
        fig = ax.figure

    circle = unit_circle(n_points)
    moved = apply_transform(A, circle)
    info = area_ratio_check(A, circle)

    # (1) 격자: 변환 전후
    if show_grid:
        for line in grid_lines():
            ax.plot(*line, color="#B8C2C8", lw=0.6, zorder=1)
            ax.plot(*apply_transform(A, line), color="#F2B5B0", lw=0.6, zorder=1)

    # (2) 단위 원(전)과 변환 결과(후)
    ax.plot(*circle, color="#2F6DB5", lw=2.2, label="변환 전: 단위 원", zorder=3)
    ax.plot(*moved, color="#C8423A", lw=2.2, label="변환 후", zorder=4)

    # (3) 표시점 8개: 0°, 45°, …, 315° 위치. 빈 점(전) → 꽉 찬 점(후), 점선으로 이동 표시
    #     (원 위 100개 점 중에서 번호로 고르면 45° 간격이 정확히 맞지 않아서 각도로 따로 만든다)
    angles = np.arange(8) * np.pi / 4
    marks = np.vstack([np.cos(angles), np.sin(angles)])   # (2, 8)
    marks_moved = apply_transform(A, marks)                # (2, 8)
    for p, q in zip(marks.T, marks_moved.T):
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#7A858B", lw=0.8, ls=":", zorder=2)
    ax.scatter(*marks, s=36, facecolors="white", edgecolors="#2F6DB5", lw=1.4, zorder=5)
    ax.scatter(*marks_moved, s=36, color="#C8423A", zorder=6)

    # (4) 기저벡터: 전(회색 얇은 화살표) → 후(굵은 화살표). A의 열이 곧 도착점
    basis_colors = ["#1D6B74", "#9A6514"]
    for j, color in enumerate(basis_colors):
        e = np.eye(2)[:, j]                                # e₁ = [1, 0], e₂ = [0, 1]
        ax.annotate("", xy=e, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#7A858B", lw=1.2), zorder=7)
        ax.annotate("", xy=A[:, j], xytext=(0, 0),       # A @ e_j = A의 j번 열
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2), zorder=8)

    # (5) det와 넓이 비율 글상자
    text = (f"det(A) = {info['det']:.4f}\n"
            f"넓이: {info['area_before']:.4f} → {info['area_after']:.4f}\n"
            f"넓이 비율 = {info['ratio']:.4f}\n"
            f"오차 = {info['error_pct']:.4f}%  ({'통과' if info['passed'] else '실패'}, 기준 1%)")
    ax.text(0.98, 0.02, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#B8C2C8"), zorder=10)

    # (6) 축: 원과 변환 결과가 모두 보이도록 범위를 잡고, x·y 눈금 간격을 같게(원이 원으로 보이게)
    reach = max(1.5, np.abs(moved).max(), np.abs(A).max()) * 1.18
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal")
    ax.axhline(0, color="#56676F", lw=0.8, zorder=0)
    ax.axvline(0, color="#56676F", lw=0.8, zorder=0)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # (7) 범례는 그림 아래 바깥에 (곡선·화살표와 겹치지 않게)
    handles = [
        Line2D([], [], color="#2F6DB5", lw=2.2, label="변환 전: 단위 원"),
        Line2D([], [], color="#C8423A", lw=2.2, label="변환 후: A · (단위 원)"),
        Line2D([], [], ls="", marker="o", ms=6, mfc="white", mec="#2F6DB5",
               label="표시점 8개: ○ 전 → ● 후"),
        Line2D([], [], color="#B8C2C8", lw=1.2, label="격자: 회색 전 → 분홍 후"),
        Line2D([], [], color=basis_colors[0], lw=2.2,
               label=f"Ae₁ = A의 0번 열 = [{A[0, 0]:.2f}, {A[1, 0]:.2f}]"),
        Line2D([], [], color=basis_colors[1], lw=2.2,
               label=f"Ae₂ = A의 1번 열 = [{A[0, 1]:.2f}, {A[1, 1]:.2f}]"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=2, fontsize=9, frameon=False)
    return fig, info


# ============================================================
# 5. 고유값: Power Iteration
# ============================================================
def power_iteration(A: np.ndarray, num_iter: int = 1000, tol: float = 1e-6,
                    v0: np.ndarray | None = None) -> dict:
    """Power Iteration으로 절댓값이 가장 큰 고유값과 그 고유벡터를 구한다.

    고유값·고유벡터:  A · u = λ · u   (A를 곱해도 방향은 그대로, 길이만 λ배가 되는 u)

    원리: 아무 벡터 v₀를 고유벡터들로 나눠 쓰면 v₀ = c₁u₁ + c₂u₂ + … 이고,
          A를 k번 곱하면 각 성분이 λᵢ를 k번 곱한 만큼 커진다.
              Aᵏ v₀ = λ₁ᵏ · ( c₁u₁ + c₂ (λ₂/λ₁)ᵏ u₂ + … )
          |λ₂/λ₁| < 1 이므로 u₁ 말고 나머지 성분은 반복마다 그 비율만큼 줄어들고,
          v는 u₁ 방향으로 정렬된다. (비율이 1에 가까울수록 느리다)

    한 번 반복:
        w = A · v              ← A를 곱한다
        v ← w / ‖w‖            ← 길이를 1로 맞춘다 (안 하면 λ₁ᵏ 때문에 숫자가 폭발하거나 0이 된다)
        λ ← vᵀ · A · v          ← Rayleigh 몫. v가 고유벡터면 vᵀ(λv) = λ‖v‖² = λ

    멈춤 조건: 벡터 변화량 ‖v_new − v‖ < tol.
        λ < 0이면 v의 부호가 매번 뒤집히므로 min(‖v_new − v‖, ‖v_new + v‖)로 잰다.
        고유값 변화량으로 멈추지 않는 이유: 고유값은 벡터보다 약 2배 빨리 수렴해서
        A=[[4,1],[1,3]]이면 12회에 멈추고 벡터의 소수점 4번째 자리가 틀린다 (벡터 기준은 21회).

    Args:
        A: shape (n, n) 정사각 행렬.
        num_iter: 최대 반복 횟수.
        tol: 멈춤 기준 (벡터 변화량).
        v0: 시작 벡터. None이면 np.random.randn(n) (시드는 호출하는 쪽에서 고정).
            시작 벡터가 u₁과 정확히 직각(c₁ = 0)이면 u₁을 찾지 못하지만, 무작위 벡터는 거의 그럴 일이 없다.

    Returns:
        dict
          - eigenvalue: 고유값 λ₁ (Rayleigh 몫)
          - eigenvector: 단위 고유벡터 u₁ (절댓값이 가장 큰 성분이 양수가 되도록 부호 정리)
          - n_iter: 실제 반복 횟수
          - converged: tol 안에 들어왔는지
          - history_vectors: shape (n_iter + 1, n), v₀부터 매 반복의 v
          - history_eigenvalues: shape (n_iter + 1,), 매 반복의 λ 추정값
          - history_deltas: shape (n_iter,), 매 반복의 벡터 변화량

    Raises:
        ValueError: A가 정사각 행렬이 아니거나, v0가 영벡터이거나, A·v가 영벡터가 될 때.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A는 정사각 행렬이어야 합니다. 받은 shape: {A.shape}")
    n = A.shape[0]

    v = np.random.randn(n) if v0 is None else np.asarray(v0, dtype=float).copy()
    if np.linalg.norm(v) == 0:
        raise ValueError("시작 벡터가 영벡터입니다.")
    v = v / np.linalg.norm(v)                  # 길이 1로 시작
    lam = float(v @ A @ v)                     # 시작 벡터의 Rayleigh 몫

    vectors, eigenvalues, deltas = [v.copy()], [lam], []
    converged = False
    for _ in range(num_iter):
        w = A @ v                              # (n, n) @ (n,) → (n,)
        w_norm = np.linalg.norm(w)
        if w_norm == 0:
            raise ValueError("A·v가 영벡터가 되었습니다 (v가 고유값 0의 방향).")
        v_new = w / w_norm                     # 길이 1로 맞추기
        delta = min(np.linalg.norm(v_new - v), np.linalg.norm(v_new + v))  # 부호 뒤집힘은 같은 방향으로 본다
        v = v_new
        lam = float(v @ A @ v)                 # λ ≈ vᵀ A v
        vectors.append(v.copy())
        eigenvalues.append(lam)
        deltas.append(delta)
        if delta < tol:
            converged = True
            break

    v_out = v * np.sign(v[np.argmax(np.abs(v))])   # 부호 정리: 가장 큰 성분을 양수로
    return {
        "eigenvalue": lam,
        "eigenvector": v_out,
        "n_iter": len(deltas),
        "converged": converged,
        "history_vectors": np.array(vectors),
        "history_eigenvalues": np.array(eigenvalues),
        "history_deltas": np.array(deltas),
    }


def plot_power_iteration(A: np.ndarray, result: dict, n_show: int = 6, tol: float = 1e-6) -> plt.Figure:
    """2×2 행렬의 Power Iteration 과정을 세 칸 그림으로 보여 준다.

    ① 단위 원 확대: v₀, v₁, v₂ … 화살표가 최대 고유벡터 방향(점선)으로 모여 드는 모습.
    ② 고유값 추정값 λ_k = v_k · (A v_k) 가 λ₁으로 다가가는 모습.
    ③ 벡터 변화량 ‖v_k − v_(k−1)‖ (로그 눈금). 반복마다 거의 일정한 비율 |λ₂/λ₁|로 줄어
       멈춤 기준 tol 아래로 내려가면 멈춘다.
       2×2에서는 λ₁ + λ₂ = trace(A) (대각합)이므로 λ₂ = trace(A) − λ₁로 구한다.

    Args:
        A: shape (2, 2) 행렬.
        result: power_iteration(A)의 반환값.
        n_show: ①에 그릴 반복 화살표 개수 (v₀ 포함 n_show + 1개).
        tol: ③에 그릴 멈춤 기준선.

    Returns:
        Figure.
    """
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    A = np.asarray(A, dtype=float)
    if A.shape != (2, 2):
        raise ValueError("plot_power_iteration은 2×2 행렬만 그립니다.")
    lam1, u1 = result["eigenvalue"], result["eigenvector"]
    lam2 = float(np.trace(A)) - lam1                     # λ₁ + λ₂ = 대각합
    ratio = abs(lam2 / lam1)
    n_iter = result["n_iter"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.4),
                                        gridspec_kw={"width_ratios": [1, 1, 1.1]})

    # ---------- ① 화살표가 고유벡터 방향으로 모인다 (단위 원 확대) ----------
    circle = unit_circle(400)
    ax1.plot(*circle, color="#2F6DB5", lw=1.3, alpha=0.7)
    u1_angle = np.degrees(np.arctan2(u1[1], u1[0]))
    ax1.plot([0, u1[0] * 1.32], [0, u1[1] * 1.32], color="#15232B", lw=1.2, ls="--")
    ax1.text(*(u1 * 1.36), f"u₁ ({u1_angle:.1f}°)\n최대 고유벡터", fontsize=9, color="#15232B",
             ha="left", va="center")

    vecs = result["history_vectors"][: n_show + 1]
    colors = plt.cm.Greens(np.linspace(0.35, 1.0, len(vecs)))
    for k, (vec, color) in enumerate(zip(vecs, colors)):
        vec = vec * np.sign(vec @ u1) if vec @ u1 != 0 else vec   # λ < 0이면 부호가 뒤집히므로 같은 쪽으로
        ax1.annotate("", xy=vec, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))
        angle = np.degrees(np.arctan2(vec[1], vec[0]))
        if k <= 3:   # 뒤쪽 화살표는 u₁ 근처에 몰려 라벨이 겹치므로 앞 4개만 이름표
            ax1.text(*(vec * 1.08), f"v{k}  {angle:.1f}°", fontsize=9, color=color,
                     ha="left", va="center", fontweight="bold")
    last = vecs[-1] * np.sign(vecs[-1] @ u1)
    ax1.text(*(last * 0.55 + np.array([-0.05, 0.08])), f"v4 ~ v{len(vecs) - 1}", fontsize=8.5,
             color=colors[-1], ha="right", va="bottom")

    ax1.set_xlim(-0.35, 1.75)
    ax1.set_ylim(-0.6, 1.05)
    ax1.set_aspect("equal")
    ax1.axhline(0, color="#56676F", lw=0.6)
    ax1.axvline(0, color="#56676F", lw=0.6)
    ax1.set_title("① A 곱하기 → 길이 1로 맞추기를 반복하면\n화살표가 최대 고유벡터 방향으로 모인다", fontsize=11)
    ax1.set_xlabel("x  (파란 곡선 = 단위 원: v는 늘 길이 1)")
    ax1.set_ylabel("y")

    # ---------- ② 고유값 추정값이 λ₁으로 다가간다 ----------
    lams = result["history_eigenvalues"]
    ks_all = np.arange(len(lams))
    ax2.plot(ks_all, lams, "o-", color="#1D6B74", ms=4, lw=1.6, label="추정값 λ_k = v_k · (A v_k)")
    ax2.axhline(lam1, color="#C8423A", lw=1.2, ls="--", label=f"최종 λ₁ = {lam1:.4f}")
    for k in range(min(4, len(lams))):
        ax2.annotate(f"{lams[k]:.4f}", (k, lams[k]), textcoords="offset points", xytext=(8, -12),
                     fontsize=8.5, color="#56676F")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_title("② 고유값 추정값은 몇 번 만에\n거의 다 맞는다", fontsize=11)
    ax2.set_xlabel("반복 횟수 k")
    ax2.set_ylabel("고유값 추정값")
    ax2.grid(True, color="#D9E2E6", lw=0.6)
    ax2.legend(fontsize=9, loc="lower right")

    # ---------- ③ 변화량이 일정한 비율로 줄어든다 ----------
    deltas = result["history_deltas"]
    ks = np.arange(1, len(deltas) + 1)
    ax3.semilogy(ks, deltas, "o-", color="#1D6B74", ms=4, lw=1.6, label="벡터 변화량 ‖v_k − v_(k−1)‖")
    ax3.semilogy(ks, deltas[0] * ratio ** (ks - 1), ":", color="#9A6514", lw=1.8,
                 label=f"이론: 매번 |λ₂/λ₁| = {ratio:.4f}배")
    ax3.axhline(tol, color="#C8423A", lw=1.2, ls="--", label=f"멈춤 기준 tol = {tol:g}")
    ax3.axvline(n_iter, color="#7A858B", lw=0.8)
    ax3.text(n_iter - 0.4, deltas[0], f"{n_iter}회에 멈춤", fontsize=9, color="#56676F", ha="right", va="top")
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"1e{int(round(np.log10(y)))}"))
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.set_title(f"③ 변화량은 매번 약 {ratio:.2f}배로 줄어든다\n"
                  f"(λ₂ = trace(A) − λ₁ = {lam2:.4f})", fontsize=11)
    ax3.set_xlabel("반복 횟수 k")
    ax3.set_ylabel("벡터 변화량 (로그 눈금)")
    ax3.grid(True, which="major", color="#D9E2E6", lw=0.6)
    ax3.legend(fontsize=9, loc="lower left")

    fig.suptitle(f"Power Iteration: A = {A.tolist()}  →  λ₁ = {lam1:.4f}, "
                 f"u₁ = [{u1[0]:.4f}, {u1[1]:.4f}]  ({n_iter}회, tol = {tol:g})", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig
