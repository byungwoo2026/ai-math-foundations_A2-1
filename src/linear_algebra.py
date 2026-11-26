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


# ============================================================
# 6. SVD 이미지 압축
# ============================================================
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114])   # 흑백 변환 가중치 (ITU-R BT.601: 사람 눈은 초록에 가장 민감)


def load_grayscale_image(path: str | None = None, size: int = 128) -> np.ndarray:
    """이미지를 읽어 size×size 흑백 배열(0~1 실수)로 만든다.

    처리 순서:
      ① 읽기: path가 None이면 matplotlib에 들어 있는 공개 도메인 사진 grace_hopper.jpg
         (미 해군 공식 초상, 600×512 컬러)를 쓴다. 따로 내려받을 필요가 없어 어디서나 재현된다.
      ② 0~1로: 정수(uint8, 0~255)면 255로 나눈다.
      ③ 흑백:  밝기 = 0.299·R + 0.587·G + 0.114·B        (H, W, 3) → (H, W)
      ④ 가운데 정사각형 자르기: 짧은 변 길이에 맞춘다. (600×512 → 512×512)
      ⑤ 블록 평균 축소: f×f 픽셀의 평균을 1픽셀로. (512 ÷ 128 = 4 → 4×4 평균)
            (s, s) → reshape (size, f, size, f) → 1·3번 축 평균 → (size, size)

    Args:
        path: 이미지 파일 경로. None이면 내장 사진.
        size: 결과 한 변 길이 (픽셀).

    Returns:
        shape (size, size), 값 범위 0~1인 float 배열.

    Raises:
        ValueError: 이미지가 size보다 작을 때 (확대는 하지 않는다).
    """
    import matplotlib.image as mpimg
    from matplotlib import cbook

    if path is None:
        with cbook.get_sample_data("grace_hopper.jpg") as f:
            img = mpimg.imread(f)
    else:
        img = mpimg.imread(path)

    img = img.astype(float)
    if img.max() > 1.0:                                  # ② 0~255 정수 → 0~1
        img = img / 255.0
    if img.ndim == 3:                                    # ③ 컬러 → 흑백 (알파 채널이 있으면 버림)
        img = img[..., :3] @ LUMA_WEIGHTS                # (H, W, 3) @ (3,) → (H, W)

    h, w = img.shape
    side = min(h, w)
    if side < size:
        raise ValueError(f"이미지({h}×{w})가 요청 크기 {size}보다 작습니다.")
    f = side // size                                     # 블록 한 변 (512 // 128 = 4)
    side = f * size                                      # 나누어떨어지게 맞춘 한 변
    top, left = (h - side) // 2, (w - side) // 2         # ④ 가운데 자르기
    img = img[top:top + side, left:left + side]
    return img.reshape(size, f, size, f).mean(axis=(1, 3))   # ⑤ 블록 평균 → (size, size)


def svd_compress(image: np.ndarray, k: int) -> dict:
    """SVD의 앞 k개 성분만으로 이미지를 복원한다.

    SVD:  A = U · Σ · Vᵀ = σ₁·u₁v₁ᵀ + σ₂·u₂v₂ᵀ + … + σ_r·u_rv_rᵀ      (σ₁ ≥ σ₂ ≥ … ≥ 0)
          각 σᵢ·uᵢvᵢᵀ는 "세로 패턴 uᵢ × 가로 패턴 vᵢ"로 만든 한 장의 층(rank-1 행렬).
    압축: A_k = σ₁·u₁v₁ᵀ + … + σ_k·u_kv_kᵀ = U[:, :k] · diag(σ[:k]) · Vᵀ[:k, :]
          shape: (m, k) · (k, k) · (k, n) → (m, n)
    저장할 숫자: 원본 m·n개  vs  압축 k·(m + n + 1)개 (u 열 k개 + v 행 k개 + σ k개)
    복원 오차:   ‖A − A_k‖_F / ‖A‖_F  (모든 픽셀 차이의 제곱합의 제곱근, 원본 대비)
               = √(버린 σᵢ²의 합 / 전체 σᵢ²의 합)       ← Eckart–Young 정리
    에너지 보존: 남긴 σᵢ²의 합 / 전체 σᵢ²의 합

    np.linalg.svd는 NumPy 함수라 허용된다 (과제가 금지한 것은 고유값 계산에 eig를 쓰는 것).

    Args:
        image: shape (m, n) 흑백 이미지.
        k: 남길 성분 개수. 최대 r = min(m, n)개까지만 있으므로 넘으면 r로 잘린다.

    Returns:
        dict
          - reconstruction: 복원 이미지 (m, n). 0~1 밖의 값이 조금 생길 수 있다 (그릴 때 잘라서 표시).
          - k_requested, k_used, rank_max: 요청 k, 실제 쓴 k, 최대 성분 수 r
          - original_numbers, stored_numbers, storage_ratio: 원본·압축 저장 숫자 개수와 비율
          - relative_error: ‖A − A_k‖_F / ‖A‖_F (직접 계산)
          - relative_error_from_sigma: √(버린 σ² 합 / 전체 σ² 합) (공식)
          - energy: 남긴 σ² 합 / 전체 σ² 합
          - singular_values: 전체 σ (r,)
    """
    A = np.asarray(image, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"흑백 이미지(2차원)만 받습니다. 받은 shape: {A.shape}")
    if k < 1:
        raise ValueError("k는 1 이상이어야 합니다.")
    m, n = A.shape

    U, s, Vt = np.linalg.svd(A, full_matrices=False)   # U (m, r), s (r,), Vt (r, n), r = min(m, n)
    r = len(s)
    k_used = min(k, r)

    A_k = (U[:, :k_used] * s[:k_used]) @ Vt[:k_used, :]  # U·diag(σ)는 U의 각 열에 σ를 곱한 것과 같다
    energy_total = np.sum(s ** 2)
    return {
        "reconstruction": A_k,
        "k_requested": k,
        "k_used": k_used,
        "rank_max": r,
        "original_numbers": m * n,
        "stored_numbers": k_used * (m + n + 1),
        "storage_ratio": k_used * (m + n + 1) / (m * n),
        "relative_error": float(np.linalg.norm(A - A_k) / np.linalg.norm(A)),
        "relative_error_from_sigma": float(np.sqrt(np.sum(s[k_used:] ** 2) / energy_total)),
        "energy": float(np.sum(s[:k_used] ** 2) / energy_total),
        "singular_values": s,
    }


def plot_svd_compression(image: np.ndarray, ks: tuple[int, ...] = (10, 50, 100)) -> plt.Figure:
    """원본과 k개 성분 복원 결과들을 한 Figure에 나란히 그린다 (과제 필수 그림).

    각 칸 제목에 저장 숫자 개수(원본 대비 %)와 복원 오차·에너지 보존율을 적는다.

    Args:
        image: shape (m, n) 흑백 이미지 (0~1).
        ks: 비교할 k 값들.

    Returns:
        Figure.
    """
    m, n = image.shape
    fig, axes = plt.subplots(1, len(ks) + 1, figsize=(4.0 * (len(ks) + 1), 4.9))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title(f"원본 {m}×{n}\n저장 숫자 {m * n:,}개", fontsize=11)

    for ax, k in zip(axes[1:], ks):
        res = svd_compress(image, k)
        ax.imshow(np.clip(res["reconstruction"], 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        note = " ← 원본보다 많음" if res["storage_ratio"] > 1 else ""
        ax.set_title(f"k = {res['k_used']}\n저장 {res['stored_numbers']:,}개 (원본의 {res['storage_ratio']:.1%}){note}\n"
                     f"오차 {res['relative_error']:.2%} · 에너지 {res['energy']:.3%}", fontsize=10.5)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("SVD 이미지 압축: 앞 k개 층만 남겨 복원", fontsize=13, y=1.03)
    fig.tight_layout()
    return fig


def plot_svd_layers(image: np.ndarray, n_layers: int = 5) -> plt.Figure:
    """SVD의 층 σᵢ·uᵢvᵢᵀ 하나하나와, 앞에서부터 쌓은 결과를 두 줄로 보여 준다.

    위 줄: i번째 층 하나 (양수 빨강·음수 파랑. 층에는 음수도 있다: 앞 층을 고치는 보정값)
    아래 줄: 1번 ~ i번 층을 더한 결과 = A_i

    Args:
        image: shape (m, n) 흑백 이미지.
        n_layers: 보여 줄 층 개수.

    Returns:
        Figure.
    """
    U, s, Vt = np.linalg.svd(np.asarray(image, dtype=float), full_matrices=False)
    fig, axes = plt.subplots(2, n_layers, figsize=(2.9 * n_layers, 6.4))
    running = np.zeros_like(image, dtype=float)
    for i in range(n_layers):
        layer = s[i] * np.outer(U[:, i], Vt[i, :])     # σᵢ · uᵢ vᵢᵀ: (m,) 바깥곱 (n,) → (m, n)
        running = running + layer
        lim = np.abs(layer).max()
        axes[0, i].imshow(layer, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        axes[0, i].set_title(f"{i + 1}번 층 (σ{i + 1} = {s[i]:.2f})", fontsize=10)
        axes[1, i].imshow(np.clip(running, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1, i].set_title(f"1~{i + 1}번 층 합 (k = {i + 1})", fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_ylabel("층 하나\n(빨강 +, 파랑 −)", fontsize=10)
    axes[1, 0].set_ylabel("앞에서부터 쌓기", fontsize=10)
    fig.suptitle("SVD = 중요한 순서로 쌓은 층: 층 하나는 '세로 패턴 u × 가로 패턴 v'", fontsize=13, y=1.0)
    fig.tight_layout()
    return fig


def plot_singular_values(image: np.ndarray, ks: tuple[int, ...] = (10, 50, 100)) -> plt.Figure:
    """특이값 크기와, k에 따른 복원 오차·저장 비율을 그린다.

    왼쪽: σᵢ (로그 눈금). 앞쪽 몇 개가 압도적으로 크고 뒤로 갈수록 작아진다 → 앞 k개만 남겨도 되는 이유.
    오른쪽: k마다 복원 오차(%)와 저장 비율(원본 대비 %).
            저장 비율 = k(m + n + 1) / (m·n) 이 100%를 넘는 k부터는 압축이 아니다.

    Args:
        image: shape (m, n) 흑백 이미지.
        ks: 표시할 k 값들.

    Returns:
        Figure.
    """
    from matplotlib.ticker import FuncFormatter

    A = np.asarray(image, dtype=float)
    m, n = A.shape
    s = np.linalg.svd(A, compute_uv=False)            # 특이값만 (r,)
    r = len(s)
    k_all = np.arange(1, r + 1)
    tail = np.sqrt(np.maximum(np.sum(s ** 2) - np.cumsum(s ** 2), 0) / np.sum(s ** 2))  # k개 남겼을 때 오차
    storage = k_all * (m + n + 1) / (m * n)
    breakeven = m * n / (m + n + 1)                     # 저장 비율이 100%가 되는 k

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    ax1.semilogy(k_all, s, color="#1D6B74", lw=1.8)
    for k in ks:
        ax1.axvline(min(k, r), color="#9A6514", lw=0.9, ls="--")
        ax1.text(min(k, r), s[0], f" k={k}", color="#9A6514", fontsize=9, va="top")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax1.set_title(f"특이값 σ: 1번 = {s[0]:.1f}, 10번 = {s[9]:.2f}, 마지막({r}번) = {s[-1]:.4f}", fontsize=11)
    ax1.set_xlabel("성분 번호 i")
    ax1.set_ylabel("특이값 σᵢ (로그 눈금)")
    ax1.grid(True, color="#D9E2E6", lw=0.6)

    ax2.plot(k_all, tail * 100, color="#C8423A", lw=1.8, label="복원 오차 ‖A − A_k‖ / ‖A‖")
    ax2.plot(k_all, storage * 100, color="#2F6DB5", lw=1.8, label="저장 숫자 (원본 대비)")
    ax2.axhline(100, color="#7A858B", lw=0.8)
    ax2.axvline(breakeven, color="#2F6DB5", lw=0.9, ls=":")
    ax2.text(breakeven, 190, f" k > {breakeven:.1f}이면\n 원본보다 많이 저장", color="#2F6DB5", fontsize=9, va="top")
    for k in ks:
        kk = min(k, r)
        ax2.plot(kk, tail[kk - 1] * 100, "o", color="#C8423A", ms=5)
        ax2.annotate(f"k={kk}: {tail[kk - 1]:.1%}", (kk, tail[kk - 1] * 100), textcoords="offset points",
                     xytext=(7, 3), fontsize=9, color="#C8423A")
    ax2.set_ylim(0, 205)
    ax2.set_title("k를 늘리면 오차는 줄고 저장 숫자는 늘어난다", fontsize=11)
    ax2.set_xlabel("남긴 성분 개수 k")
    ax2.set_ylabel("%")
    ax2.grid(True, color="#D9E2E6", lw=0.6)
    ax2.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    return fig
