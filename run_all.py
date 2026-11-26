"""모든 모듈을 실행해 outputs/에 그림(PNG)을 저장하고 과제 통과 기준을 확인한다.

실행 (레포 최상위 폴더에서):
    python run_all.py

출력: 항목마다 [통과]/[실패]와 근거 숫자, 마지막에 전체 통과 개수.
하나라도 실패하면 종료 코드 1로 끝난다.
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장 (터미널·서버에서도 동작)
import matplotlib.pyplot as plt
import numpy as np

from src import linear_algebra as la
from src.plot_utils import save_figure, setup_korean_font

np.random.seed(42)  # 재현성: 과제 규칙대로 시드 고정

RESULTS: list[tuple[str, bool]] = []  # (항목 이름, 통과 여부)


def check(name: str, passed: bool, detail: str) -> None:
    """검증 결과 한 줄을 출력하고 RESULTS에 기록한다.

    Args:
        name: 검증 항목 이름.
        passed: 통과 여부.
        detail: 근거 숫자 등 설명.
    """
    RESULTS.append((name, passed))
    print(f"  [{'통과' if passed else '실패'}] {name}: {detail}")


# ============================================================
# Step 1. 행렬 변환과 행렬식
# ============================================================
def run_step1() -> None:
    """회전·스케일링·전단 그림을 저장하고, det와 넓이 비율이 1% 이내로 맞는지 확인한다."""
    print("\n[Step 1] 행렬 변환과 행렬식 (det vs 넓이 비율, 기준 1%)")

    # 과제 필수 3종 + det ≠ 1 확인용 2종 (필수 3종은 det가 모두 1이라 검증이 쉬워지므로)
    cases = [
        ("회전 R(30°)", la.rotation_matrix(np.pi / 6), "transform_rotation.png"),
        ("스케일링 S(2, 0.5)", la.scaling_matrix(2, 0.5), "transform_scaling.png"),
        ("전단 Sh(1)", la.shear_matrix(1.0), "transform_shear.png"),
        ("일반 행렬 [[4,1],[1,3]]", np.array([[4.0, 1.0], [1.0, 3.0]]), "transform_general.png"),
        ("뒤집기 [[1,0],[0,-1]]", np.array([[1.0, 0.0], [0.0, -1.0]]), None),  # 그림 없이 숫자만
    ]
    for name, A, filename in cases:
        if filename is None:
            info = la.area_ratio_check(A)
        else:
            fig, info = la.plot_transformation(A, f"{name}: 단위 원 변환 전후")
            path = save_figure(fig, filename)
            plt.close(fig)
            print(f"  그림 저장: outputs/{path.name}")
        check(f"{name} det vs 넓이 비율", info["passed"],
              f"det={info['det']:.4f}, 넓이 비율={info['ratio']:.4f}, 오차={info['error_pct']:.2e}% "
              f"(원 넓이 π 기준이면 {info['ratio_vs_pi']:.4f}, 오차 {info['error_vs_pi_pct']:.3f}%)")


# ============================================================
# Step 2. Power Iteration
# ============================================================
def verify_with_eig(A: np.ndarray) -> tuple[float, np.ndarray]:
    """[검증 전용] np.linalg.eig로 절댓값이 가장 큰 고유값과 고유벡터를 구한다.

    과제 규칙: np.linalg.eig는 Power Iteration 결과와 비교하는 검증 코드에서만 쓴다.
    그래서 src/ 모듈이 아니라 이 실행 파일 안에만 둔다.

    Args:
        A: 정사각 행렬.

    Returns:
        (고유값, 단위 고유벡터). 고유벡터 부호는 power_iteration과 같게 가장 큰 성분을 양수로.
    """
    values, vectors = np.linalg.eig(A)
    i = int(np.argmax(np.abs(values)))          # Power Iteration이 찾는 것 = 절댓값 최대
    u = vectors[:, i].real
    u = u / np.linalg.norm(u)
    u = u * np.sign(u[np.argmax(np.abs(u))])
    return float(values[i].real), u


def run_step2() -> None:
    """Power Iteration 그림을 저장하고, 고유값이 np.linalg.eig와 5% 이내로 맞는지 확인한다."""
    print("\n[Step 2] Power Iteration vs np.linalg.eig (고유값 오차 5% 이내)")
    np.random.seed(42)  # 이 Step만 따로 돌려도 시작 벡터가 같도록 다시 고정

    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    main_result = la.power_iteration(A, tol=1e-6)
    fig = la.plot_power_iteration(A, main_result)
    path = save_figure(fig, "power_iteration.png")
    plt.close(fig)
    print(f"  그림 저장: outputs/{path.name}")

    B = np.random.randn(5, 5)
    cases = [
        ("[[4,1],[1,3]] (과제 예시, 대칭)", A, main_result),
        ("3×3 [[2,-1,0],[-1,2,-1],[0,-1,2]]", np.array([[2.0, -1, 0], [-1, 2, -1], [0, -1, 2]]), None),
        ("[[1,2],[3,4]] (비대칭)", np.array([[1.0, 2.0], [3.0, 4.0]]), None),
        ("5×5 무작위 대칭 (최대 고유값이 음수)", (B + B.T) / 2, None),
    ]
    for name, M, result in cases:
        if result is None:
            result = la.power_iteration(M, tol=1e-6)
        lam_eig, u_eig = verify_with_eig(M)
        err_pct = abs(result["eigenvalue"] - lam_eig) / abs(lam_eig) * 100
        cos = abs(float(result["eigenvector"] @ u_eig))   # 방향 일치도: 1이면 같은 방향
        print(f"  {name}")
        print(f"    Power Iteration (수렴 {result['n_iter']}회, tol=1e-6): "
              f"λ={result['eigenvalue']:.4f}, v={np.round(result['eigenvector'], 4).tolist()}")
        print(f"    np.linalg.eig  : λ={lam_eig:.4f}, v={np.round(u_eig, 4).tolist()}")
        check(f"{name} 고유값", result["converged"] and err_pct <= 5,
              f"오차 {err_pct:.2e}% (기준 5%), 고유벡터 방향 일치도 |cos| = {cos:.8f}")


def main() -> int:
    """모든 Step을 차례로 실행하고 통과 개수를 요약한다.

    Returns:
        종료 코드. 모두 통과하면 0, 하나라도 실패하면 1.
    """
    font = setup_korean_font()
    print(f"한글 폰트: {font or '없음 (그림의 한글이 깨질 수 있음)'}")

    run_step1()
    run_step2()

    n_pass = sum(passed for _, passed in RESULTS)
    print(f"\n전체: {n_pass}/{len(RESULTS)} 통과")
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
