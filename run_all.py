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

from src import calculus as cal
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


# ============================================================
# Step 3. SVD 이미지 압축
# ============================================================
def run_step3() -> None:
    """흑백 이미지를 SVD로 압축·복원한 그림 3장을 저장하고, 계산이 맞는지 확인한다."""
    print("\n[Step 3] SVD 이미지 압축 (k = 10, 50, 100 비교)")
    image = la.load_grayscale_image(size=128)       # 내장 공개 도메인 사진 → 128×128 흑백
    print(f"  이미지: shape {image.shape}, 값 범위 {image.min():.3f} ~ {image.max():.3f}")

    ks = (10, 50, 100)
    for filename, fig in [("svd_compression.png", la.plot_svd_compression(image, ks)),
                          ("svd_layers.png", la.plot_svd_layers(image, n_layers=5)),
                          ("svd_singular_values.png", la.plot_singular_values(image, ks))]:
        save_figure(fig, filename)
        plt.close(fig)
        print(f"  그림 저장: outputs/{filename}")

    results = [la.svd_compress(image, k) for k in ks]
    for res in results:
        print(f"  k={res['k_used']:>3}: 저장 {res['stored_numbers']:>6,}개 ({res['storage_ratio']:6.1%}), "
              f"오차 {res['relative_error']:.4%}, 에너지 {res['energy']:.4%}")

    errors = [res["relative_error"] for res in results]
    check("k가 클수록 복원 오차 감소", all(a > b for a, b in zip(errors, errors[1:])),
          " > ".join(f"{e:.2%}" for e in errors))

    gap = max(abs(res["relative_error"] - res["relative_error_from_sigma"]) for res in results)
    check("복원 오차 = √(버린 σ² 합 / 전체 σ² 합) (Eckart–Young)", gap < 1e-10, f"최대 차이 {gap:.1e}")

    full = la.svd_compress(image, image.shape[0])
    check("모든 성분(k = 128)으로 복원하면 원본과 같음", full["relative_error"] < 1e-12,
          f"오차 {full['relative_error']:.1e}")

    # Step 2와 연결: σ₁² = AᵀA의 최대 고유값 → 직접 만든 Power Iteration으로 σ₁을 다시 구해 본다
    np.random.seed(42)
    sigma1_power = np.sqrt(la.power_iteration(image.T @ image, tol=1e-10)["eigenvalue"])
    sigma1_svd = results[0]["singular_values"][0]
    rel = abs(sigma1_power - sigma1_svd) / sigma1_svd
    check("σ₁ (np.linalg.svd) = √(Power Iteration으로 구한 AᵀA의 최대 고유값)", rel < 1e-6,
          f"{sigma1_svd:.6f} vs {sigma1_power:.6f}, 상대 오차 {rel:.1e}")


# ============================================================
# Step 4. 수치 미분과 Gradient
# ============================================================
def bowl(p: np.ndarray) -> np.ndarray:
    """f(x, y) = x² + y²  (좌표를 첫 번째 축에 담은 배열을 받는다)."""
    return p[0] ** 2 + p[1] ** 2


def ellipse_bowl(p: np.ndarray) -> np.ndarray:
    """f(x, y) = x² + 10y²  (Step 6에서 Momentum 비교에 쓰는 길쭉한 그릇)."""
    return p[0] ** 2 + 10 * p[1] ** 2


def run_step4() -> None:
    """수치 미분 오차를 확인하고, 등고선 + Gradient 화살표 그림을 저장한다."""
    print("\n[Step 4] 수치 미분(중심차분)과 Gradient")

    # (1) f(x) = x², x = 3 → 해석적 미분 2x = 6
    value = cal.numerical_derivative(lambda x: x ** 2, 3.0, h=1e-5)
    check("f(x)=x²의 x=3 수치 미분 vs 6", abs(value - 6) <= 1e-4,
          f"{value:.12f}, 오차 {abs(value - 6):.1e} (기준 1e-4)")
    forward = cal.forward_derivative(lambda x: x ** 2, 3.0, h=1e-5)
    tiny = cal.numerical_derivative(lambda x: x ** 2, 3.0, h=1e-12)
    print(f"    참고: 전방차분(h=1e-5) 오차 {abs(forward - 6):.1e}, "
          f"중심차분인데 h=1e-12로 너무 작으면 오차 {abs(tiny - 6):.1e} (기준 초과)")

    fig = cal.plot_derivative_error()
    save_figure(fig, "derivative_error.png")
    plt.close(fig)
    print("  그림 저장: outputs/derivative_error.png")

    # (2) f(x, y) = x² + y²: 등고선 f = 1, 4, 9 위 18개 점
    points = cal.points_on_levels([1, 4, 9], [4, 6, 8], b=1.0, offset_deg=22.5)
    fig, info = cal.plot_gradient_field(bowl, points, "f(x, y) = x² + y²: Gradient는 등고선과 직각",
                                        xlim=(-4.2, 4.2), ylim=(-4.2, 4.2), arrow_len=0.6)
    save_figure(fig, "gradient_field.png")
    plt.close(fig)
    print("  그림 저장: outputs/gradient_field.png")

    exact = 2 * points                                        # 해석적 Gradient (2x, 2y)
    grad_err = float(np.max(np.abs(info["gradients"] - exact)))
    check("수치 Gradient vs 해석적 (2x, 2y)", grad_err <= 1e-6, f"점 {points.shape[1]}개, 최대 오차 {grad_err:.1e}")
    check("Gradient ⟂ 등고선 (x² + y²)", info["max_angle_error"] <= 0.1,
          f"각도 평균 {info['angles'].mean():.4f}°, 90°와 최대 차이 {info['max_angle_error']:.1e}°")

    # (3) 미리보기: 길쭉한 그릇 x² + 10y² (Step 6)
    points_e = cal.points_on_levels([1, 4, 9], [4, 8, 10], b=10.0, offset_deg=15)
    fig, info_e = cal.plot_gradient_field(ellipse_bowl, points_e,
                                          "f(x, y) = x² + 10y²: 여기서도 직각이지만, 화살표가 중심을 향하지 않는다",
                                          xlim=(-4.2, 4.2), ylim=(-1.9, 1.9), arrow_len=0.55, show_to_center=True)
    save_figure(fig, "gradient_field_ellipse.png")
    plt.close(fig)
    print("  그림 저장: outputs/gradient_field_ellipse.png")
    check("Gradient ⟂ 등고선 (x² + 10y², 추가 확인)", info_e["max_angle_error"] <= 0.1,
          f"각도 평균 {info_e['angles'].mean():.4f}°, 90°와 최대 차이 {info_e['max_angle_error']:.1e}°")


def main() -> int:
    """모든 Step을 차례로 실행하고 통과 개수를 요약한다.

    Returns:
        종료 코드. 모두 통과하면 0, 하나라도 실패하면 1.
    """
    font = setup_korean_font()
    print(f"한글 폰트: {font or '없음 (그림의 한글이 깨질 수 있음)'}")

    run_step1()
    run_step2()
    run_step3()
    run_step4()

    n_pass = sum(passed for _, passed in RESULTS)
    print(f"\n전체: {n_pass}/{len(RESULTS)} 통과")
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
