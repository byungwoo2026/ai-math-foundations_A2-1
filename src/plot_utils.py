"""그림 공통 도구: 한글 폰트 설정과 outputs/ 폴더 저장.

모든 모듈과 노트북이 같은 방식으로 그림을 저장하도록 한곳에 모았다.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

# 이 파일(src/plot_utils.py)의 한 단계 위 = 레포 최상위 → 그 아래 outputs/
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

# 운영체제별로 흔히 설치된 한글 폰트 (앞에 있는 것부터 찾는다)
KOREAN_FONT_CANDIDATES = [
    "Malgun Gothic",        # Windows
    "AppleGothic",          # macOS
    "Apple SD Gothic Neo",  # macOS
    "NanumGothic",          # 리눅스 등에 따로 설치한 경우
    "Noto Sans CJK KR",
    "Noto Sans KR",
]


def setup_korean_font() -> str | None:
    """설치된 한글 폰트 중 첫 번째를 matplotlib 기본 폰트로 정한다.

    설치되지 않은 폰트 이름을 그냥 넘기면 matplotlib이 경고를 쏟아내므로,
    실제로 설치된 폰트 목록에서 골라서 하나만 지정한다.

    Returns:
        고른 폰트 이름. 한글 폰트가 하나도 없으면 None(그림의 한글이 네모로 보일 수 있음).
    """
    installed = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((name for name in KOREAN_FONT_CANDIDATES if name in installed), None)
    if chosen is not None:
        # 한글 폰트에 없는 글자(아래첨자 ₀, ‖ 등)는 matplotlib 기본 폰트 DejaVu Sans로 대신 그린다
        plt.rcParams["font.family"] = [chosen, "DejaVu Sans"]
    # 한글 폰트에는 수학용 마이너스 기호(−)가 없는 경우가 많아 일반 하이픈(-)으로 그린다
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def save_figure(fig: plt.Figure, filename: str, dpi: int = 150) -> Path:
    """그림을 outputs/ 폴더에 PNG로 저장한다.

    Args:
        fig: 저장할 matplotlib Figure.
        filename: 파일 이름 (예: "transform_rotation.png").
        dpi: 해상도. 150이면 README에 넣기 적당한 크기.

    Returns:
        저장한 파일의 전체 경로.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
