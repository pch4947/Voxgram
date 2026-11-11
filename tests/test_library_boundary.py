"""Tests for tools/check_library_boundary.py."""

import importlib.util
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "check_library_boundary.py"
_spec = importlib.util.spec_from_file_location("check_library_boundary", _TOOL)
boundary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(boundary)


def messages(source: str) -> list[str]:
    return [message for _, message in boundary.check_source(source)]


def test_image_io_is_allowed():
    source = (
        "import cv2\n"
        "import numpy as np\n"
        "img = cv2.imread('card.png', cv2.IMREAD_GRAYSCALE)\n"
        "buf = cv2.imdecode(np.fromfile('card.png', dtype=np.uint8), cv2.IMREAD_COLOR)\n"
        "cv2.imwrite('out.png', img)\n"
    )
    assert messages(source) == []


def test_excluded_shortcuts_are_rejected():
    names = [
        "threshold",
        "findContours",
        "connectedComponents",
        "findHomography",
        "warpPerspective",
        "resize",
        "equalizeHist",
        "inpaint",
    ]
    for name in names:
        found = messages(f"import cv2\nx = cv2.{name}(img)\n")
        assert found == [f"cv2.{name} is outside the image I/O allowlist"], name


def test_aliased_import_is_tracked():
    found = messages("import cv2 as cv\nx = cv.warpPerspective(img, H, size)\n")
    assert found == ["cv2.warpPerspective is outside the image I/O allowlist"]


def test_from_import_is_checked():
    assert messages("from cv2 import imread\n") == []
    assert messages("from cv2 import resize\n") == ["cv2.resize is outside the image I/O allowlist"]


def test_forbidden_modules_are_rejected():
    assert messages("import skimage.measure\n") != []
    assert messages("from skimage import measure\n") != []
    assert messages("from scipy import ndimage\n") != []
    assert messages("import scipy.ndimage as ndi\n") != []


def test_unrelated_attributes_are_ignored():
    source = "import numpy as np\nclass T:\n    resize = 1\nt = T()\nx = t.resize\ny = np.resize\n"
    assert messages(source) == []
