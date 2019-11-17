import cv2
import numpy as np
from IPython.display import Image, display
from ipywidgets import widgets


def imshow(img):
    """画像を Notebook 上に表示する。
    """
    ret, encoded = cv2.imencode(".png", img)
    display(Image(encoded))


def canny(img, thresh, apertureSize, L2gradient):
    """2値化処理を行い、結果を表示する。
    """
    thresh1, thresh2 = thresh
    edges = cv2.Canny(
        img, thresh1, thresh2, apertureSize=apertureSize, L2gradient=L2gradient
    )
    imshow(edges)


# パラメータ「threshold1」「threshold2」を設定するスライダー
thresh_slider = widgets.SelectionRangeSlider(
    options=np.arange(1000), index=(100, 200), description=f"threshold"
)
thresh_slider.layout.width = "400px"

# パラメータ「apertureSize」を設定するスライダー
aperture_size_slider = slider = widgets.IntSlider(
    min=3, max=7, step=2, value=3, description="apertureSize: "
)
aperture_size_slider.layout.width = "400px"

# パラメータ「L2gradient」を設定するチェックボックス
l2_gradient_checkbox = widgets.Checkbox(value=False, description="L2gradient: ")
l2_gradient_checkbox.layout.width = "400px"

# 画像を読み込む。
img = cv2.imread("../RoadDamageDataset/All/JPEGImages/Numazu_20170906091504.jpg")
assert img.any()

# ウィジェットを表示する。
widgets.interactive(
    canny,
    img=widgets.fixed(img),
    thresh=thresh_slider,
    apertureSize=aperture_size_slider,
    L2gradient=l2_gradient_checkbox,
)
