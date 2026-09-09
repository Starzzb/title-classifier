import numpy as np
import cv2
from title_classifier.utils.image import stitch_images
from title_classifier.core.scene_detector import compute_frame_change_score

def test_stitch_images_basic():
    # 模拟 5 张正方形图片
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
    timestamps = [0.0, 1.0, 2.0, 3.0, 4.0]
    
    # 拼接为 2x2 网格
    stitched = stitch_images(
        frames,
        grid_shape=(2, 2),
        sub_max_size=50,
        draw_labels=True,
        timestamps=timestamps
    )
    
    # 5张图用 2x2（容量 4）网格拼接，应当生成 2 张网格大图
    assert len(stitched) == 2
    # 每个子图最长边等比例降为 50 像素。2x2 拼接后大图应当是 100x100
    assert stitched[0].shape == (100, 100, 3)
    assert stitched[1].shape == (100, 100, 3)

def test_compute_frame_change_score():
    # 相同图片变化率应当极近于 0（因 HSV/灰度 结构完全相同）
    img1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
    score_same = compute_frame_change_score(img1, img1)
    assert score_same == 0.0
    
    # 截然不同的图片变化率应当大于 0
    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    score_diff = compute_frame_change_score(img1, img2)
    assert score_diff > 0.1
