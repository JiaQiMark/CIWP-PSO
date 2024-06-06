import numpy as np
from PIL import Image
# from skimage import color


def get_kapur_entropy(thresholds, image):
    thresholds = np.round(thresholds).astype(int)
    thresholds.sort()
    if np.count_nonzero(thresholds == 0) > 0:
        return 0
    gray_value_max = np.amax(image)
    # 计算直方图
    hist, _ = np.histogram(image, bins=gray_value_max, range=(0, gray_value_max))
    # 每个灰度值的概率分布
    p_gray = hist.astype(float) / hist.sum()

    # 初始化变量
    kapur_entropy = 0

    m = len(thresholds)
    for i in range(0, m+1):
        start_t = 0 if i == 0 else thresholds[i-1]
        end_t = gray_value_max if i == m else (thresholds[i] - 1)

        W = p_gray[start_t:end_t].sum()
        e = 1e-20
        if W == 0:
            W = e
        arr = ((p_gray[start_t:end_t] + e) / W) * np.log((p_gray[start_t:end_t] + e) / W)
        arr_no_nan = np.nan_to_num(arr, nan=0)
        H = -np.sum(arr_no_nan)
        kapur_entropy += H

    return kapur_entropy


def get_cross_entropy(original_image, segmented_image):
    # 将图像像素值映射到[0, 1]范围
    original_image = original_image / np.max(original_image)
    segmented_image = segmented_image / np.max(segmented_image)

    # 防止log(0)的情况，将图像中的0替换为一个很小的值
    epsilon = 1e-10
    original_image = np.clip(original_image, epsilon, 1 - epsilon)
    segmented_image = np.clip(segmented_image, epsilon, 1 - epsilon)

    # 计算交叉熵
    cross_entropy_value = -np.sum(original_image * np.log(segmented_image) +
                                  (1 - original_image) * np.log(1 - segmented_image))

    return cross_entropy_value


















# tiff_path = "D:\\003__program\\00__DATASET\\01_img_Seg\\SIPI_USC\\Male.tiff"
# tiff_img = Image.open(tiff_path)
# tiff_img_array = np.array(tiff_img)
#
# ke = get_kapur_entropy([0, 0, 0, 0, 0, 0, 102, 0], tiff_img_array)
