import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# 设置全局字体样式和大小
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['font.size'] = 10


def multi_threshold_segmentation(img, th):
    segmented = np.zeros_like(img)
    gray_value_max = np.amax(img)
    th.sort()

    m = len(th)
    for t in range(m + 1):
        start_t = 0 if t == 0 else th[t - 1]
        end_t = gray_value_max if t == m else th[t] - 1

        # 将灰度值在 [i * 10, (i+1) * 10 - 1] 范围内的像素点设置为对应的值
        segmented[(img >= start_t) & (img < end_t)] = start_t

    return segmented


def image_show(des, img):
    cv2.imshow(des, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# tiff_path = "D:\\003__program\\00__DATASET\\01_img_Seg\\SIPI_USC\\Male.tiff"
# tiff_img = Image.open(tiff_path)
# tiff_img_array = np.array(tiff_img)
# image_show(tiff_path, tiff_img_array)

def algorithm_result(b_or_t, alg_name, axles, p):
    exp_list = ['bench', 'tumer']
    exp_name = exp_list[b_or_t]
    thresholds_path = f"result//result_{alg_name}//{exp_name}//{alg_name}_{exp_name}-{i}_thresholds_4.npy"
    convergence_path = f"result//result_{alg_name}//{exp_name}//{alg_name}_{exp_name}-{i}_convergence_curve_4.npy"
    thresholds = np.load(thresholds_path)
    thresholded_image = multi_threshold_segmentation(image, thresholds)
    axles[1, p].imshow(thresholded_image, cmap='gray')
    axles[1, p].set_title(f'{alg_name} \n\n T{thresholds.shape[0]}')
    curve = np.load(convergence_path)
    axles[1, p + 1].plot(curve)
    axles[1, p + 1].set_title(f'{np.round(np.max(curve), decimals=4)}')

    thresholds_path = f"result//result_{alg_name}//{exp_name}//{alg_name}_{exp_name}-{i}_thresholds_6.npy"
    convergence_path = f"result//result_{alg_name}//{exp_name}//{alg_name}_{exp_name}-{i}_convergence_curve_6.npy"
    thresholds = np.load(thresholds_path)
    thresholded_image = multi_threshold_segmentation(image, thresholds)
    axles[2, p].imshow(thresholded_image, cmap='gray')
    axles[2, p].set_title(f'T{thresholds.shape[0]}')
    curve = np.load(convergence_path)
    axles[2, p + 1].plot(curve)
    axles[2, p + 1].set_title(f'{np.round(np.max(curve), decimals=4)}')

    thresholds_path = f"result//result_{alg_name}//{exp_name}//{alg_name}_{exp_name}-{i}_thresholds_8.npy"
    convergence_path = f"result//result_{alg_name}//{exp_name}//{alg_name}_{exp_name}-{i}_convergence_curve_8.npy"
    thresholds = np.load(thresholds_path)
    thresholded_image = multi_threshold_segmentation(image, thresholds)
    axles[3, p].imshow(thresholded_image, cmap='gray')
    axles[3, p].set_title(f'T{thresholds.shape[0]}')
    curve = np.load(convergence_path)
    axles[3, p + 1].plot(curve)
    axles[3, p + 1].set_title(f'{np.round(np.max(curve), decimals=4)}')


bench_or_tumer = 0
for i in range(1, 10):
    # algname_list = ['BES', 'GWO', 'WOA', 'SSA', 'PSO', 'GA', 'FA', 'WOAmjq', 'PSOsa', 'PSOssa']
    algname_list = ['PSO', 'PSOsa', 'PSOssa', 'PSOmjq']
    # 创建一个包含两个子图的图形
    fig, axes = plt.subplots(4, 2 * len(algname_list), figsize=(18, 6))

    if bench_or_tumer == 0:
        np_data_paht = f"D:\\000__program\\00__dataset\\01_img_Seg\\04__BSDS300\\selected_gray_scaled\\{i}.npy"
        image = np.load(np_data_paht)
    elif bench_or_tumer == 1:
        matlab_data_path = f"D:\\000__program\\00__dataset\\01_img_Seg\\01__brainTumor_1_766\\{i}.mat"
        with h5py.File(matlab_data_path, 'r') as file:
            variables = list(file.keys())
            matlab_data = file['cjdata']
            print(type(matlab_data))

            label = np.array(matlab_data['label'])
            image = np.array(matlab_data['image'])
            patient_id = np.array(matlab_data['PID'])
            tumor_border = np.array(matlab_data['tumorBorder'])
            tumor_mask = np.array(matlab_data['tumorMask'])

            # cvu8_image = cv2.convertScaleAbs(image)
            cvu8_image = image.astype(np.uint8)
            # equailization_image = cv2.equalizeHist(cvu8u_image)

            bit8_image = ((image / 4095) * 255).astype(int)

        axes[0, 1].set_title('mask')
        axes[0, 1].imshow(tumor_mask, cmap='gray')

        axes[0, 2].set_title('cvu8')
        axes[0, 2].imshow(cvu8_image, cmap='gray')

        axes[0, 3].set_title('8 bit')
        axes[0, 3].imshow(cvu8_image, cmap='gray')

    # 在第一个子图中显示图像1
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title(f'Original Image {i}')
    axes[0, 0].axis('off')


    for an in range(len(algname_list)):
        algname = algname_list[an]
        algorithm_result(bench_or_tumer, algname, axes, 2 * an)


    # 调整子图之间的间距
    plt.tight_layout()
    plt.show()

