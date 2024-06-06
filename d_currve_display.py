import numpy as np
from tqdm import tqdm
import matplotlib


def draw(exp_id, img_id, nth,
         gwo, woa, ssa, pso, ga, fa, psosa, psossa, psomjq):
    import matplotlib.pyplot as plt
    # 设置全局字体样式和大小
    # w_cm = 3.6
    # h_cm = 2.8
    # plt.figure(figsize=(w_cm / 2.54, h_cm / 2.54))
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Times New Roman']
    matplotlib.rcParams['font.size'] = 18

    expname_list = ['bench', 'tumer']
    algname_list = ['GWO', 'WOA', 'SSA', 'PSO', 'GA', 'FA', 'PSOsa', 'PSOssa', 'PSOmjq']
    exp = expname_list[exp_id]

    if gwo:
        algname = algname_list[0]
        curve_path = f"result//result_{algname}//{exp}//{algname}_{exp}-{img_id}_convergence_curve_{nth}.npy"
        curve_gwo = np.load(curve_path)
    if woa:
        algname = algname_list[1]
        curve_path = f"result//result_{algname}//{exp}//{algname}_{exp}-{img_id}_convergence_curve_{nth}.npy"
        curve_woa = np.load(curve_path)
    if ssa:
        algname = algname_list[2]
        curve_path = f"result//result_{algname}//{exp}//{algname}_{exp}-{img_id}_convergence_curve_{nth}.npy"
        curve_ssa = np.load(curve_path)
    if pso:
        algname = algname_list[3]
        curve_path = f"result//result_{algname}//{exp}//{algname}_{exp}-{img_id}_convergence_curve_{nth}.npy"
        curve_pso = np.load(curve_path)
    if ga:
        algname = algname_list[4]
        curve_path = f"result//result_{algname}//{exp}//{algname}_{exp}-{img_id}_convergence_curve_{nth}.npy"
        curve_ga = np.load(curve_path)
    if fa:
        algname = algname_list[5]
        curve_path = f"result//result_{algname}//{exp}//{algname}_{exp}-{img_id}_convergence_curve_{nth}.npy"
        curve_fa = np.load(curve_path)
    if psosa:
        algname = algname_list[6]
        curve_path = f"result//result_{algname}//{exp}//{algname}_{exp}-{img_id}_convergence_curve_{nth}.npy"
        curve_psosa = np.load(curve_path)
    if psossa:
        algname = algname_list[7]
        curve_path = f"result//result_{algname}//{exp}//{algname}_{exp}-{img_id}_convergence_curve_{nth}.npy"
        curve_psossa = np.load(curve_path)
    if psomjq:
        algname = algname_list[8]
        curve_path = f"result//result_{algname}//{exp}//{algname}_{exp}-{img_id}_convergence_curve_{nth}.npy"
        curve_psomjq = np.load(curve_path)

    if gwo:
        plt.plot(curve_gwo, label='GWO', linewidth=0.5, color='cyan', linestyle='--')
    if woa:
        plt.plot(curve_woa, label='WOA', linewidth=0.3, color='black')
    if ssa:
        plt.plot(curve_ssa, label='SSA', linewidth=0.5, color='red', linestyle='--')
    if pso:
        plt.plot(curve_pso, label='PSO', linewidth=0.5, color='green')
    if ga:
        plt.plot(curve_ga, label='GA', linewidth=0.5, color='yellow', linestyle='--')
    if fa:
        plt.plot(curve_fa, label='FA', linewidth=0.5, color='black', linestyle='--')
    if psosa:
        plt.plot(curve_psosa, label='PSOSA', linewidth=0.5, color='blue', linestyle='--',
                 marker='o', markevery=20, markersize=3)
    if psossa:
        plt.plot(curve_psossa, label='PSOH', linewidth=0.5, color='blue', linestyle='--',
                 marker='^', markevery=20, markersize=3)
    if psomjq:
        plt.plot(curve_psomjq, label='CIWP-PSO', linewidth=0.5, color='blue', linestyle='--',
                 marker='s', markevery=12, markersize=5)

    plt.xlabel('Iteration number')
    plt.ylabel('Kapur entropy')
    plt.legend()
    # plt.title(f"Image {img_id}  nTh={nth}")
    # 调整 x 轴和 y 轴的线宽
    # 获取当前的轴对象
    ax = plt.gca()
    v = 0.2
    ax.spines['bottom'].set_linewidth(v)  # x 轴线宽
    ax.spines['left'].set_linewidth(v)    # y 轴线宽
    ax.spines['top'].set_linewidth(v)     # 上边界线宽
    ax.spines['right'].set_linewidth(v)   # 右边界线宽
    # 调整 x 轴和 y 轴的刻度线粗细和长度
    ax.tick_params(axis='x', which='both', direction='in', width=v, length=1)
    ax.tick_params(axis='y', which='both', direction='in', width=v, length=1)
    plt.tight_layout()
    # plt.show()
    prefix_path_save = 'D:\\000__program_paper\\02__image_Segmentation\\01_threshold_seg_0529\\result\\curve'
    plt.savefig(f'{prefix_path_save}\\ConvergenceCurve_Image{img_id}_nTh{nth}.png',
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.clf()


if __name__ == '__main__':
    exp_idx = 0
    # [1, 2, 5, 8, 9, 12, 16, 18, 21, 27, 28, 32, 34, 35, 37, 40]
    # image_list = [1, 2, 5, 8, 9, 12, 16, 18, 21, 27, 28, 32, 34, 35, 37, 40]
    image_list = [2, 21]
    for image_id in tqdm(image_list, 'saving picture...'):
        for num_th in [4, 6, 8]:
            flag_gwo = 1
            flag_woa = 1
            flag_ssa = 1
            flag_pso = 1
            flag_ga = 1
            flag_fa = 1
            flag_psosa = 1
            flag_psossa = 1
            flag_psomjq = 1

            draw(exp_idx, image_id, num_th,
                 flag_gwo, flag_woa, flag_ssa, flag_pso, flag_ga, flag_fa, flag_psosa, flag_psossa, flag_psomjq)




