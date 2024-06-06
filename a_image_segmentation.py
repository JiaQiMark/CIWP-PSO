import h5py
import cv2
import numpy as np

from c_fitness_function import get_kapur_entropy
from optimization.GWO import gray_wolf_optimization
from optimization.BES import bald_eagle_search
from optimization.WOA import whale_optimization
from optimization.SSA import sparrow_search_optimization
from optimization.PSO import particle_swarm_optimization
from optimization.GA import genetic_algorithm
from optimization.FA import firefly_algorithm
from optimization.WOAmjq import mjq_whale_optimization
from optimization.PSOsa import pso_self_adaptive
from optimization.PSOssa import pso_ssa
from optimization.PSOmjq import pso_mjq


def save_result(bench_or_tumer, optimal_th, curve, img_index, alg_name, Dn):
    exp_name = ['bench', 'tumer']
    exp = exp_name[bench_or_tumer]
    thresholds_path  = f"result//result_{alg_name}//{exp}//{alg_name}_{exp}-{img_index}_thresholds_{Dn}.npy"
    convergence_path = f"result//result_{alg_name}//{exp}//{alg_name}_{exp}-{img_index}_convergence_curve_{Dn}.npy"
    np.save(thresholds_path, optimal_th)
    np.save(convergence_path, curve)
    print(f"{alg_name}-{exp_name[bench_or_tumer]}-{img_index}:", optimal_th)


def get_optimal_threshold(bench_or_tumer, max_iterations, population_size, Dn, search_range,
                          image_index, alg_name,
                          f_gwo, f_bes, f_woa, f_ssa, f_pso, f_ga, f_fa, f_woamjq, f_psosa, f_psossa, f_psomjq):
    if bench_or_tumer == 0:
        np_data_path = (f"D:\\000__program_paper\\02__image_Segmentation\\01_threshold_seg_0529\\picture"
                        f"\\bench__GrayScaled\\benchmark_ ({image_index}).npy")
        image = np.load(np_data_path)
    elif bench_or_tumer == 1:
        matlab_data_path = f"D:\\000__program\\00__dataset\\01_img_Seg\\01__brainTumor_1_766\\{image_index}.mat"
        with h5py.File(matlab_data_path, 'r') as file:
            variables = list(file.keys())
            matlab_data = file['cjdata']
            # print(type(matlab_data))

            label = np.array(matlab_data['label'])
            image = np.array(matlab_data['image'])
            patient_id = np.array(matlab_data['PID'])
            tumor_border = np.array(matlab_data['tumorBorder'])
            tumor_mask = np.array(matlab_data['tumorMask'])
            cvu8_image = cv2.convertScaleAbs(image)

    def objective_function(p):
        return get_kapur_entropy(p, image)

    def objective_function_getmin(p):
        return -get_kapur_entropy(p, image)

    if f_bes:
        # 开始优化图像的阈值
        optimal_thresholds, convergence_curve = bald_eagle_search(max_iterations, population_size, Dn,
                                                                  search_range, objective_function)
        ni = 0
        save_result(bench_or_tumer, optimal_thresholds, convergence_curve, image_index, alg_name[ni], Dn)

    if f_gwo:
        # 开始优化图像的阈值
        optimal_thresholds, convergence_curve = gray_wolf_optimization(max_iterations, population_size, Dn,
                                                                       search_range, objective_function)
        ni = 1
        save_result(bench_or_tumer, optimal_thresholds, convergence_curve, image_index, alg_name[ni], Dn)

    if f_woa:
        # 开始优化图像的阈值
        optimal_thresholds, convergence_curve = whale_optimization(max_iterations, population_size, Dn,
                                                                   search_range, objective_function)
        ni = 2
        save_result(bench_or_tumer, optimal_thresholds, convergence_curve, image_index, alg_name[ni], Dn)

    if f_ssa:
        # 开始优化图像的阈值
        optimal_thresholds, convergence_curve = sparrow_search_optimization(max_iterations, population_size, Dn,
                                                                            search_range, objective_function)
        ni = 3
        save_result(bench_or_tumer, optimal_thresholds, convergence_curve, image_index, alg_name[ni], Dn)

    if f_pso:
        # 开始优化图像的阈值
        optimal_thresholds, convergence_curve = particle_swarm_optimization(max_iterations, population_size, Dn,
                                                                            search_range, objective_function_getmin)
        ni = 4
        save_result(bench_or_tumer, optimal_thresholds, -1 * np.array(convergence_curve), image_index, alg_name[ni], Dn)

    if f_ga:
        # 开始优化图像的阈值
        optimal_thresholds, convergence_curve = genetic_algorithm(max_iterations, population_size, Dn,
                                                                  search_range, objective_function)
        ni = 5
        save_result(bench_or_tumer, optimal_thresholds, convergence_curve, image_index, alg_name[ni], Dn)

    if f_fa:
        # 开始优化图像的阈值
        optimal_thresholds, convergence_curve = firefly_algorithm(max_iterations, population_size, Dn,
                                                                  search_range, objective_function_getmin)
        ni = 6
        save_result(bench_or_tumer, optimal_thresholds, -1 * np.array(convergence_curve), image_index, alg_name[ni], Dn)

    if f_woamjq:
        # 开始优化图像的阈值
        optimal_thresholds, convergence_curve = mjq_whale_optimization(max_iterations, population_size, Dn,
                                                                       search_range, objective_function)
        ni = 7
        save_result(bench_or_tumer, optimal_thresholds, convergence_curve, image_index, alg_name[ni], Dn)

    if f_psosa:
        # 开始优化图像的阈值
        optimal_thresholds, convergence_curve = pso_self_adaptive(max_iterations, population_size, Dn,
                                                                  search_range, objective_function_getmin)
        ni = 8
        save_result(bench_or_tumer, optimal_thresholds, -1 * np.array(convergence_curve), image_index, alg_name[ni], Dn)

    if f_psossa:
        # 开始优化图像的阈值
        optimal_thresholds, convergence_curve = pso_ssa(max_iterations, population_size, Dn,
                                                        search_range, objective_function_getmin)
        ni = 9
        save_result(bench_or_tumer, optimal_thresholds, -1 * np.array(convergence_curve), image_index, alg_name[ni], Dn)

    if f_psomjq:
        # 开始优化图像的阈值
        optimal_thresholds, convergence_curve = pso_mjq(max_iterations, population_size, Dn,
                                                        search_range, objective_function)
        ni = 10
        save_result(bench_or_tumer, optimal_thresholds, np.array(convergence_curve), image_index, alg_name[ni], Dn)


max_ite = 100
p_size = 30
dimension_n = 6
grayvalue_max = 4095
ranges = [0, grayvalue_max]

flag_bes = 0
flag_gwo = 1
flag_woa = 1
flag_ssa = 1
flag_pso = 1
flag_ga  = 1
flag_fa  = 1
flag_woamjq = 0
flag_psosa = 1
flag_psossa = 1
flag_psomjq = 1
algname = ['BES', 'GWO', 'WOA', 'SSA', 'PSO', 'GA', 'FA', 'WOAmjq', 'PSOsa', 'PSOssa', 'PSOmjq']


def main():
    bench_or_tumer = 0
    for i in range(39, 41):
        for dn in [4, 6, 8]:
            get_optimal_threshold(bench_or_tumer, max_ite, p_size, dn, ranges, i, algname,
                                  flag_gwo, flag_bes, flag_woa, flag_ssa, flag_pso, flag_ga, flag_fa, flag_woamjq,
                                  flag_psosa, flag_psossa, flag_psomjq)


if __name__ == '__main__':
    main()
