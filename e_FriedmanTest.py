import numpy as np
from scipy.stats import friedmanchisquare, rankdata
from d_result_docx import algorithm_result


def result_orgnazition(algname_l, image_l):
    groups = []
    for an in algname_l:
        group = []
        for i in image_l:
            kapur = algorithm_result(exp_name=exp_name, alg_name=an, img_i=i)
            group += kapur.tolist()
        groups.append(group)

    return groups


def get_average_rank(groups):
    groups = np.array(groups).T
    rank_l = []
    for i in range(len(groups)):
        rank = rankdata(-groups[i])
        rank_l.append(rank)
    rank_np = np.array(rank_l).T

    a_rank = []
    for i in range(len(rank_np)):
        a_rank.append(round(rank_np[i].mean(), 4))
    return a_rank


if __name__ == '__main__':
    exp_name = 'tumer'
    algname_list = ['PSO', 'PSOsa', 'PSOssa', 'PSOmjq']
    # algname_list = ['GWO', 'WOA', 'SSA', 'PSO', 'GA', 'FA', 'PSOsa', 'PSOssa', 'PSOmjq']
    image_id_l = [1, 60, 85, 182, 186, 248, 270, 281, 286]
    gs = result_orgnazition(algname_list, image_id_l)

    # 使用Friedman秩和检验
    statistic, p_value = friedmanchisquare(gs[0], gs[1], gs[2], gs[3])
    print(statistic, p_value)
    # 判断是否拒绝零假设
    alpha = 0.05
    if p_value < alpha:
        print("拒绝零假设，说明至少有一组数据的中位数不同。")
    else:
        print("无法拒绝零假设，说明各组数据的中位数相等。")

    average_ranks = get_average_rank(gs)
    print(average_ranks)

