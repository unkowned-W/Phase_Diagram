import pandas as pd
import numpy as np
from scipy.stats import qmc
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
from scipy.spatial.distance import cdist

def load_data(path):
    data = pd.read_csv(path, usecols=["Gd(ClO4)3", "Co(Ac)2", "NaAc", "NaOH", "Y"])
    X = data[["Gd(ClO4)3", "Co(Ac)2", "NaAc", "NaOH"]].values
    Y = data["Y"].astype(bool).values
    return X, Y
 
def train_models(X, Y):
    """训练多个模型并返回模型列表和表现"""
    model_behavior = []
    clfs = []
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state = i, stratify=Y)
        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)
        model_behavior.append(models)
        clfs.append(clf)
    return clfs, model_behavior

def generate_lhs_samples(n_samples=1000, random_state=None):
        """生成拉丁超立方采样的候选点"""
        sampler = qmc.LatinHypercube(d=4, seed=random_state)
        samples = sampler.random(n=n_samples)
        bounds = np.array([
            [1.0, 3.0],    # Gd(ClO4)3范围
            [0.5, 2.0],    # Co(Ac)2范围
            [0.5, 2.0],    # NaAc范围
            [2.0, 4.0]     # NaOH范围
        ])
        points = qmc.scale(samples, bounds[:, 0], bounds[:, 1])
        return points

def calculate_entropy(clfs, model_behavior, all_points):
    """计算熵值"""
    all_probability = []
    valid_model_count = 0  # 统计有效模型数，避免空列表
    
    # 遍历有效模型，而非固定100次
    for i, clf in enumerate(clfs):
        if i >= len(model_behavior):  # 防止索引越界
            break
        target_model = model_behavior[i].index[0]
        if target_model not in clf.models:  # 提前判断，避免异常
            continue
        
        try:
            # 直接调用predict_proba，批量处理all_points
            prob = clf.models[target_model].predict_proba(all_points)
            all_probability.append(prob)
            valid_model_count += 1
        except AttributeError:  # 精准捕获不支持predict_proba的异常
            continue
    
    # 处理无有效模型的情况
    if valid_model_count == 0:
        raise ValueError("没有找到支持predict_proba的有效模型")
    
    # 计算多个模型的平均概率分布，再计算熵值
    all_probability = np.array(all_probability)
    ensemble_probability = all_probability.mean(axis=0)
    point_entropy = entropy(ensemble_probability, axis=1)
    
    # 拼接结果
    points_with_entropy = np.hstack([all_points, point_entropy.reshape(-1, 1)])
    return points_with_entropy

def select_points(sorted_points_with_entropy, min_distance=0.5, n_select=100):
    """选择满足最小距离的点(用cdist批量计算距离,降低时间复杂度)"""
    if len(sorted_points_with_entropy) < n_select:
        raise ValueError("候选点数量不足，无法选择{}个点".format(n_select))
    
    # 提取所有候选点的前4列（坐标），后续只处理坐标
    candidate_coords = sorted_points_with_entropy[:, :4]
    selected_coords = []
    selected_all = []  # 保存完整的行（坐标+熵值）
    
    #先选第一个点
    selected_coords.append(candidate_coords[0])
    selected_all.append(sorted_points_with_entropy[0])
    #批量计算距离
    for i in range(1, len(candidate_coords)):
        if len(selected_all) >= n_select:
            break
        # 批量计算当前候选点与所有已选点的距离
        current_coord = candidate_coords[i].reshape(1, -1)
        distances = cdist(current_coord, np.array(selected_coords), metric='euclidean')[0]
        # 判断最小距离是否满足要求
        if np.min(distances) >= min_distance:
            selected_coords.append(candidate_coords[i])
            selected_all.append(sorted_points_with_entropy[i])
    
    return selected_all