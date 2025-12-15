import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.semi_supervised import LabelSpreading

def normalize_data(X):
    """对多维特征数组进行归一化处理"""
    # 输入校验
    if not isinstance(X, np.ndarray):
        X = np.array(X)  # 兼容列表输入
    if X.ndim != 2:
        raise ValueError(f"输入X必须是二维数组,当前维度: {X.ndim}")
    if X.shape[0] < 2:
        raise ValueError(f"样本数不能少于2,当前样本数: {X.shape[0]}")
    
    # 初始化参数字典
    scaler_params = {}
    normalized_X = X.copy().astype(np.float64)  # 避免修改原数据，统一浮点型
 
    # Min-Max归一化: (X - min) / (max - min)
    scaler_params['min'] = np.min(normalized_X, axis=0)  # 按列（特征）计算最小值
    scaler_params['max'] = np.max(normalized_X, axis=0)
    # 处理最大值=最小值的情况（避免除以0）
    range_vals = scaler_params['max'] - scaler_params['min']
    range_vals[range_vals == 0] = 1e-8  # 极小值替代0
    normalized_X = (normalized_X - scaler_params['min']) / range_vals
    
    return normalized_X

def train_LSmodels(X, Y):
    """训练LabelSpreading模型"""
    best_score = -1  # 初始化为极小值，确保第一个模型能更新
    best_model = None
    all_performance = [] 
    for random_state in range(42, 58):
        print(f"Training LabelSpreading with random_state={random_state}")       
        # 划分训练集/测试集（8:2）
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=random_state
        )
        # 训练集再划分：实际训练集/验证集（8:2，最终训练集占整体64%）
        X_train_real, X_val, y_train_real, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_state
        )
        # 直接初始化并训练随机森林
        LS_model = LabelSpreading(
            kernel='rbf',  # LabelSpreading默认核函数，显式指定更清晰
            gamma=20      # 默认值，可根据数据调整
        )
        X_train_real = X_train_real.astype(np.float64)
        X_val = X_val.astype(np.float64)
        X_test = X_test.astype(np.float64)
        LS_model.fit(X_train_real, y_train_real)
        # 在验证集上评估（用于模型选择）
        val_score = LS_model.score(X_val, y_val)
        # 在测试集上评估（最终性能）
        test_performance = get_evaluation(LS_model, X_test, y_test)
        all_performance.append(test_performance)
        # 更新最佳模型
        if val_score > best_score:
            best_score = val_score
            best_model = LS_model
    # 边界检查：避免无模型时报错
    if best_model is None:
        raise ValueError("未训练出任何模型，请检查输入数据或循环范围")
    print(f"最佳模型验证集准确率: {best_score:.4f}")
    return best_model, all_performance

def train_RFmodels(X, Y):
    """训练随机森林模型,返回训练好的模型和其性能指标，和特征重要性"""
    best_score = -1  # 初始化为极小值，确保第一个模型能更新
    best_model = None
    all_performance = [] 
    all_feature_importance = []  # 存储每一轮的特征重要性
    for random_state in range(42, 58):
        print(f"Training Random Forest with random_state={random_state}")       
        # 划分训练集/测试集（8:2）
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=random_state
        )
        # 训练集再划分：实际训练集/验证集（8:2，最终训练集占整体64%）
        X_train_real, X_val, y_train_real, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_state
        )
        # 直接初始化并训练随机森林
        RF_model = RandomForestClassifier(
        n_estimators=100,    # 树的数量（默认100，可根据数据调整，如200/300）
        max_depth=None,      # 树的最大深度（None表示不限制，可设为10/20防止过拟合）
        min_samples_split=2, # 拆分节点的最小样本数（默认2，小数据集可设5/10）
        min_samples_leaf=1,  # 叶节点最小样本数（默认1）
        bootstrap=True,      # 是否使用自助采样（默认True，推荐保留）
        random_state=random_state,  # 固定随机种子，保证复现性
        n_jobs=-1            # 并行训练（使用所有CPU核心，大幅提升训练速度）
    )
        RF_model.fit(X_train_real, y_train_real)
        feature_importance = RF_model.feature_importances_
        all_feature_importance.append(feature_importance)
        # 在验证集上评估（用于模型选择）
        val_score = RF_model.score(X_val, y_val)

        # 在测试集上评估（最终性能）
        test_performance = get_evaluation(RF_model, X_test, y_test)
        all_performance.append(test_performance)
        # 更新最佳模型
        if val_score > best_score:
            best_score = val_score
            best_model = RF_model
    # 边界检查：避免无模型时报错
    if best_model is None:
        raise ValueError("未训练出任何模型，请检查输入数据或循环范围")
    print(f"最佳模型验证集准确率: {best_score:.4f}")
    all_feature_importance = np.array(all_feature_importance)
    return best_model, all_performance, all_feature_importance

def get_evaluation(model, X_test, y_test):
    """计算模型的四项评估指标"""
    # 预测
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    except Exception as e:
        raise ValueError(f"模型预测失败: {str(e)}")
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')  
    # ROC AUC需要概率预测，处理无法获取概率的情况
    roc_auc = None
    if y_pred_proba is not None and len(y_pred_proba) > 0:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        roc_auc = float('nan')
        print("警告: 模型不支持概率预测，无法计算ROC AUC")
    
    return [accuracy, balanced_acc, f1, roc_auc]

def select_best_model(clfs, model_behavior, metric_weights=None):
    """基于四项指标加权排名选择最优模型"""
    if metric_weights is None:
        metric_weights = [0.25, 0.25, 0.25, 0.25]
    
    # 收集每个模型在所有随机种子下的指标
    model_all_metrics = {}  # {model_name: [[acc1, bal_acc1, roc1, f11], [acc2, bal_acc2, roc2, f12], ...]}
    
    for models_df in model_behavior:
        for model_name in models_df.index:
            try:
                # 从DataFrame中获取四项指标
                metrics = [
                    models_df.loc[model_name, 'Accuracy'],
                    models_df.loc[model_name, 'Balanced Accuracy'],
                    models_df.loc[model_name, 'ROC AUC'],
                    models_df.loc[model_name, 'F1 Score']
                ]
                
                if model_name not in model_all_metrics:
                    model_all_metrics[model_name] = []
                
                model_all_metrics[model_name].append(metrics)
            except KeyError as e:
                print(f"警告: 模型 {model_name} 缺少指标 {e}，跳过")
                continue
    
    if not model_all_metrics:
        raise ValueError("没有找到有效的模型指标数据")
    
    # 计算每个模型的平均指标
    model_avg_metrics = {}
    for model_name, metrics_list in model_all_metrics.items():
        metrics_array = np.array(metrics_list)
        model_avg_metrics[model_name] = np.mean(metrics_array, axis=0)
    
    # 计算加权综合得分
    model_scores = {}
    for model_name, avg_metrics in model_avg_metrics.items():
        # 加权平均计算综合得分
        weighted_score = np.average(avg_metrics, weights=metric_weights)
        model_scores[model_name] = weighted_score
    
    # 选择综合得分最高的模型
    best_model_name = max(model_scores.items(), key=lambda x: x[1])[0]
    best_avg_metrics = model_avg_metrics[best_model_name].tolist()
    
    # 查找对应的分类器实例
    best_clf = None
    for clf in clfs:
        # 检查clf是否包含该模型
        if hasattr(clf, 'models') and best_model_name in clf.models:
            best_clf = clf
            break
        # 有些版本的LazyClassifier模型存储在predictions中
        elif hasattr(clf, 'predictions') and best_model_name in clf.predictions:
            best_clf = clf
            break
    
    if best_clf is None:
        # 如果无法直接找到，返回第一个包含该模型名的clf
        for clf in clfs:
            try:
                # 尝试获取模型
                model = clf.models[best_model_name]
                best_clf = clf
                break
            except (KeyError, AttributeError):
                continue
    
    if best_clf is None:
        print(f"警告: 未能找到模型 {best_model_name} 对应的分类器实例")
    
    return best_clf, best_model_name, best_avg_metrics #返回最佳模型实例、名称及其平均指标

def select_best_model_byrank(clfs, model_behavior):
    """基于平均排名选择最优模型"""
    model_all_metrics = {}
    
    # 收集指标数据
    for models_df in model_behavior:
        for model_name in models_df.index:
            try:
                metrics = [
                    models_df.loc[model_name, 'Accuracy'],
                    models_df.loc[model_name, 'Balanced Accuracy'],
                    models_df.loc[model_name, 'ROC AUC'],
                    models_df.loc[model_name, 'F1 Score']
                ]
                
                if model_name not in model_all_metrics:
                    model_all_metrics[model_name] = []
                
                model_all_metrics[model_name].append(metrics)
            except KeyError:
                continue
    
    # 计算每个模型的平均指标
    model_avg_metrics = {}
    for model_name, metrics_list in model_all_metrics.items():
        model_avg_metrics[model_name] = np.mean(metrics_list, axis=0)
    
    # 转换为DataFrame便于排序
    import pandas as pd
    metrics_df = pd.DataFrame.from_dict(model_avg_metrics, orient='index',
                                        columns=['Accuracy', 'Balanced Accuracy', 'ROC AUC', 'F1 Score'])
    
    # 对每项指标进行排名（值越大排名越高）
    rank_df = metrics_df.rank(ascending=True, method='min')  # 改为ascending=True，值越大排名越高
    
    # 计算平均排名（平均排名越高越好）
    rank_df['Avg_Rank'] = rank_df.mean(axis=1)
    
    # 选择平均排名最高的模型
    best_model_name = rank_df['Avg_Rank'].idxmax()
    best_avg_metrics = metrics_df.loc[best_model_name].values.tolist()
    
    # 查找对应的分类器实例
    best_clf = None
    for clf in clfs:
        if hasattr(clf, 'models') and best_model_name in clf.models:
            best_clf = clf
            break
    
    return best_clf, best_model_name, best_avg_metrics