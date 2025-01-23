from sklearn.metrics import precision_recall_curve, f1_score
import matplotlib.pyplot as plt

def evaluate_precision_recall(similarity_matrix, ground_truth):
    """
    计算 Precision-Recall 曲线，并计算 F1-max.
    
    Args:
        similarity_matrix: 查询样本和数据库样本的相似度矩阵 [num_queries, num_database_samples]
        ground_truth: 每个查询对应的真实标签 [num_queries, num_database_samples]
    """
    # 将相似度值拉平成一维数组
    predictions = similarity_matrix.ravel()
    ground_truth = ground_truth.ravel()

    # 计算 Precision-Recall 曲线
    precision, recall, _ = precision_recall_curve(ground_truth, predictions)
    
    # 计算 F1 分数
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    F1_max = np.max(f1_scores)
    
    print(f'F1-max: {F1_max:.4f}')
    
    # 绘制 Precision-Recall 曲线
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

# ground_truth 是查询与数据库样本的真实匹配标签，二进制矩阵，1 表示匹配，0 表示不匹配
# 计算 Precision-Recall 曲线和 F1-max
evaluate_precision_recall(similarity_matrix, ground_truth)
