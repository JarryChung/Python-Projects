# encoding:utf8

from math import log
import operator


# 创建一个简单的数据集
def create_data_set():
    data_set = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    # 这里的labels保存的是属性名称
    labels = ['no surfacing', 'flippers']
    return data_set, labels


# 编写函数计算熵
def calc_entropy(data_set):
    # 获取总的训练数据数
    num_entries = len(data_set)
    # 创建一个字典统计各个类别的数据量
    label_counts = {}
    # 遍历每个实例，统计标签的频数
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    entropy = 0.0
    for key in label_counts:
        p = float(label_counts[key]) / num_entries
        # 以2为底的对数
        entropy -= p * log(p, 2)
    return entropy


# 编写函数，实现按照给定特征划分数据集
def split_data_set(data_set, axis, value):
    return_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            # 隔开axis这一列提取其它列的数据
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            return_data_set.append(reduced_feat_vec)
    return return_data_set


# 实现特征选择函数。遍历整个数据集，循环计算熵和split_data_set()函数，找到最好的特征划分方式
def choose_best_feature_to_split(data_set):
    # 获取属性个数，保存到变量num_features
    # 注意数据集中最后一列是分类结果
    num_features = len(data_set[0]) - 1
    base_entropy = calc_entropy(data_set)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        # 获取数据集中某一属性的所有取值
        feat_list = [example[i] for example in data_set]
        # 获取该属性所有不重复的取值，保存到unique_vals中
        # 可使用set()函数去重
        unique_vals = set(feat_list)
        new_entropy = 0.0
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)
            # 计算按照第i列的某一个值分割数据集后的熵
            prob = len(sub_data_set) / float(len(data_set))
            # 条件熵的计算
            new_entropy += prob * calc_entropy(sub_data_set)
            # 信息增益，就yes熵的减少，也就yes不确定性的减少
            info_gain = base_entropy - new_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i
    return best_feature


# 决策树创建过程中会采用递归的原则处理数据集
# 递归的终止条件为：程序遍历完所有划分数据集的属性；
# 或者每一个分支下的所有实例都具有相同的分类。如果数据集已经处理了所有属性，
# 但是类标签依然不是唯一的，此时我们需要决定如何定义该叶子节点，
# 在这种情况下，通常会采用多数表决的方法决定分类。
def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


# 创建决策树
def create_tree(data_set, labels):
    # 获取类别列表，类别信息在数据集中的最后一列
    # 使用变量class_list
    class_list= [example[-1] for example in data_set]
    # 以下两段是递归终止条件

    # 如果数据集中所有数据都属于同一类则停止划分
    # 可以使用classList.count(XXX)函数获得XXX的个数，
    # 然后那这个数和class_list的长度进行比较，相等则说明
    # 所有数据都属于同一类，返回该类别即可
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # 如果已经遍历完所有属性则进行投票，调用上一步的函数
    # 注意，按照所有属性分割完数据集后，数据集中会只剩下
    # 一列，这一列是分类结果
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)

    # 调用特征选择函数选择最佳分割属性，保存到bestFeat
    # 根据bestFeat获取属性名称，保存到bestFeatLabel中
    best_feat = choose_best_feature_to_split(data_set)  # 最优划分特征
    best_feat_label = labels[best_feat]

    # 初始化决策树，可以先把第一个属性填好
    # 使用字典类型储存树的信息
    my_tree = {best_feat_label: {}}

    # 删除最佳分离属性的名称以便递归调用
    del(labels[best_feat])

    # 获取最佳分离属性的所有不重复的取值保存到unique_vals
    feat_values = [example[best_feat] for example in data_set]
    unique_vals = set(feat_values)
    for value in unique_vals:
        # 复制属性名称，以便递归调用
        sub_label = labels[:]
        # 递归调用本函数生成决策树
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_label)
    return my_tree


# 利用构建好的决策树进行分类
def classify(input_tree, feat_labels, test_vec):
    # 获取树的第一个节点，即属性名称
    first_str = list(input_tree.keys())[0]
    # 获取该节点下的值
    second_dict = input_tree[first_str]
    # 获取该属性名称在原属性名称列表中的下标
    # 保存到变量feat_index中
    # 可使用index(XXX)函数获得XXX的下标

    feat_index = feat_labels.index(first_str)

    # 获取待分类数据中该属性的取值，然后在secondDict
    # 中寻找对应的项的取值
    # 如果得到的是一个字典型的数据，说明在该分支下还
    # 需要进一步比较，因此进行循环调用分类函数；
    # 如果得到的不是字典型数据，说明得到了分类结果
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label


if __name__ == '__main__':
    data, labels = create_data_set()
    my_tree = create_tree(data, labels)
    print(my_tree)
    label = create_data_set()[1]
    class_label = classify(my_tree, label, [1, 0])
    print(class_label)
    # 上面的代码中[1,0]为新数据
