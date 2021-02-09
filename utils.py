import numpy as np
import random
import scipy.sparse as sp
from sklearn.metrics import f1_score
import copy


def load_dataset(dataset, node_file, edge_file):
    node_num = 0
    feature_num = 0
    label_set = set()
    label2node = dict()
    node_set = set()
    with open(node_file, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split('\t'))
            if len(temp) == 2:
                node_num = int(temp[0])
                feature_num = int(temp[1])
                print('node num: ', node_num)
                print('feature num: ', feature_num)
                continue
            node_id = int(temp[0])
            node_set.add(int(temp[0]))
            label = int(temp[1])
            if label not in label2node:
                label2node[label] = list()
            label2node[label].append(node_id)
    if dataset == 'email':
        train_proportion = 0.1
        val_proportion = 0.1
        test_proportion = 0.8
        label2node_new = dict()
        node_set_new = set()
        for label in label2node:
            if len(label2node[label]) >= 10:
                label2node_new[label] = label2node[label]
                for n in label2node[label]:
                    node_set_new.add(n)
        val_list = list()
        test_list = list()
        for label in label2node_new:
            val_num = int(len(label2node_new[label]) * val_proportion)
            test_num = int(len(label2node_new[label]) * test_proportion)
            test_val_node_list = list()
            test_val_node_list.extend(random.sample(label2node_new[label], val_num + test_num))
            test_list += test_val_node_list[-test_num:]  # test
            val_list += test_val_node_list[:-test_num]
        test_val_set = set(test_list).union(set(val_list))
        train_list = list()
        for label in label2node_new:
            candidate_list = list()
            train_num = int(len(label2node_new[label]) * train_proportion)
            for n in label2node_new[label]:
                if n not in test_val_set:
                    candidate_list.append(n)
            train_list.extend(random.sample(candidate_list, train_num))
    elif dataset == 'cora':
        train_proportion = 0.1
        val_proportion = 0.1
        test_proportion = 0.8
        label2node_new = dict()
        node_set_new = set()
        for label in label2node:
            if len(label2node[label]) >= 50:
                label2node_new[label] = label2node[label]
                for n in label2node[label]:
                    node_set_new.add(n)
        val_list = list()
        test_list = list()
        for label in label2node_new:
            val_num = int(len(label2node_new[label]) * val_proportion)
            test_num = int(len(label2node_new[label]) * test_proportion)
            test_val_node_list = list()
            test_val_node_list.extend(random.sample(label2node_new[label], val_num + test_num))
            test_list += test_val_node_list[-test_num:]  # test
            val_list += test_val_node_list[:-test_num]
        test_val_set = set(test_list).union(set(val_list))
        train_list = list()
        for label in label2node_new:
            candidate_list = list()
            train_num = int(len(label2node_new[label]) * train_proportion)
            for n in label2node_new[label]:
                if n not in test_val_set:
                    candidate_list.append(n)
            train_list.extend(random.sample(candidate_list, train_num))
    else:
        # TO DO
        train_list = list()
        test_list = list()
        val_list = list()
    features = np.zeros((node_num, feature_num))
    node2label_dict = dict()
    with open(node_file, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split('\t'))
            if len(temp) == 2:
                continue
            node_id = int(temp[0])
            features[node_id] = temp[2:]
            node2label_dict[node_id] = int(temp[1])
            label_set.add(int(temp[1]))
    label_num = len(label_set)
    adj = np.zeros((node_num, node_num))
    with open(edge_file, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split('\t'))
            adj[int(temp[0]), int(temp[1])] = 1.0
    if dataset == 'email':
        label_set.clear()
        head_label_set = set()
        for label in label2node:
            if len(label2node[label]) >= 10:
                label_set.add(label)
            if len(label2node[label]) >= 55:
                head_label_set.add(label)
        print(len(head_label_set))
        label_num = len(label_set)
        head_class = np.zeros(shape=(label_num,), dtype=np.int32)
        label_index = 0
        label2index = dict()
        for label in label_set:
            label2index[label] = label_index
            if label in head_label_set:
                head_class[label_index] = 1
            label_index += 1
        for node_id in node2label_dict:
            if node2label_dict[node_id] in label2index:
                node2label_dict[node_id] = label2index[node2label_dict[node_id]]
    elif dataset == 'cora':
        label_set.clear()
        head_label_set = set()
        for label in label2node:
            if len(label2node[label]) >= 10:
                label_set.add(label)
            if len(label2node[label]) >= 300:
                head_label_set.add(label)
        label_num = len(label_set)
        head_class = np.zeros(shape=(label_num,), dtype=np.int32)
        label_index = 0
        label2index = dict()
        for label in label_set:
            label2index[label] = label_index
            if label in head_label_set:
                head_class[label_index] = 1
            label_index += 1
        for node_id in node2label_dict:
            if node2label_dict[node_id] in label2index:
                node2label_dict[node_id] = label2index[node2label_dict[node_id]]
    else:
        # TO DO
        label2index = dict()
        head_class = np.zeros(shape=(label_num,), dtype=np.int32)
        pass
    y_train = np.zeros((node_num, label_num))
    y_val = np.zeros((node_num, label_num))
    y_test = np.zeros((node_num, label_num))
    train_mask = np.zeros((node_num,))
    val_mask = np.zeros((node_num,))
    test_mask = np.zeros((node_num,))

    for node_id in train_list:
        y_train[node_id, node2label_dict[node_id]] = 1.0
        train_mask[node_id] = 1.0
    for node_id in val_list:
        y_val[node_id, node2label_dict[node_id]] = 1.0
        val_mask[node_id] = 1.0
    for node_id in test_list:
        y_test[node_id, node2label_dict[node_id]] = 1.0
        test_mask[node_id] = 1.0

    features_01 = features
    #     features = features / features.sum(axis=1)[:,None]
    #     features_new = preprocess_features_other_dataset(features)
    # if dataset in {"amazon"}:
    #     fea_norm = np.linalg.norm(features, axis=1)
    #     features_new = features / fea_norm[:, None]
    # else:
    #     features_new = preprocess_features(features)
    fea_norm = np.linalg.norm(features, axis=1)
    features_new = features / fea_norm[:, None]

    features = np.mat(features_new)
    features_01 = np.mat(features_01)

    train_mask = np.array(train_mask, dtype=np.bool)
    val_mask = np.array(val_mask, dtype=np.bool)
    test_mask = np.array(test_mask, dtype=np.bool)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, label2index, node2label_dict, head_class


def load_segments(root_dir, label2index, segment_num_per_class, segment_length):
    segments_idx = np.zeros(shape=(len(label2index), segment_num_per_class, segment_length), dtype=np.int32)
    segments_mask = np.zeros(shape=(len(label2index), segment_num_per_class, segment_length), dtype=np.int32)
    segments_set = set()
    for label in label2index:
        with open(root_dir + 'segments/' + str(label) + '.txt', 'r') as fr:
            lines = fr.readlines()
            row_idx = 0
            for line in lines:
                temp = list(line.strip('\n').split(' '))
                col_idx = 0
                for n in temp:
                    segments_set.add(int(n))
                    segments_idx[int(label2index[label]), row_idx, col_idx] = int(n)
                    segments_mask[int(label2index[label]), row_idx, col_idx] = 1
                    col_idx += 1
                row_idx += 1
    return segments_idx, segments_mask, segments_set


def check_segments(segments_idx, segments_mask, segments_set):
    for i in range(segments_idx.shape[0]):
        for j in range(segments_idx.shape[1]):
            for k in range(segments_idx.shape[2]):
                if segments_mask[i][j][k] == 1:
                    if segments_idx[i][j][k] not in segments_set:
                        print('error')


def preprocess_features(features):
    row_sum = np.array(features.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    normalized_features = r_mat_inv.dot(features)
    return normalized_features


def read_node_file(node_file):
    node_num = 0
    feature_num = 0
    label2node = dict()
    with open(node_file, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split('\t'))
            if len(temp) == 2:
                node_num = int(temp[0])
                feature_num = int(temp[1])
                break
    node2label = np.zeros((node_num,), dtype=np.int32)
    feature = np.zeros((node_num, feature_num), dtype=np.float32)
    with open(node_file, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split('\t'))
            if len(temp) > 2:
                node_id = int(temp[0])
                label = int(temp[1])
                node2label[node_id] = label
                feature[node_id] = temp[2:]
                if label not in label2node:
                    label2node[label] = list()
                    label2node[label].append(node_id)
                else:
                    label2node[label].append(node_id)
    label2node_list = list()
    for i in range(len(label2node)):
        label2node_list.append(label2node[i])
    return node2label, label2node_list, feature


def generate_neighbor_dict(edge_file, node_num):
    neighbor_dict = [[] for _ in range(node_num)]
    with open(edge_file, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split('\t'))
            if len(temp) > 0:
                node_0 = int(temp[0])
                node_1 = int(temp[1])
                neighbor_dict[node_0].append(node_1)
    return neighbor_dict


def preprocess_adj_normalization(adj):
    num_nodes = adj.shape[0]
    adj = adj + np.eye(num_nodes)  # self-loop
    adj[adj > 0.0] = 1.0
    D_ = np.diag(np.power(np.sum(adj, axis=1), -0.5))

    adjNor = np.dot(np.dot(D_, adj), D_)
    return adj, adjNor


def processNodeInfo(adj, mask_nor, node_num):
    max_nei_num = np.max(np.sum(adj, axis=1)).astype(np.int32)

    neis = np.zeros((node_num, max_nei_num), dtype=np.int32)
    neis_mask = np.zeros((node_num, max_nei_num), dtype=np.float32)
    neis_mask_nor = np.zeros((node_num, max_nei_num), dtype=np.float32)
    neighboursDict = []
    inner_index = 0
    for i in range(node_num):
        inner_index = 0
        nd = []
        for j in range(node_num):

            if adj[i][j] == 1.0:
                neis[i][inner_index] = j
                neis_mask[i][inner_index] = 1.0
                neis_mask_nor[i][inner_index] = mask_nor[i][j]
                if i != j:
                    nd.append(j)
                inner_index += 1
        neighboursDict.append(nd)

    return neis, neis_mask, neis_mask_nor, neighboursDict


def prepareDataset_bigdata(nodes_num, max_degree, node_ids, eval_node_ids, node2label, neisMatrix,
                           neisMatrix_mask, neisMatrix_mask_nor, max_degree_setting=40):
    nodes_self = np.arange(nodes_num, dtype=np.int32)
    ones_nodes_self = np.ones(nodes_num, dtype=np.float32)
    neisMatrix = np.concatenate((nodes_self[:, None], neisMatrix), axis=1)  # shape=(nodes_num, max_degree+1)
    neisMatrix_mask = np.concatenate((ones_nodes_self[:, None], neisMatrix_mask),
                                     axis=1)  # shape=(nodes_num, max_degree+1)
    neisMatrix_mask_nor = np.concatenate((ones_nodes_self[:, None], neisMatrix_mask_nor),
                                         axis=1)  # shape=(nodes_num, max_degree+1)
    max_degree = max_degree + 1
    max_degree_setting = max_degree_setting + 1
    if max_degree > max_degree_setting:
        max_degree = max_degree_setting

    adj_1 = neisMatrix[node_ids]  # shape=(valid_node_num, max_neis_num)
    mask_1 = neisMatrix_mask[node_ids]  # shape=(valid_node_num, max_neis_num)
    mask_1_nor = neisMatrix_mask_nor[node_ids]  # shape=(valid_node_num, max_neis_num)

    adj_1_max_nei_num = np.max(np.sum(mask_1, axis=1)).astype(np.int32)
    if adj_1_max_nei_num > max_degree_setting:
        adj_1_max_nei_num = max_degree_setting
    adj_1 = adj_1[:, :adj_1_max_nei_num]
    mask_1 = mask_1[:, :adj_1_max_nei_num]
    mask_1_nor = mask_1_nor[:, :adj_1_max_nei_num]
    mask_1_nor[:, 1:] = mask_1[:, 1:] / np.sum(mask_1[:, 1:], axis=-1)[:, None]
    mask_1_nor[np.isinf(mask_1_nor)] = 0.

    valid_node_num_adj_1 = np.sum(mask_1).astype(np.int32)
    adj_1_new = np.zeros_like(adj_1).astype(np.int32)
    adj_2 = np.zeros((valid_node_num_adj_1, max_degree)).astype(np.int32)
    mask_2 = np.zeros((valid_node_num_adj_1, max_degree)).astype(np.float32)
    mask_2_nor = np.zeros((valid_node_num_adj_1, max_degree)).astype(np.float32)
    eval_node_dict = dict()
    index = 0
    for i in range(adj_1.shape[0]):
        nodeId = node_ids[i]
        eval_node_dict[nodeId] = i
        valid_num_row = np.sum(mask_1[i]).astype(np.int32)
        valid_ids_row = adj_1[i, :valid_num_row]
        adj_2[index: index + valid_num_row] = neisMatrix[valid_ids_row, :max_degree]
        mask_2[index:index + valid_num_row] = neisMatrix_mask[valid_ids_row, :max_degree]
        mask_2_nor[index:index + valid_num_row] = neisMatrix_mask_nor[valid_ids_row, :max_degree]
        adj_1_new[i, :valid_num_row] = np.arange(index, index + valid_num_row)

        index += valid_num_row

    adj_2_max_nei_num = np.max(np.sum(mask_2, axis=1)).astype(np.int32)
    adj_2 = adj_2[:, :adj_2_max_nei_num]
    mask_2 = mask_2[:, :adj_2_max_nei_num]
    mask_2_nor = mask_2_nor[:, :adj_2_max_nei_num]
    #     mask_2_nor = mask_2 / np.sum(mask_2, axis=-1)[:,None]
    mask_2_nor[:, 1:] = mask_2[:, 1:] / np.sum(mask_2[:, 1:], axis=-1)[:, None]
    mask_2_nor[np.isinf(mask_2_nor)] = 0.

    label_num_dict = dict()
    for n in eval_node_ids:
        if node2label[n] not in label_num_dict:
            label_num_dict[node2label[n]] = 0
        label_num_dict[node2label[n]] += 1
    label_num = 28
    max_node_num = 0
    for label in label_num_dict:
        if label_num_dict[label] > max_node_num:
            max_node_num = label_num_dict[label]
    node_type_idx = np.zeros(shape=(label_num, max_node_num), dtype=np.int32)
    node_type_mask = np.zeros(shape=(label_num, max_node_num))
    label_index = dict()
    for i in range(label_num):
        label_index[i] = 0
    for n in eval_node_ids:
        node_type_idx[node2label[n], label_index[node2label[n]]] = eval_node_dict[n]
        node_type_mask[node2label[n], label_index[node2label[n]]] = 1
        label_index[node2label[n]] += 1
    return adj_2, mask_2, mask_2_nor, adj_1_new, mask_1, mask_1_nor, node_type_idx, node_type_mask, eval_node_dict


def re_idx_segment_node(segments_idx, segments_mask, node_ids, node_dict):
    temp_idx = copy.deepcopy(segments_idx)
    for i in range(segments_idx.shape[0]):
        for j in range(segments_idx.shape[1]):
            for k in range(segments_idx.shape[2]):
                if segments_mask[i][j][k] == 1:
                    temp_idx[i][j][k] = node_dict[temp_idx[i][j][k]]
    temp_node_idx = copy.deepcopy(node_ids)
    for i in range(len(temp_node_idx)):
        temp_node_idx[i] = node_dict[node_ids[i]]
    return temp_idx, temp_node_idx


def get_val_test_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return minibatches


def micro_macro_f1_removeMiLabels(y_true, y_pred, labelsList):
    return f1_score(y_true, y_pred, labels=labelsList, average="micro"), f1_score(y_true, y_pred, average="macro")
