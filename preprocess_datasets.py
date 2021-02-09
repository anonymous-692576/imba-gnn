import argparse
import scipy.sparse as sp
import numpy as np


def load_email_dataset():
    node_dict = dict()
    graph_dict = dict()
    node_class_dict = dict()
    node_feature_dict = dict()
    node_index = 0
    with open('./datasets/email/labels.txt', "r") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split(' '))
            if temp[0] not in node_dict:
                node_dict[temp[0]] = node_index
                node_index += 1
    # reverse_node_dict = dict(zip(node_dict.values(), node_dict.keys()))
    for n in node_dict.keys():
        graph_dict[node_dict[n]] = set()
    edge_num = 0
    with open('./datasets/email/edges.txt', "r") as fr:
        lines = fr.readlines()
        for line in lines:
            edge_num += 1
            temp = list(line.strip('\n').split(' '))
            graph_dict[node_dict[temp[0]]].add(node_dict[temp[1]])
            graph_dict[node_dict[temp[1]]].add(node_dict[temp[0]])
    with open('./datasets/email/labels.txt', "r") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split(' '))
            node_class_dict[node_dict[temp[0]]] = temp[1]
    with open('./datasets/email/graph.embeddings', "r") as fr:
        lines = fr.readlines()
        for line in lines:
            if len(line) == 2:
                continue
            temp = list(line.strip('\n').split(' '))
            node_feature_dict[int(temp[0])] = temp[1:]
    print('dataset ' + dataset)
    print('total node number: %d' % len(graph_dict.keys()))
    print('total edge number: %d' % edge_num)
    return graph_dict, node_class_dict, node_feature_dict


def load_cora_dataset():
    graph_dict = dict()
    node_class_dict = dict()
    node_feature_dict = dict()
    with np.load('./datasets/cora/cora.npz', allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])
        attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                    shape=loader['attr_shape'])
        labels = loader['labels']

        node_names = loader.get('idx_to_node')
        attr_names = loader.get('idx_to_attr')
        class_names = loader.get('idx_to_class')
    adj = adj_matrix.A
    node_num = len(node_names.tolist())
    for i in range(node_num):
        if i not in graph_dict:
            graph_dict[i] = set()
    nnz = adj_matrix.nonzero()
    for i, j in zip(nnz[0], nnz[1]):
        graph_dict[i].add(j)
        graph_dict[j].add(i)
    # for i in range(node_num):
    #     for j in range(node_num):
    #         if adj[i][j] == 1.:
    #             graph_dict[i].add(j)
    label_list = labels.tolist()
    for i in range(node_num):
        node_class_dict[i] = label_list[i]
    features = attr_matrix.A
    for i in range(node_num):
        node_feature_dict[i] = features[i].tolist()
    return graph_dict, node_class_dict, node_feature_dict


def save_graph(graph_dict, save_dir):
    with open(save_dir, 'w') as fw:
        for n in graph_dict.keys():
            fw.write(str(n))
            for adj in graph_dict[n]:
                fw.write(' ' + str(adj))
            fw.write('\n')


def generate_input_file(graph_dict, node_class_dict, node_feature_dict):
    with open('./data/' + dataset + '/graph.edge', 'w') as fw:
        for n in graph_dict:
            for adj in graph_dict[n]:
                fw.write(str(n) + '\t' + str(adj) + '\n')
    with open('./data/' + dataset + '/graph.node', 'w') as fw:
        fw.write(str(len(node_class_dict)) + '\t' + str(len(node_feature_dict[0])) + '\n')
        for n in graph_dict:
            fw.write(str(n) + '\t' + str(node_class_dict[n]))
            for d in node_feature_dict[n]:
                fw.write('\t' + str(d))
            fw.write('\n')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='preprocess dataset')
    # parser.add_argument('-d', metavar='dataset', required=True,
    #                     dest='dataset', action='store',
    #                     help='dataset to be preprocessed')
    # args = parser.parse_args()
    dataset = 'cora'
    if dataset == 'email':
        graph, node_class, node_feature = load_email_dataset()
        generate_input_file(graph, node_class, node_feature)
    elif dataset == 'cora':
        graph, node_class, node_feature = load_cora_dataset()
        generate_input_file(graph, node_class, node_feature)
    else:
        print('error')
