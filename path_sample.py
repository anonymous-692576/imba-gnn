import numpy as np
import os
import utils
import tqdm
import random
import math
import datetime
import time


def prepare_random_walk_and_segments(root_dir, dataset, random_walk_path_file, sample_batch_size, sample_times_per_node,
                                     max_length_path, min_length_segment, max_length_segment, segments_file,
                                     segment_batch_size):
    start_time = time.time()
    print('Start to load nodes information, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    root_dir = root_dir + dataset + '/'
    node_file = root_dir + 'graph.node'
    edge_file = root_dir + 'graph.edge'
    if os.path.exists(root_dir + 'node2label.npy'):
        node2label = np.load(root_dir + 'node2label.npy', allow_pickle=True)
        label2node = np.load(root_dir + 'label2node.npy', allow_pickle=True)
        feature = np.load(root_dir + 'feature.npy', allow_pickle=True)
    else:
        node2label, label2node, feature = utils.read_node_file(node_file)
        np.save(root_dir + 'node2label.npy', node2label)
        np.save(root_dir + 'label2node.npy', label2node)
        np.save(root_dir + 'feature.npy', feature)
    node2label_dict = dict()
    node_num = node2label.shape[0]
    label_num = len(label2node)
    feature_num = feature.shape[1]
    for i in range(node2label.shape[0]):
        node2label_dict[i] = node2label[i]
    print('Start to load edges information, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    if os.path.exists(root_dir + 'neighbor_dict.npy'):
        neighbor_dict = list(np.load(root_dir + 'neighbor_dict.npy', allow_pickle=True))
    else:
        neighbor_dict = utils.generate_neighbor_dict(edge_file, node_num)
        np.save(root_dir + 'neighbor_dict.npy', neighbor_dict)

    print('Start to random walk sampling, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    if not os.path.exists(random_walk_path_file):
        random_walk_sample(root_dir + random_walk_path_file, neighbor_dict, node_num, sample_batch_size,
                           sample_times_per_node, max_length_path)
    print('Extract subpaths from the sampled paths, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    if not os.path.exists(segments_file):
        segment_extraction(root_dir, random_walk_path_file, segment_batch_size,
                           min_length_segment, max_length_segment, segments_file, node2label_dict)
    return node2label_dict


def random_walk_sample(save_dir, neighbor_dict, node_num, batch_size, times_per_node, max_length_path):
    batch_num = math.ceil(node_num / batch_size)
    all_node = np.arange(node_num)
    with open(save_dir, 'w') as output:
        for b in tqdm.tqdm(range(batch_num), 'random walk sample batch num'):
            ids_arr = all_node[b * batch_size: (b + 1) * batch_size]
            walks = np.repeat(ids_arr, times_per_node)[None, :]
            for i in range(1, max_length_path):
                new_row = np.array([random.choice(neighbor_dict[j]) for j in walks[i - 1]])[None, :]
                walks = np.concatenate((walks, new_row), axis=0)
            all_str = ""
            for i in range(walks.shape[1]):
                tmp = " ".join([str(j) for j in walks[:, i]])
                all_str += tmp + "\n"
            output.write(all_str)


def segment_extraction(root_dir, random_walk_path_file, segment_batch_size, min_length_segment, max_length_segment,
                       segment_file, node2label_dict):
    path_batch = list()
    segment_output = open(root_dir + segment_file, 'w')
    with open(root_dir + random_walk_path_file, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split(' '))
            temp = [int(x) for x in temp]
            path_batch.append(temp)
            if len(path_batch) == segment_batch_size:
                walks = np.transpose(np.array(path_batch))
                path_batch.clear()
                for i in range(walks.shape[0] - 1):
                    for j in range(i + min_length_segment - 1, i + max_length_segment):
                        if j < walks.shape[0]:
                            rows = np.arange(i, j + 1)
                            segments = walks[rows]
                            all_str = ""
                            for s in range(segments.shape[1]):
                                if node2label_dict[segments[0, s]] != node2label_dict[segments[-1, s]]:
                                    continue
                                temp_segment = " ".join([str(n) for n in segments[:, s]])
                                all_str += temp_segment + "\n"
                            segment_output.write(all_str)
                        else:
                            break
    segment_output.close()


def save_segment_by_class(root_dir, dataset, segments_file, node2label_dict, segments_num_per_class):
    print('Start to save segment by class, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    start_time = time.time()

    segments_save_dir = root_dir + dataset + '/segments' + '/'
    if not os.path.exists(segments_save_dir):
        os.makedirs(segments_save_dir)

    save_segments(segments_save_dir, root_dir + dataset + '/' + segments_file, node2label_dict, segments_num_per_class)

    end_time = time.time()
    print('All cost time =', end_time - start_time, ' s')


def save_segments(segments_save_dir, segments_file, node2label_dict, segments_num_per_class):
    class_set = set()
    for n in node2label_dict:
        class_set.add(node2label_dict[n])
    class_num = len(class_set)
    for c in range(class_num):
        segments_set = set()
        with open(segments_save_dir + str(c) + '.txt', 'w') as fw:
            with open(segments_file, 'r') as fr:
                lines = fr.readlines()
                for line in lines:
                    temp = list(line.strip('\n').split(' '))
                    if int(node2label_dict[int(temp[0])]) == c:
                        temp = " ".join([str(n) for n in temp])
                        segments_set.add(temp)
                if len(segments_set) >= segments_num_per_class:
                    segments = random.sample(list(segments_set), segments_num_per_class)
                else:
                    print(len(segments_set))
                    segments = over_sampler(list(segments_set), segments_num_per_class)
                for s in segments:
                    fw.write(s + '\n')


def over_sampler(data, sample_num):
    res = list()
    while len(res) < sample_num:
        res.append(random.sample(data, 1)[0])
    return res


if __name__ == '__main__':
    root_dir = './data/'
    dataset = 'email'
    random_walk_path_file = 'random_walk_path.txt'
    segments_file = 'segments.txt'
    sample_batch_size = 100
    segment_batch_size = 100
    sample_times_per_node = 50
    max_length_path = 20
    min_length_segment = 2
    max_length_segment = 5
    segments_num_per_class = 2000
    node2label = prepare_random_walk_and_segments(root_dir, dataset, random_walk_path_file, sample_batch_size,
                                                  sample_times_per_node,
                                                  max_length_path, min_length_segment, max_length_segment,
                                                  segments_file,
                                                  segment_batch_size)
    save_segment_by_class(root_dir, dataset, segments_file, node2label, segments_num_per_class)
