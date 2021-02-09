import configparser
import model_1
import numpy as np
import tensorflow as tf
import datetime
import os
import utils
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def train(root_dir, dataset, gnn_inner_dim, dropout, train_max_epoch_num, patience, test_batch_size, lr, l2_coef,
          segment_num_per_class, segment_length, alpha, beta, gamma):
    options = locals().copy()

    node_file = root_dir + 'graph.node'
    edge_file = root_dir + 'graph.edge'
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, label2index, node2label, \
    head_class = utils.load_dataset(dataset, node_file, edge_file)
    segments_idx, segments_mask, segments_set = utils.load_segments(root_dir, label2index, segment_num_per_class,
                                                                    segment_length)
    features = features.A
    neis_nums = adj.sum(axis=1)
    # neis_nums = np.squeeze(neis_nums.A)
    neis_nums = np.squeeze(neis_nums)
    print('Start to load adj-self information, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    if os.path.exists(root_dir + 'adj_self.npy'):
        adj_self = np.load(root_dir + 'adj_self.npy', allow_pickle=True)
        adj_self_nor = np.load(root_dir + 'adj_self_nor.npy', allow_pickle=True)
    else:
        # adj_self, adj_self_nor = processTools.preprocess_adj_normalization_sparse(adj)
        adj_self, adj_self_nor = utils.preprocess_adj_normalization(adj)
        np.save(root_dir + 'adj_self.npy', adj_self)
        np.save(root_dir + 'adj_self_nor.npy', adj_self_nor)

    nodes_num = np.shape(neis_nums)[0]
    max_degree = int(np.max(neis_nums))
    max_degree_self = max_degree + 1
    options['class_num'] = class_num = y_train.shape[1]
    print('class_num: %d' % options['class_num'])
    options['feature_num'] = features.shape[1]

    neis, neis_mask, neis_mask_nor, neighboursDict = utils.processNodeInfo(adj_self, adj_self_nor, nodes_num)
    del adj
    del adj_self
    del adj_self_nor
    labels_combine = y_train + y_val + y_test
    labels_combine_sum = np.sum(labels_combine, axis=0)  # shape=(7,)
    max_index = np.argmax(labels_combine_sum)
    mi_f1_labels = [i for i in range(class_num) if i != max_index]


    feature_tensor = tf.placeholder(dtype=tf.float32, shape=(None, None), name='feature_tensor')
    # feature_tensor = tf.convert_to_tensor(features, dtype=tf.float32, name='feature_tensor')
    head_class_tensor = tf.convert_to_tensor(head_class, dtype=tf.int32, name='head_class_tensor')
    dropout_tensor = tf.placeholder(dtype=tf.float32, shape=(), name='dropout_tensor')
    eval_node_tensor = tf.placeholder(dtype=tf.int32, shape=(None,), name='eval_node_tensor')
    label_tensor = tf.placeholder(dtype=tf.float32, shape=(None, options['class_num']),
                                  name='label_tensor')
    adj_2_tensor = tf.placeholder(dtype=tf.int32, shape=(None, None), name='adj_2_tensor')
    mask_2_tensor = tf.placeholder(dtype=tf.float32, shape=(None, None), name='mask_2_tensor')
    adj_1_tensor = tf.placeholder(dtype=tf.int32, shape=(None, None), name='adj_1_tensor')
    mask_1_tensor = tf.placeholder(dtype=tf.float32, shape=(None, None), name='mask_1_tensor')
    segments_idx_tensor = tf.placeholder(dtype=tf.int32, shape=(None, None, None), name='segments_idx')
    segments_mask_tensor = tf.placeholder(dtype=tf.float32, shape=(None, None, None), name='segments_mask')
    node_type_idx_tensor = tf.placeholder(dtype=tf.int32, shape=(None, None), name='node_type_idx')
    node_type_mask_tensor = tf.placeholder(dtype=tf.float32, shape=(None, None), name='node_type_mask')

    train_op, gvs_tensor, pred_tensor, loss_tensor, acc_tensor, train_loss = model_1.build_model(options,
                                                                                                 feature_tensor,
                                                                                                 head_class_tensor,
                                                                                                 adj_2_tensor,
                                                                                                 mask_2_tensor,
                                                                                                 adj_1_tensor,
                                                                                                 mask_1_tensor,
                                                                                                 eval_node_tensor,
                                                                                                 label_tensor,
                                                                                                 dropout_tensor,
                                                                                                 segments_idx_tensor,
                                                                                                 segments_mask_tensor,
                                                                                                 node_type_idx_tensor,
                                                                                                 node_type_mask_tensor)

    saver = tf.train.Saver()
    cur_path = os.getcwd()
    checkpt_file = cur_path + "/modelsSave/model_save_" + dataset + ".ckpt"
    if not os.path.exists(cur_path + "/modelsSave/"):
        os.makedirs(cur_path + "/modelsSave/")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    acc_max = 0.0
    loss_min = 1000.0
    acc_early_stop = 0.0
    loss_early_stop = 0.0
    epoch_early_stop = 0
    curr_step = 0

    with tf.Session(config=config) as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        train_node_ids = np.array([i for i in range(nodes_num) if train_mask[i] == 1])
        train_node_ids_all = np.array([i for i in range(nodes_num) if (train_mask[i] == 1 or i in segments_set)])
        val_node_ids = np.array([i for i in range(nodes_num) if val_mask[i] == 1])
        test_node_ids = np.array([i for i in range(nodes_num) if test_mask[i] == 1])

        y_train_only = y_train[train_node_ids]
        y_val_only = y_val[val_node_ids]
        y_test_only = y_test[test_node_ids]


        train_adj_2, train_mask_2, train_mask_2_nor, train_adj_1, train_mask_1, train_mask_1_nor, \
        node_type_idx_train, node_type_mask_train, eval_node_dict = utils.prepareDataset_bigdata(nodes_num,
                                                                                                 max_degree_self,
                                                                                                 train_node_ids_all,
                                                                                                 train_node_ids,
                                                                                                 node2label,
                                                                                                 neis,
                                                                                                 neis_mask,
                                                                                                 neis_mask_nor)
        segments_idx_train, train_node_ids_idx = utils.re_idx_segment_node(segments_idx, segments_mask, train_node_ids,
                                                                           eval_node_dict)

        val_batches_eval = utils.get_val_test_minibatches_idx(val_node_ids.shape[0],
                                                              test_batch_size)
        test_batches_eval = utils.get_val_test_minibatches_idx(test_node_ids.shape[0],
                                                               test_batch_size)

        val_dataset = []
        for item in range(len(val_batches_eval)):
            batch_eval_array = np.array(val_batches_eval[item])
            val_eval_node_ids_batch = val_node_ids[batch_eval_array]
            val_node_ids_batch = np.array(
                [i for i in range(nodes_num) if (i in segments_set or i in set(val_eval_node_ids_batch))])
            val_batch_adj_2, val_batch_mask_2, val_batch_mask_2_nor, val_batch_adj_1, val_batch_mask_1, \
            val_batch_mask_1_nor, node_type_idx_val_batch, \
            node_type_mask_val_batch, eval_node_dict = utils.prepareDataset_bigdata(
                nodes_num, max_degree_self, val_node_ids_batch, val_eval_node_ids_batch, node2label, neis, neis_mask,
                neis_mask_nor)

            segments_idx_val_batch, val_eval_node_idx = utils.re_idx_segment_node(segments_idx, segments_mask,
                                                                                  val_eval_node_ids_batch,
                                                                                  eval_node_dict)
            val_tuple = (val_batch_adj_2, val_batch_mask_2, val_batch_mask_2_nor, val_batch_adj_1, val_batch_mask_1,
                         val_batch_mask_1_nor, val_eval_node_idx, y_val_only[batch_eval_array],
                         segments_idx_val_batch, node_type_idx_val_batch, node_type_mask_val_batch)
            val_dataset.append(val_tuple)
        test_dataset = []
        for item in range(len(test_batches_eval)):
            batch_eval_array = np.array(test_batches_eval[item])
            test_eval_node_ids_batch = test_node_ids[batch_eval_array]
            test_node_ids_batch = np.array(
                [i for i in range(nodes_num) if (i in segments_set or i in set(test_eval_node_ids_batch))])

            test_batch_adj_2, test_batch_mask_2, test_batch_mask_2_nor, test_batch_adj_1, test_batch_mask_1, \
            test_batch_mask_1_nor, node_type_idx_test_batch, \
            node_type_mask_test_batch, eval_node_dict = utils.prepareDataset_bigdata(
                nodes_num, max_degree_self, test_node_ids_batch, test_eval_node_ids_batch, node2label, neis, neis_mask,
                neis_mask_nor)
            segments_idx_test_batch, test_eval_node_idx = utils.re_idx_segment_node(segments_idx, segments_mask,
                                                                                    test_eval_node_ids_batch,
                                                                                    eval_node_dict)
            test_tuple = (
                test_batch_adj_2, test_batch_mask_2, test_batch_mask_2_nor, test_batch_adj_1, test_batch_mask_1,
                test_batch_mask_1_nor, test_eval_node_idx, y_test_only[batch_eval_array],
                segments_idx_test_batch, node_type_idx_test_batch, node_type_mask_test_batch)
            test_dataset.append(test_tuple)

        # 下面进行循环
        for epoch in range(train_max_epoch_num):
            print('running')
            _, train_loss_value, train_acc_value = sess.run([train_op, train_loss, acc_tensor],
                                                            feed_dict={feature_tensor: features,
                                                                       adj_2_tensor: train_adj_2,
                                                                       mask_2_tensor: train_mask_2_nor,
                                                                       adj_1_tensor: train_adj_1,
                                                                       mask_1_tensor: train_mask_1_nor,
                                                                       eval_node_tensor: train_node_ids_idx,
                                                                       label_tensor: y_train_only,
                                                                       dropout_tensor: dropout,
                                                                       segments_idx_tensor: segments_idx_train,
                                                                       segments_mask_tensor: segments_mask,
                                                                       node_type_idx_tensor: node_type_idx_train,
                                                                       node_type_mask_tensor: node_type_mask_train
                                                                       })

            val_preds = []
            val_losses = []
            for val_batch in val_dataset:
                val_batch_adj_2, val_batch_mask_2, val_batch_mask_2_nor, val_batch_adj_1, \
                val_batch_mask_1, val_batch_mask_1_nor, val_node_batch, y_val_batch, \
                segments_idx_val, node_type_idx_val, node_type_mask_val = val_batch
                val_pred_value_batch, val_loss_value_batch, val_acc_value_batch = sess.run(
                    [pred_tensor, loss_tensor, acc_tensor],
                    feed_dict={feature_tensor: features,
                               adj_2_tensor: val_batch_adj_2,
                               mask_2_tensor: val_batch_mask_2_nor,
                               adj_1_tensor: val_batch_adj_1,
                               mask_1_tensor: val_batch_mask_1_nor,
                               eval_node_tensor: val_node_batch,
                               label_tensor: y_val_batch,
                               dropout_tensor: 0.0,
                               segments_idx_tensor: segments_idx_val,
                               segments_mask_tensor: segments_mask,
                               node_type_idx_tensor: node_type_idx_val,
                               node_type_mask_tensor: node_type_mask_val})
                val_preds.append(val_pred_value_batch)
                val_losses.append(val_loss_value_batch)

            val_preds_array = np.concatenate(val_preds)
            y_true_val = np.argmax(y_val_only, axis=-1)
            y_pred_val = np.argmax(val_preds_array, axis=-1)
            val_acc_value = accuracy_score(y_true_val, y_pred_val)
            val_loss_value = float(np.mean(val_losses))

            # 下面进行test
            test_preds = []
            test_losses = []
            for test_batch in test_dataset:
                test_batch_adj_2, test_batch_mask_2, test_batch_mask_2_nor, test_batch_adj_1, \
                test_batch_mask_1, test_batch_mask_1_nor, test_node_batch, y_test_batch, \
                segments_idx_test, node_type_idx_test, node_type_mask_test = test_batch
                test_pred_value_batch, test_loss_value_batch, test_acc_value_batch = sess.run(
                    [pred_tensor, loss_tensor, acc_tensor],
                    feed_dict={feature_tensor: features,
                               adj_2_tensor: test_batch_adj_2,
                               mask_2_tensor: test_batch_mask_2_nor,
                               adj_1_tensor: test_batch_adj_1,
                               mask_1_tensor: test_batch_mask_1_nor,
                               eval_node_tensor: test_node_batch,
                               label_tensor: y_test_batch,
                               dropout_tensor: 0.0,
                               segments_idx_tensor: segments_idx_test,
                               segments_mask_tensor: segments_mask,
                               node_type_idx_tensor: node_type_idx_test,
                               node_type_mask_tensor: node_type_mask_test})
                test_preds.append(test_pred_value_batch)
                test_losses.append(test_loss_value_batch)

            test_preds_array = np.concatenate(test_preds)
            y_true_test = np.argmax(y_test_only, axis=-1)
            y_pred_test = np.argmax(test_preds_array, axis=-1)
            test_acc_value = accuracy_score(y_true_test, y_pred_test)
            test_loss_value = float(np.mean(test_losses))

            print(
                'Epoch: %d | Train: loss = %.5f, acc = %.5f | '
                'Val: loss = %.5f, acc = %.5f | Test: loss = %.5f, acc = %.5f' % (
                    epoch, train_loss_value, train_acc_value, val_loss_value, val_acc_value, test_loss_value,
                    test_acc_value))

            if val_acc_value >= acc_max or val_loss_value <= loss_min:
                if val_acc_value >= acc_max and val_loss_value <= loss_min:
                    # 将记录结果的变量进行赋值
                    acc_early_stop = val_acc_value
                    loss_early_stop = val_loss_value
                    epoch_early_stop = epoch
                    # 将现有参数进行保存
                    saver.save(sess, checkpt_file)
                    print('------------------------------------------------------------------------------------')
                # 将现有结果进行记录
                acc_max = np.max((val_acc_value, acc_max))
                loss_min = np.min((val_loss_value, loss_min))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step >= patience and epoch > 800:
                    print('Early stop model validation loss: ', loss_early_stop, ', accuracy: ', acc_early_stop,
                          ', epoch: ', epoch_early_stop)
                    break

        test_preds = []
        test_losses = []
        for test_batch in test_dataset:
            test_batch_adj_2, test_batch_mask_2, test_batch_mask_2_nor, test_batch_adj_1, \
            test_batch_mask_1, test_batch_mask_1_nor, test_node_batch, y_test_batch, \
            segments_idx_test, node_type_idx_test, node_type_mask_test = test_batch
            test_pred_value_batch, test_loss_value_batch, test_acc_value_batch = sess.run(
                [pred_tensor, loss_tensor, acc_tensor],
                feed_dict={feature_tensor: features,
                           adj_2_tensor: test_batch_adj_2,
                           mask_2_tensor: test_batch_mask_2_nor,
                           adj_1_tensor: test_batch_adj_1,
                           mask_1_tensor: test_batch_mask_1_nor,
                           eval_node_tensor: test_node_batch,
                           label_tensor: y_test_batch,
                           dropout_tensor: 0.0,
                           segments_idx_tensor: segments_idx_test,
                           segments_mask_tensor: segments_mask,
                           node_type_idx_tensor: node_type_idx_test,
                           node_type_mask_tensor: node_type_mask_test})
            test_preds.append(test_pred_value_batch)
            test_losses.append(test_loss_value_batch)

        test_preds_array = np.concatenate(test_preds)
        y_true_test = np.argmax(y_test_only, axis=-1)
        y_pred_test = np.argmax(test_preds_array, axis=-1)
        c_m = confusion_matrix(y_true_test, y_pred_test)
        test_acc_value = accuracy_score(y_true_test, y_pred_test)

        # test_losses_array = np.concatenate(test_losses)
        test_loss_value = np.mean(test_losses)
        test_micro_f1, test_macro_f1 = utils.micro_macro_f1_removeMiLabels(y_true_test, y_pred_test,
                                                                           mi_f1_labels)

        # 将结果输出
        print('----------------------------------------------------------------------------------')
        print('Train early stop epoch == ', epoch_early_stop)
        print('End train, ', 'mi-f1 == ', test_micro_f1)
        print('End train, ', 'ma-f1 == ', test_macro_f1)
        print('End train, Test acc == ', test_acc_value)
        print('----------------------------------------------------------------------------------')

    return test_micro_f1, test_macro_f1, test_acc_value


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

    root_dir = config.get('DEFAULT', 'root_dir')
    dataset = config.get('DEFAULT', 'dataset')
    root_dir = root_dir + dataset + '/'
    gpu = config.get('DEFAULT', 'gpu_id')

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    gnn_inner_dim = config.getint('DEFAULT', 'gnn_inner_dim')
    dropout = config.getfloat('DEFAULT', 'dropout')
    lr = config.getfloat('DEFAULT', 'lr')
    l2_coef = config.getfloat('DEFAULT', 'l2_coef')
    train_epoch_num = config.getint('DEFAULT', 'train_epoch_num')
    segment_num_per_class = config.getint('DEFAULT', 'segment_num_per_class')
    segment_length = config.getint('DEFAULT', 'segment_length')
    alpha = config.getfloat('DEFAULT', 'alpha')
    beta = config.getfloat('DEFAULT', 'beta')
    gamma = config.getfloat('DEFAULT', 'gamma')

    patience = config.getint('DEFAULT', 'patience')
    test_batch_size = config.getint('DEFAULT', 'test_batch_size')
    num = 30
    micro_f1 = 0
    macro_f1 = 0
    acc_value = 0
    micro_f1_res = list()
    macro_f1_res = list()
    acc_value_res = list()
    for i in range(num):
        tf.reset_default_graph()
        test_micro_f1, test_macro_f1, test_acc_value = train(root_dir, dataset, gnn_inner_dim, dropout, train_epoch_num,
                                                             patience, test_batch_size, lr, l2_coef,
                                                             segment_num_per_class, segment_length, alpha, beta, gamma)
        micro_f1 += test_micro_f1
        macro_f1 += test_macro_f1
        acc_value += test_acc_value
        micro_f1_res.append(test_micro_f1)
        macro_f1_res.append(test_macro_f1)
        acc_value_res.append(test_acc_value)
    print('final results: ')
    print(micro_f1 / num)
    print(macro_f1 / num)
    print(acc_value / num)
    print(micro_f1_res)
    print(macro_f1_res)
    print(acc_value_res)
