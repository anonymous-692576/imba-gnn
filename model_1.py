import tensorflow as tf
import numpy as np


def build_model(options, feature_tensor, head_class_tensor, adj_2_tensor, mask_2_tensor, adj_1_tensor, mask_1_tensor,
                eval_node_tensor, label_tensor, dropout_tensor, segments_idx, segments_mask, node_type_idx,
                node_type_mask):
    model_variables, gnn_variables, segment_variables = init_variables(options)
    gnn_pred = gnn_forward(feature_tensor, adj_2_tensor, mask_2_tensor, adj_1_tensor, mask_1_tensor, dropout_tensor,
                           gnn_variables)

    segments_embedding = tf.nn.embedding_lookup(gnn_pred, segments_idx)

    s_c = tf.reduce_mean(gru_layer(segments_embedding, segments_mask, options['segment_length'], segment_variables),
                         axis=1)

    p_c_all = tf.einsum('ijk,ij->ijk', tf.nn.embedding_lookup(gnn_pred, node_type_idx), node_type_mask)
    p_c = tf.divide(tf.reduce_sum(p_c_all, axis=1),
                    tf.reduce_sum(node_type_mask, axis=1, keep_dims=True))
    gamma_c = tf.nn.leaky_relu(tf.matmul(p_c, model_variables['w_gamma']) + model_variables['b_gamma'])
    beta_c = tf.nn.leaky_relu(tf.matmul(p_c, model_variables['w_beta']) + model_variables['b_beta'])
    g_c = tf.multiply(gamma_c + 1, model_variables['g']) + beta_c
    m_c = s_c - p_c
    gnn_pred_eval = tf.nn.embedding_lookup(gnn_pred, eval_node_tensor)
    model_loss = tf.nn.softmax_cross_entropy_with_logits(logits=gnn_pred_eval, labels=label_tensor)
    model_acc = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(gnn_pred_eval, axis=1), axis=1),
                                            tf.argmax(label_tensor, axis=1))  # acc

    model_loss = tf.reduce_mean(model_loss)
    model_acc = tf.reduce_mean(model_acc)

    tail_class_tensor = tf.convert_to_tensor(np.ones(shape=(options['class_num'],)), dtype=np.int32) - head_class_tensor
    rep_label_all = tf.one_hot(np.arange(options['class_num'], dtype=np.int32), depth=options['class_num'],
                               dtype=np.float32)
    rep_logits = tf.nn.leaky_relu(
        tf.matmul(model_variables['w_reg'], tf.nn.embedding_lookup(m_c - g_c, tail_class_tensor)) + model_variables[
            'b_reg'])
    rep_labels = tf.nn.embedding_lookup(rep_label_all, tail_class_tensor)
    rep_loss = tf.nn.softmax_cross_entropy_with_logits(logits=rep_logits, labels=rep_labels)
    rep_loss = tf.reduce_mean(rep_loss)
    model_loss += (tf.nn.l2_loss(gnn_variables['w_1']) + tf.nn.l2_loss(gnn_variables['w_2']) + tf.nn.l2_loss(
        model_variables['g']) + tf.nn.l2_loss(model_variables['w_gamma']) + tf.nn.l2_loss(
        model_variables['w_beta']) + tf.nn.l2_loss(model_variables['b_gamma']) + tf.nn.l2_loss(
        model_variables['b_beta']) + tf.nn.l2_loss(model_variables['w_reg']) + tf.nn.l2_loss(
        model_variables['b_reg'])) * options['l2_coef']
    head_m_c = tf.nn.embedding_lookup(m_c, head_class_tensor)
    head_g_c = tf.nn.embedding_lookup(g_c, head_class_tensor)
    model_train_loss = model_loss + tf.losses.mean_squared_error(head_m_c, head_g_c) * options['alpha'] + rep_loss * \
                       options['beta']

    optimizer = tf.train.AdamOptimizer(options['lr'], name='train_update_op')
    gvs = optimizer.compute_gradients(model_train_loss)
    train_update_op = optimizer.apply_gradients(gvs)

    return train_update_op, gvs, gnn_pred_eval, model_loss, model_acc, model_train_loss


def init_variables(options):
    model_variables = dict()
    gnn_variables = dict()
    segment_variables = dict()
    var_initializer = tf.contrib.layers.xavier_initializer()
    gnn_variables['w_1'] = tf.Variable(tf.random_uniform([options['gnn_inner_dim'], options['class_num']], -0.01, 0.01),
                                       dtype=tf.float32, name="w_1")
    gnn_variables['w_2'] = tf.Variable(
        tf.random_uniform([options['feature_num'], options['gnn_inner_dim']], -0.01, 0.01), dtype=tf.float32,
        name="w_2")
    model_variables['g'] = tf.get_variable('g', [options['class_num'], options['class_num']],
                                           initializer=var_initializer)
    model_variables['w_gamma'] = tf.get_variable('w_gamma', [options['class_num'], options['class_num']],
                                                 initializer=var_initializer)
    model_variables['w_beta'] = tf.get_variable('w_beta', [options['class_num'], options['class_num']],
                                                initializer=var_initializer)
    model_variables['b_gamma'] = tf.get_variable('b_gamma', [options['class_num'], ],
                                                 initializer=var_initializer)
    model_variables['b_beta'] = tf.get_variable('b_beta', [options['class_num'], ],
                                                initializer=var_initializer)
    model_variables['w_reg'] = tf.get_variable('w_reg', [options['class_num'], options['class_num']],
                                               initializer=var_initializer)
    model_variables['b_reg'] = tf.get_variable('b_reg', [options['class_num'], ],
                                               initializer=var_initializer)
    segment_variables['w_z'] = tf.get_variable('w_z', [options['class_num'], options['class_num']],
                                               initializer=var_initializer)
    segment_variables['w_r'] = tf.get_variable('w_r', [options['class_num'], options['class_num']],
                                               initializer=var_initializer)
    segment_variables['w_h'] = tf.get_variable('w_h', [options['class_num'], options['class_num']],
                                               initializer=var_initializer)

    segment_variables['u_z'] = tf.get_variable('u_z', [options['class_num'], options['class_num']],
                                               initializer=var_initializer)
    segment_variables['u_r'] = tf.get_variable('u_r', [options['class_num'], options['class_num']],
                                               initializer=var_initializer)
    segment_variables['u_h'] = tf.get_variable('u_h', [options['class_num'], options['class_num']],
                                               initializer=var_initializer)

    segment_variables['b_z'] = tf.get_variable('b_z', [options['class_num'], ], initializer=var_initializer)
    segment_variables['b_r'] = tf.get_variable('b_r', [options['class_num'], ], initializer=var_initializer)
    segment_variables['b_h'] = tf.get_variable('b_h', [options['class_num'], ], initializer=var_initializer)
    return model_variables, gnn_variables, segment_variables


def gnn_forward(feature_tensor, adj_2_tensor, mask_2_tensor, adj_1_tensor, mask_1_tensor, dropout_tensor,
                variables):
    h_2 = gnn_layer(feature_tensor, adj_2_tensor, mask_2_tensor, variables['w_2'], dropout_tensor)
    h_1 = gnn_layer(h_2, adj_1_tensor, mask_1_tensor, variables['w_1'], dropout_tensor)
    return h_1


def gnn_layer(emb, adj, mask, w, in_drop):
    adj = adj[:, 1:]
    mask = mask[:, 1:]

    seq = tf.nn.dropout(emb, 1.0 - in_drop)

    seq_fts = tf.matmul(seq, w)

    seq_fts = tf.nn.embedding_lookup(seq_fts, adj)
    seq_fts = seq_fts * mask[:, :, None]
    ret = tf.reduce_sum(seq_fts, axis=1)

    return tf.nn.elu(ret)


def gru_layer(h_gcn, segments_mask, segments_len, variables):
    h_i = tf.zeros(tf.shape(h_gcn[:, :, 0, :]),
                   dtype=tf.float32)
    for j in range(segments_len):
        i = segments_len - j - 1
        x_i = h_gcn[:, :, i, :]
        m_i = segments_mask[:, :, i]
        h_i_new = gru_step(h_i, x_i, variables)
        h_i = h_i_new * m_i[:, :, None] + h_i * (1. - m_i[:, :, None])
    return h_i


def gru_step(h_tm1, x_t, variables):
    h_tm1_new = tf.reshape(h_tm1, [-1, tf.shape(h_tm1)[-1]])
    x_t_new = tf.reshape(x_t, [-1, tf.shape(x_t)[-1]])
    z_t = tf.sigmoid(tf.matmul(x_t_new, variables['w_z']) + tf.matmul(h_tm1_new, variables['u_z']) + variables['b_z'])
    r_t = tf.sigmoid(tf.matmul(x_t_new, variables['w_r']) + tf.matmul(h_tm1_new, variables['u_r']) + variables['b_r'])
    h_proposal = tf.tanh(
        tf.matmul(x_t_new, variables['w_h']) + tf.matmul(tf.multiply(r_t, h_tm1_new), variables['u_h']) + variables[
            'b_h'])
    h_t = tf.multiply(1. - z_t, h_tm1_new) + tf.multiply(z_t, h_proposal)
    h_t_new = tf.reshape(h_t, [tf.shape(x_t)[0], tf.shape(x_t)[1], tf.shape(x_t)[2]])

    return h_t_new
