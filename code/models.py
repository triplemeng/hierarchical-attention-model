import os
import re
import argparse
import numpy as np 
import pickle as pl 
import pandas as pd
import tensorflow as tf
from components import sequence, attention, BucketedDataIterator, get_sentence, visualize_sentence_format, visualize

def build_graph(
    inputs,
    revlens,
    keep_probs,
    hidden_size = 50,
    atten_size = 50,
    nclasses = 2,
    embeddings = None
    ):

    # Placeholders
    print(inputs.shape)
    print(revlens.shape)
    
    max_rev_length = int(inputs.shape[1])
    sent_length = int(inputs.shape[2])
    print(max_rev_length, sent_length)
    
    _, embedding_size = embeddings.shape
    word_rnn_inputs = tf.nn.embedding_lookup( tf.convert_to_tensor(embeddings, np.float32), inputs)
    print("word rnn inputs: "+str(word_rnn_inputs))
    word_rnn_inputs_formatted = tf.reshape(word_rnn_inputs, [-1, sent_length, embedding_size])
    print('word rnn inputs formatted: '+str(word_rnn_inputs_formatted))
    
    reuse_value = None
    
    with tf.variable_scope("word_rnn"):
        word_rnn_outputs = sequence(word_rnn_inputs_formatted, hidden_size, None)
    
    # now add the attention mech on words:
    # Attention mechanism at word level
        
    atten_inputs = tf.concat(word_rnn_outputs, 2)
    combined_hidden_size = int(atten_inputs.shape[2])

    atten_inputs = tf.nn.dropout(atten_inputs, keep_probs[0])
    with tf.variable_scope("word_atten"):
        sent_outs, alphas_words = attention(atten_inputs, atten_size)
    
    sent_outs_formatted = tf.reshape(sent_outs, [-1, max_rev_length, combined_hidden_size])
    print("sent outs formatted: "+str(sent_outs_formatted))
    sent_rnn_inputs_formatted = sent_outs_formatted
    print('sent rnn inputs formatted: '+str(sent_rnn_inputs_formatted))

    with tf.variable_scope("sent_rnn"):
        sent_rnn_outputs = sequence(sent_rnn_inputs_formatted, hidden_size, revlens)
    
    # attention at sentence level:
    sent_atten_inputs = tf.concat(sent_rnn_outputs, 2)
    sent_atten_inputs = tf.nn.dropout(sent_atten_inputs, keep_probs[1])
    
    with tf.variable_scope("sent_atten"):
        rev_outs, alphas_sents = attention(sent_atten_inputs, atten_size)

    with tf.variable_scope("out_weights1", reuse=reuse_value) as out:
        weights_out = tf.get_variable(name="output_w", dtype=tf.float32, shape=[hidden_size*2, nclasses])
        biases_out = tf.get_variable(name="output_bias", dtype=tf.float32, shape=[nclasses])
    dense = tf.matmul(rev_outs, weights_out) + biases_out
    print(dense)
    
    return dense, alphas_words, alphas_sents


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parameters for building the model.')
    parser.add_argument('-b', '--batch_size', type=int, default=512,
                   help='training batch size')
    parser.add_argument('-r', '--resume', type=bool, default=False,
                   help='pick up the latest check point and resume')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                   help='epochs for training')

    args = parser.parse_args()
    train_batch_size = args.batch_size
    resume = args.resume
    epochs = args.epochs

    working_dir = "../data/aclImdb"
    log_dir = "../logs"
    train_filename = os.path.join(working_dir, "train_df_file")
    test_filename = os.path.join(working_dir, "test_df_file")
    emb_filename = os.path.join(working_dir, "emb_matrix")
    print("load dataframe for training...")
    df_train = pd.read_pickle(train_filename)
    max_rev_length, sent_length = df_train['review'][0].shape
    print("load dataframe for testing...")
    df_test = pd.read_pickle(test_filename)
    print(df_test.shape)
    print("load embedding matrix...")
    (emb_matrix, word2index, index2word) = pl.load(open(emb_filename, "rb"))

    nclasses = 2
    y_ = tf.placeholder(tf.int32, shape=[None, nclasses])
    inputs = tf.placeholder(tf.int32, [None, max_rev_length, sent_length])
    revlens = tf.placeholder(tf.int32, [None])
    keep_probs = tf.placeholder(tf.float32, [2])

    dense, alphas_words, alphas_sents = build_graph(inputs, revlens, keep_probs, embeddings=emb_matrix, nclasses=nclasses)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=dense))
    with tf.variable_scope('optimizers', reuse=None):
        optimizer = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    y_predict = tf.argmax(dense, 1)
    correct_prediction = tf.equal(y_predict, tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    tf.summary.scalar("cost", cross_entropy)
    tf.summary.scalar("accuracy", accuracy)
    summary_op = tf.summary.merge_all()

    total_batch = int(len(df_train)/(train_batch_size))

    num_buckets = 3
    data = BucketedDataIterator(df_train, num_buckets)

    depth = nclasses
    on_value = 1
    off_value = 0

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
#         insert this snippet to restore a model:
        resume_from_epoch = -1 
        if resume:
            latest_cpt_file = tf.train.latest_checkpoint('../logs')
            print("the code pick up from lateset checkpoint file: {}".format(latest_cpt_file))
            resume_from_epoch = int(str(latest_cpt_file).split('-')[1])
            print("it resumes from previous epoch of {}".format(resume_from_epoch))
            saver.restore(sess, latest_cpt_file)
        for epoch in range(resume_from_epoch+1, resume_from_epoch+epochs+1):
            avg_cost = 0.0
            print("epoch {}".format(epoch))
            for i in range(total_batch):
                batch_data, batch_label, seqlens  = data.next_batch(train_batch_size)
                batch_label_formatted = tf.one_hot(indices=batch_label, depth=depth, on_value=on_value, off_value=off_value, axis=-1)
        
                batch_labels = sess.run(batch_label_formatted)
                feed = {inputs: batch_data, revlens: seqlens, y_: batch_labels, keep_probs: [0.9, 0.9]}
                _, c, summary_in_batch_train = sess.run([optimizer, cross_entropy, summary_op], feed_dict=feed)
                avg_cost += c/total_batch
                train_writer.add_summary(summary_in_batch_train, epoch*total_batch + i)
            saver.save(sess, os.path.join(log_dir, "model.ckpt"), epoch, write_meta_graph=False)
            print("avg cost in the training phase epoch {}: {}".format(epoch, avg_cost))

        print("evaluating...")

        x_test = np.asarray(df_test['review'].tolist())
        y_test = df_test['label'].values.tolist()
        test_review_lens = df_test['length'].tolist()
        test_batch_size = 1000
        total_batch2 = int(len(x_test)/(test_batch_size))
        avg_accu = 0.0

        for i in range(total_batch2):
        #for i in range(0):
            batch_x = x_test[i*test_batch_size:(i+1)*test_batch_size]
            batch_y = y_test[i*test_batch_size:(i+1)*test_batch_size]
            batch_seqlen = test_review_lens[i*test_batch_size:(i+1)*test_batch_size]
            
            batch_label_formatted2 =tf.one_hot(indices=batch_y, depth=depth, on_value=on_value, off_value=off_value, axis=-1)
    
            batch_labels2 = sess.run(batch_label_formatted2)
            feed = {inputs: batch_x, revlens: batch_seqlen, y_: batch_labels2, keep_probs: [1.0, 1.0]}
            accu  = sess.run(accuracy, feed_dict=feed)
            avg_accu += 1.0*accu/total_batch2

        print("prediction accuracy on test set is {}".format(avg_accu))
        visual_sample_index = 99
        visualize(sess, inputs, revlens, max_rev_length, keep_probs, index2word, alphas_words, alphas_sents,  x_test, y_test, y_predict, visual_sample_index)
