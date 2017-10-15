import os
import argparse
import numpy as np 
import pickle as pl 
import pandas as pd
import tensorflow as tf
from components import sequence, attention, BucketedDataIterator, get_sentence

def build_graph(
    inputs,
    revlens,
    hidden_size = 64,
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
    visual_dir = "../data/visualization"
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

    words_visual_file = os.path.join(visual_dir, "words_in_sentence_visualization.html")
    sents_visual_file = os.path.join(visual_dir, "sents_in_review_visualization.html")
    nclasses = 2
    y_ = tf.placeholder(tf.int32, shape=[None, nclasses])
    inputs = tf.placeholder(tf.int32, [None, max_rev_length, sent_length])
    revlens = tf.placeholder(tf.int32, [None])
    dense, alphas_words, alphas_sents =build_graph(inputs, revlens, embeddings=emb_matrix, nclasses=nclasses)
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

    num_buckets = 4
    data = BucketedDataIterator(df_train, num_buckets)

    depth = nclasses
    on_value = 1
    off_value = 0

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
#         insert this snippet to restore a model:
        if resume:
            saver.restore(sess, tf.train.latest_checkpoint('../logs'))
        for epoch in range(0, epochs):
            avg_cost = 0.0
            print("epoch {}".format(epoch))
            for i in range(total_batch):
                batch_data, batch_label, seqlens  = data.next_batch(train_batch_size)
                batch_label_formatted = tf.one_hot(indices=batch_label, depth=depth, on_value=on_value, off_value=off_value, axis=-1)
        
                batch_labels = sess.run(batch_label_formatted)
                feed = {inputs: batch_data, revlens: seqlens, y_: batch_labels}
                _, c, summary_in_batch_train = sess.run([optimizer, cross_entropy, summary_op], feed_dict=feed)
                avg_cost += c/total_batch
                train_writer.add_summary(summary_in_batch_train, epoch*total_batch + i)
            saver.save(sess, os.path.join(log_dir, "model.ckpt"), epoch, write_meta_graph=False)
            print("avg cost in the training phase epoch {}: {}".format(epoch, avg_cost))

        print("evaluating...")

        x_test = np.asarray(df_test['review'].tolist())
        y_test = df_test['label'].values.tolist()
        test_review_lens = df_test['length'].tolist()
        test_batch_size = 50
        total_batch2 = int(len(x_test)/(test_batch_size))
        avg_accu = 0.0

        #for i in range(total_batch2):
        for i in range(1):
            batch_x = x_test[i*test_batch_size:(i+1)*test_batch_size]
            batch_y = y_test[i*test_batch_size:(i+1)*test_batch_size]
            batch_seqlen = test_review_lens[i*test_batch_size:(i+1)*test_batch_size]
            
            batch_label_formatted2 =tf.one_hot(indices=batch_y, depth=depth, on_value=on_value, off_value=off_value, axis=-1)
    
            batch_labels2 = sess.run(batch_label_formatted2)
            feed = {inputs: batch_x, revlens: batch_seqlen, y_: batch_labels2}
            accu  = sess.run(accuracy, feed_dict=feed)
            avg_accu += accu/total_batch2

        print("prediction accuracy on test set is {}".format(avg_accu))

        # visualization

        x_test_sample = x_test[0:1]
        alphas_words_test, alphas_sents_test = sess.run([alphas_words, alphas_sents], feed_dict={inputs:x_test_sample, revlens: [max_rev_length]}) 
        words = get_sentence(index2word, x_test_sample[0][3]).split()

        # visualize words in a sentence
        with open(words_visual_file,  "w") as html_file:
            for word, alpha in zip(words, alphas_words_test[3] / alphas_words_test[3].max()):
                html_file.write('<font style="background: rgba(255, 255, 0, %f)">%s</font>\n' % (alpha, word))

        # visualize sentences in a review
        sents = [get_sentence(index2word, x_test_sample[0][i]) for i in range(max_rev_length)]
        with open(sents_visual_file, "w") as html_file:
            for sent, alpha in zip(sents, alphas_sents_test[0] / alphas_sents_test[0].max()):
                if len(set(sent.split(' '))) == 1:
                    continue  
                visual_sent = sent
                html_file.write('<font style="background: rgba(255, 0, 0, %f)">&nbsp&nbsp&nbsp&nbsp&nbsp</font>%s<br>' % (alpha, visual_sent))
