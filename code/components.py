import os, re
import tensorflow as tf
import numpy as np

class BucketedDataIterator():
    ## bucketed data iterator uses R2RT's implementation(https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html)
    def __init__(self, df, num_buckets = 3):
        df = df.sort_values('length').reset_index(drop=True)
        self.size = int(len(df) / num_buckets)
        self.dfs = []
        for bucket in range(num_buckets):
            self.dfs.append(df.iloc[bucket*self.size: (bucket+1)*self.size])
        self.num_buckets = num_buckets

        # cursor[i] will be the cursor for the ith bucket
        self.cursor = np.array([0] * num_buckets)
        self.shuffle()

        self.epochs = 0

    def shuffle(self):
        #sorts dataframe by sequence length, but keeps it random within the same length
        for i in range(self.num_buckets):
            self.dfs[i] = self.dfs[i].sample(frac=1).reset_index(drop=True)
            self.cursor[i] = 0

    def next_batch(self, n):
        if np.any(self.cursor+n > self.size):
            self.epochs += 1
            self.shuffle()

        i = np.random.randint(0, self.num_buckets)

        res = self.dfs[i].iloc[self.cursor[i]:self.cursor[i]+n]
        self.cursor[i] += n
        return np.asarray(res['review'].tolist()), res['label'].tolist(), res['length'].tolist()

def get_sentence(vocabulary_inv, sen_index):
    return ' '.join([vocabulary_inv[index] for index in sen_index])

def sequence(rnn_inputs, hidden_size, seq_lens):
    cell_fw = tf.nn.rnn_cell.GRUCell(hidden_size)
    print('build fw cell: '+str(cell_fw))
    cell_bw = tf.nn.rnn_cell.GRUCell(hidden_size)
    print('build bw cell: '+str(cell_bw))
    rnn_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                               cell_bw, 
                                                               inputs=rnn_inputs, 
                                                               sequence_length=seq_lens,
                                                               dtype=tf.float32
                                                               )
    print('rnn outputs: '+str(rnn_outputs))   
    print('final state: '+str(final_state))
   
    return rnn_outputs
   
def attention(atten_inputs, atten_size):
    ## attention mechanism uses Ilya Ivanov's implementation(https://github.com/ilivans/tf-rnn-attention)
    print('attention inputs: '+str(atten_inputs))
    max_time = int(atten_inputs.shape[1])
    print("max time length: "+str(max_time))
    combined_hidden_size = int(atten_inputs.shape[2])
    print("combined hidden size: "+str(combined_hidden_size))
    W_omega = tf.Variable(tf.random_normal([combined_hidden_size, atten_size], stddev=0.1, dtype=tf.float32))
    b_omega = tf.Variable(tf.random_normal([atten_size], stddev=0.1, dtype=tf.float32))
    u_omega = tf.Variable(tf.random_normal([atten_size], stddev=0.1, dtype=tf.float32))
    
    v = tf.tanh(tf.matmul(tf.reshape(atten_inputs, [-1, combined_hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    print("v: "+str(v))
    # u_omega is the summarizing question vector
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    print("vu: "+str(vu))
    exps = tf.reshape(tf.exp(vu), [-1, max_time])
    print("exps: "+str(exps))
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
    print("alphas: "+str(alphas))
    atten_outs = tf.reduce_sum(atten_inputs * tf.reshape(alphas, [-1, max_time, 1]), 1)
    print("atten outs: "+str(atten_outs))
    return atten_outs, alphas

def visualize_sentence_format(sent):
    ## remove the trailing 'STOP' symbols from sent
    visual_sent = ' '.join(re.sub('STOP', '', sent).split()) 
    return visual_sent

def visualize(sess, inputs, revlens, max_rev_length, keep_probs, index2word, alphas_words, alphas_sents,  x_test, y_test, y_predict, visual_sample_index):
    visual_dir = "../visualization"
    # visualization
    sents_visual_file = os.path.join(visual_dir, "sents_in_review_visualization_{}.html".format(visual_sample_index))
    x_test_sample = x_test[visual_sample_index:visual_sample_index+1]
    y_test_sample = y_test[visual_sample_index:visual_sample_index+1]
    test_dict = {inputs:x_test_sample, revlens: [max_rev_length], keep_probs: [1.0, 1.0]}
    alphas_words_test, alphas_sents_test = sess.run([alphas_words, alphas_sents], feed_dict=test_dict) 
    y_test_predict = sess.run(y_predict, feed_dict=test_dict)
    print("test sample is {}".format(y_test_sample[0]))
    print("test sample is predicted as {}".format(y_test_predict[0]))
    print(alphas_words_test.shape)

    # visualize a review
    sents = [get_sentence(index2word, x_test_sample[0][i]) for i in range(max_rev_length)]
    index_sent = 0
    print("sents size is {}".format(len(sents)))
    with open(sents_visual_file, "w") as html_file:
        html_file.write('actual label: %f, predicted label: %f<br>' % (y_test_sample[0], y_test_predict[0]))
        for sent, alpha in zip(sents, alphas_sents_test[0] / alphas_sents_test[0].max()):
            if len(set(sent.split(' '))) == 1:
                index_sent += 1
                continue  
            visual_sent = visualize_sentence_format(sent)
            # display each sent's importance by color
            html_file.write('<font style="background: rgba(255, 0, 0, %f)">&nbsp&nbsp&nbsp&nbsp&nbsp</font>' % (alpha))
            visual_words = visual_sent.split()
            visual_words_alphas = alphas_words_test[index_sent][:len(visual_words)]
            # for each sent, display its word importance by color
            for word, alpha_w in zip(visual_words, visual_words_alphas / visual_words_alphas.max()):
                html_file.write('<font style="background: rgba(255, 255, 0, %f)">%s </font>' % (alpha_w, word))
            html_file.write('<br>')
            index_sent += 1

if __name__ == '__main__':
    pass 
