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

if __name__ == '__main__':
    pass 
