# hierarchical-attention-model
hierarchical attention model

This repo implemented "Hierarchical Attention Networks for Document Classification" by Zichao Yang et.al. 

It benefited greatly from two resources: the foremost one is Ilya Ivanov's repo on hierarchical attention model: https://github.com/ilivans/tf-rnn-attention .  I followed the way Ilya did in implementing attention and visualization. The difference is that in this implementation it also has sentence-level attention. The other one is r2rt's code in generating batch samples for dynamic rnns: https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html

The code was experimented on imdb data (with only positive and negative labels) 

To prepare the data:

1. bash data.sh

It will download the raw imdb data and uncompress it to ./data/aclImdb folder with positive samples under 'pos' and negative ones under 'neg' subdirectories. 

2. pretrain word embeddings

I've tried both training word embeddings in a supervised fashion and in an unsupervised(pretaining) fashion. The former took more computational resources and also prone to overfitting. 

    cd ./code
    python gen_word_embeddings.py
    (By default, the embedding size is 50.)

3. preprocess reviews

Preprocess reviews: each review will be composed of max_rev_len sentences. If the original review is longer than that, we truncate it, and if shorter than that, we append empty sentences to it. And each sentence will be composed of sent_length words. If the original sentence is longer than that, we truncate it, and if shorter, we append the word of 'STOP' to it. Also, we keep track of the actual number of sentences each review contains.
We directly read in pre-trained embeddings. Here we take the default dictionary size to be 10000. The words are indexed from 1 to 10000.
Any words that are not included in the dictionary are makred as "UNK", and the index for "UNK" is 0. The index for "STOP" is 10001.

    python preprocess_reviews.py --sent_length 70 --max_rev_length 15

4. run the model

Train the model and evaluate it on the test set.
    
    python models.py

    --batch_size batch size (default 512)

    --resume pick up latest checkpoint and resume running

    --epoches epoches (default 10)

Note:
   if you just want to build the model and evaluate it, you can just run it in the default mode:
   
   python models.py

   if you want to pick up latetest check point and resume the computation:
   
   python models.py -r True -e 5
   (-e 5 means another 5 more epochs after the check point)

   if you only want to use the latest check point to do the evaluation:
   
   python models.py -r True -e 0

5. visualization

The visualization module is embeded in models.py. A few examples are contained in the visualization folder. Use any html reader to display the results.
