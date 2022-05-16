import time
import numpy as np
from collections import defaultdict

from gensim.models import Word2Vec
from sklearn.cluster import KMeans

from utils.config import set_random_seed
from utils.filesystem import save_pickle, load_pickle


SOS_TOKEN = '<SOS>'
PAD_TOKEN = '<PAD>'
EOS_TOKEN = '<EOS>'
TOKENS = [SOS_TOKEN, PAD_TOKEN, EOS_TOKEN]


class Vocab:
    @classmethod
    def load(cls, config):
        path = config.path('config') / 'vocab.pkl'
        return load_pickle(path)

    def save(self, config):
        path = config.path('config')
        save_pickle(self, path / 'vocab.pkl')

    def __init__(self, config, data):
        self.config = config
        self.use_mask = config.get('use_mask')
        self.mask_freq = config.get('mask_freq')

        w2i, i2w, i2w_infreq, w2w_infreq, c2w_infreq = train_embeddings(config, data)
        
        self.w2i = w2i
        self.i2w = i2w
        self.i2w_infreq = i2w_infreq # index to cluster name
        self.w2w_infreq = w2w_infreq # word to cluster name
        self.c2w_infreq = c2w_infreq # cluster name to list of words in this cluster

        self.size = len(self.w2i)

        self.save(config)

    def get_size(self):
        return self.size

    def get_effective_size(self):
        return len(self.w2i)

    def _translate_integer(self, index):
        """
        translate an index to a word, if word is a cluster name, randomly choose one word from its member list.  
        """
        word = self.i2w[index]

        if self.c2w_infreq is not None and word in self.c2w_infreq:
            wc = int(word.split("_")[1]) # word count
            try:
                choices = [w for w in self.c2w_infreq[word] if w.count('*') == wc]
            except ValueError:
                choices = self.c2w_infreq[word]
            word  = np.random.choice(choices)
        return word

    def _translate_string(self, word):
        """
        Translate a word to index. If word belongs to a cluster, return corresponding index of cluster name.
        """
        if self.w2w_infreq is not None and word not in self.w2i:
            return self.w2i[self.w2w_infreq[word]]
        return self.w2i[word]

    def get(self, value):
        if isinstance(value, str):
            return self._translate_string(value)
        elif isinstance(value, int) or isinstance(value, np.integer):
            return self._translate_integer(value)
        raise ValueError('Value type not supported.')

    def translate(self, values):
        """
        Translate a list of words/indices to indices/words.
        """
        res = []
        for v in values:
            if v not in self.TOKEN_IDS:
                res.append(self.get(v))
            if v == self.EOS:
                break
        return res

    def append_delimiters(self, sentence):
        """
        Add SOS and EOS to sentence [word indices].
        """
        return [SOS_TOKEN] + sentence + [EOS_TOKEN]

    @property
    def EOS(self):
        return self.w2i[EOS_TOKEN]

    @property
    def PAD(self):
        return self.w2i[PAD_TOKEN]

    @property
    def SOS(self):
        return self.w2i[SOS_TOKEN]

    @property
    def TOKEN_IDS(self):
        return [self.SOS, self.EOS, self.PAD]


def calculate_frequencies(sentences):
    """
    Calculate word counts.
    """
    w2f = defaultdict(int)

    for sentence in sentences:
        for word in sentence:
            w2f[word] += 1

    return w2f


def train_embeddings(config, data):
    start = time.time()
    print("Training and clustering embeddings...", end=" ")
    
    embed_size = config.get('embed_size')
    embed_window = config.get('embed_window')
    mask_freq = config.get('mask_freq')
    use_mask = config.get('use_mask')
    
    i2w_infreq = None
    w2w_infreq = None
    c2w_infreq = None
    start_idx = len(TOKENS)

    if use_mask: # use cluster names for low freq words
        # data.fragments[i]: fragments of the i-th molecule
        sentences = [s.split(" ") for s in data.fragments] # sentences is a long list of lists.
        # sentences[i]: list of fragments.
        # first word embedding
        w2v = Word2Vec(
            sentences,
            size=embed_size,
            window=embed_window,
            min_count=1,
            negative=5,
            workers=20,
            iter=10,
            sg=1)

        vocab = w2v.wv.vocab # dictionary, word:index
        embeddings = w2v[vocab] # array of size vocab sizex embed_size

        w2f = calculate_frequencies(sentences) # dictionary, w2f['word']: frequency
        w2i = {k: v.index for (k, v) in vocab.items()} # dictionary, w2i['word']: index
        i2w = {v: k for (k, v) in w2i.items()} # dictionary, i2w[index]: 'word'
        
        infreq = [w2i[w] for (w, freq) in w2f.items() if freq <= mask_freq] #indices of words with low frequency
        i2w_infreq = {} # dictionary, i2w for infreq words/fragments, i2w_infreq[index]: e.g. 'cluster10_2' # frequency=10, count of * = 2 in word
        for inf in infreq:
            word = i2w[inf]
            i2w_infreq[inf] = f"cluster{w2f[word]}_{word.count('*')}"

        w2w_infreq = {i2w[k]: v for (k, v) in i2w_infreq.items()} # word to cluster name for low freq words
        c2w_infreq = defaultdict(list) # cluster name to list of words for low frequency words
        for word, cluster_name in w2w_infreq.items():
            c2w_infreq[cluster_name].append(word)

        # substitute infrequent words with cluster words in data
        data = [] # data is a list of lists, data[i] is a list of words
        for sentence in sentences:
            sentence_sub = []
            for word in sentence:
                if word in w2w_infreq:
                    word = w2w_infreq[word]
                sentence_sub.append(word)
            data.append(sentence_sub) 
            
    else: # not use mask
        data = [s.split(" ") for s in data.fragments] # list of lists, data[i] is a list of words
    
    # new word embeddings using new data with cluster names
    w2i = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2}

    w2v = Word2Vec(
            data,
            size=embed_size,
            window=embed_window,
            min_count=1,
            negative=5,
            workers=20,
            iter=10,
            sg=1)

    vocab = w2v.wv.vocab # vocab with frequent words and cluster names.
    w2i.update({k: v.index + start_idx for (k, v) in vocab.items()})
    i2w = {v: k for (k, v) in w2i.items()}
    
    tokens = np.random.uniform(-0.05, 0.05, size=(start_idx, embed_size)) # random embeddings for tokens
    embeddings = np.vstack([tokens, w2v[vocab]]) # add token embeddings
    # save embddings to a text file
    path = config.path('config') / f'emb_{embed_size}.dat'
    np.savetxt(path, embeddings, delimiter=",")

    end = time.time() - start
    elapsed = time.strftime("%H:%M:%S", time.gmtime(end))
    print(f'Done. Time elapsed: {elapsed}.')
    return w2i, i2w, i2w_infreq, w2w_infreq, c2w_infreq


def cluster_embeddings(config, embeddings, infrequent):
    data = embeddings.take(infrequent, axis=0)
    km = KMeans(n_clusters=config.get('num_clusters'), n_jobs=-1).fit(data)
    labels = km.labels_.tolist()
    return labels
