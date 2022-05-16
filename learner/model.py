import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
#from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class JTNNEncoder(nn.Module):

    def __init__(self, hidden_size, depth, embedding):
        super(JTNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.embedding = embedding
        self.outputNN = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU()
        )
        self.GRU = GraphGRU(hidden_size, hidden_size, depth=depth)

    def forward(self, fnode, fmess, node_graph, mess_graph, scope):
        fnode = create_var(fnode)
        fmess = create_var(fmess)
        node_graph = create_var(node_graph)
        mess_graph = create_var(mess_graph)
        messages = create_var(torch.zeros(mess_graph.size(0), self.hidden_size))

        fnode = self.embedding(fnode)
        fmess = index_select_ND(fnode, 0, fmess)
        messages = self.GRU(messages, fmess, mess_graph)

        mess_nei = index_select_ND(messages, 0, node_graph)
        node_vecs = torch.cat([fnode, mess_nei.sum(dim=1)], dim=-1)
        node_vecs = self.outputNN(node_vecs)

        max_len = max([x for _,x in scope])
        batch_vecs = []
        for st,le in scope:
            cur_vecs = node_vecs[st] #Root is the first node
            batch_vecs.append( cur_vecs )

        tree_vecs = torch.stack(batch_vecs, dim=0)
        return tree_vecs, messages

    @staticmethod
    def tensorize(tree_batch):
        node_batch = [] 
        scope = []
        for tree in tree_batch:
            scope.append( (len(node_batch), len(tree.nodes)) )
            node_batch.extend(tree.nodes)

        return JTNNEncoder.tensorize_nodes(node_batch, scope)
    
    @staticmethod
    def tensorize_nodes(node_batch, scope):
        messages,mess_dict = [None],{}
        fnode = []
        for x in node_batch:
            fnode.append(x.wid)
            for y in x.neighbors:
                mess_dict[(x.idx,y.idx)] = len(messages)
                messages.append( (x,y) )

        node_graph = [[] for i in xrange(len(node_batch))]
        mess_graph = [[] for i in xrange(len(messages))]
        fmess = [0] * len(messages)

        for x,y in messages[1:]:
            mid1 = mess_dict[(x.idx,y.idx)]
            fmess[mid1] = x.idx 
            node_graph[y.idx].append(mid1)
            for z in y.neighbors:
                if z.idx == x.idx: continue
                mid2 = mess_dict[(y.idx,z.idx)]
                mess_graph[mid2].append(mid1)

        max_len = max([len(t) for t in node_graph] + [1])
        for t in node_graph:
            pad_len = max_len - len(t)
            t.extend([0] * pad_len)

        max_len = max([len(t) for t in mess_graph] + [1])
        for t in mess_graph:
            pad_len = max_len - len(t)
            t.extend([0] * pad_len)

        mess_graph = torch.LongTensor(mess_graph)
        node_graph = torch.LongTensor(node_graph)
        fmess = torch.LongTensor(fmess)
        fnode = torch.LongTensor(fnode)
        return (fnode, fmess, node_graph, mess_graph, scope), mess_dict

class GraphGRU(nn.Module):

    def __init__(self, input_size, hidden_size, depth):
        super(GraphGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth

        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, h, x, mess_graph):
        mask = torch.ones(h.size(0), 1)
        mask[0] = 0 #first vector is padding
        mask = create_var(mask)
        for it in xrange(self.depth):
            h_nei = index_select_ND(h, 0, mess_graph)
            sum_h = h_nei.sum(dim=1)
            z_input = torch.cat([x, sum_h], dim=1)
            z = F.sigmoid(self.W_z(z_input))

            r_1 = self.W_r(x).view(-1, 1, self.hidden_size)
            r_2 = self.U_r(h_nei)
            r = F.sigmoid(r_1 + r_2)
            
            gated_h = r * h_nei
            sum_gated_h = gated_h.sum(dim=1)
            h_input = torch.cat([x, sum_gated_h], dim=1)
            pre_h = F.tanh(self.W_h(h_input))
            h = (1.0 - z) * sum_h + z * pre_h
            h = h * mask

        return h

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size,
                 hidden_size, hidden_layers, latent_size,
                 dropout, use_gpu):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.latent_size = latent_size
        self.use_gpu = use_gpu

        self.rnn = nn.GRU(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.hidden_layers,
            dropout=dropout,
            batch_first=True)

        self.rnn2mean = nn.Linear(
            in_features=self.hidden_size * self.hidden_layers,
            out_features=self.latent_size)

        self.rnn2logv = nn.Linear(
            in_features=self.hidden_size * self.hidden_layers,
            out_features=self.latent_size)

    def forward(self, inputs, embeddings, lengths):
        '''
        INPUTS:
        inputs: tensor, batch_size x sequence of indices
            Example: ( 0 for '<PAD>', 1 for '<SOS>', 2 for '<EOS>')
            tensor([[    1, 20025,    79,   660,   216],
                    [    1,  8866,  3705,    10,     0],
                    [    1,    20,     5,  1091,     0],
                    [    1,    44,  4866,     3,     0],
                    [    1,    26,   233, 13176,     0],
                    [    1,    31,     5,   356,     0],
                    [    1,     8,  8179,  1517,     0],
                    [    1,   153, 11176,    67,     0]
                    [    1,   153, 11176,    67,     0],
                    [    1,     6,   109,     3,     0],
                    [    1,    11,     4,     0,     0], device='cuda:0')
        embeddings: tensor, batch_size x L x embed_size
        lengths: list, example:  [5, 4, 4, 4, 4, 4, 4, 4, 4, 3]
        OUTPUTS:
        latent_sample: num_layers x batch x latent_size
        # mean, std: same
        '''
        batch_size = inputs.size(0) # batch first
        state = self.init_state(dim=batch_size, use_gpu=self.use_gpu) # num_layers x batch x hidden_size
        
        #print(inputs)
        #print('inputs:', inputs.size())
        #print('lengths:', lengths)
        #print('state:', state.size())
        
        # if B for batch_size, T for longest sequence length
        # input of this function: B x T
        # embeddings are sorted in decreasing order
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        # even when batch_fist=True for GRU, h_0 and h_n are still of shape num_layers x batch x hidden_size 
        _, state = self.rnn(packed, state) # input if packed variable length sequence, state: num_layers x batch x hidden_size
        #print('state:', state.size())
        
        #state = state.view(batch_size, self.hidden_size * self.hidden_layers) # num_layers x batch x hidden_size -> batch x hidden_layer * hidden_size
        #print('state:', state.size()) # batch x hidden_layer * hidden_size
        # the above is wrong, because it mix batch samples up
        state = state.transpose(1,0) # num_layers x batch x hidden_size -> batch x num_layers x hidden_size
        state = state.flatten(start_dim=1) # batch x hidden_layer * hidden_size
        
        mean = self.rnn2mean(state) #  mean: batch x latent_size
        logvar = self.rnn2logv(state) # logv:  batch x latent_size
        std = torch.exp(0.5 * logvar)
        #z = self.sample_normal(dim=batch_size) #z size: num_layers x batch x latent_size
        #latent_sample = z * std + mean #latent_sample size: num_layers x batch x latent_size
        
        # latent variables are for multiple layers used by decoder
        #print('mean:',mean.size())
        #print('logv:',logv.size())
        #print('z:',z.size())
        #print('latent_sample:',latent_sample.size())
        
        # I change the above to this, it is simpler because z is a vector instead of a vector for each layer
        z = self.sample_normal(dim=batch_size, use_gpu=self.use_gpu) #z size: batch x latent_size
        latent_sample = z * std + mean #latent_sample size: batch x latent_size
        return latent_sample, mean, logvar


    #def sample_normal(self, dim):
    #    z = torch.randn((self.hidden_layers, dim, self.latent_size))
    #    return z.cuda() if self.use_gpu else z
    
    def sample_normal(self, dim, use_gpu=False):
        z = torch.randn((dim, self.latent_size))
        return z.cuda() if use_gpu else z

    def init_state(self, dim, use_gpu=False):
        state = torch.zeros((self.hidden_layers, dim, self.hidden_size))
        return state.cuda() if use_gpu else state


class Decoder(nn.Module):
    def __init__(self, embed_size, latent_size, hidden_size,
                 hidden_layers, dropout, output_size):
        super().__init__()
        self.embed_size = embed_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.dropout = dropout

        self.rnn = nn.GRU(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.hidden_layers,
            dropout=self.dropout,
            batch_first=True)

        self.rnn2out = nn.Linear(
            in_features=hidden_size,
            out_features=output_size) # output_size is vocab size

    def forward(self, embeddings, state, lengths):
        '''
        INPUTS:
        embeddings: batch x L x embed_size, embeddings for input words
        state: num_layer x batch x hidden_size, initial state hidden layers
        OUTPUTS:
        output: batch x L x output_size/vocab_size 
        state: num_layer x batch x hidden_size
        '''
        #batch_size = embeddings.size(0)
        # packed input
        # embeddings: batch x L x embed_size
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        #print('state:', state.size()) # num_layers x batch x hidden_size
        hidden, state = self.rnn(packed, state) # hidden: batch x L x hidden_size, state: num_layers x batch x hidden_size
        #print('hidden:', hidden.size()) # batch x L x hidden_size? ttributeError: 'PackedSequence' object has no attribute 'size'
        #print('state:', state.size()) # num_layers x batch x hidden_size
        # state is already num_layer x batch x hidden_size
        # view below is not needed, I comment it out
        #state = state.view(self.hidden_layers, batch_size, self.hidden_size)
        #print('state:', state.size()) # num_layers x batch x hidden_size
        hidden, _ = pad_packed_sequence(hidden, batch_first=True) # output hidden: batch x L x hidden
        #print('hidden:', hidden.size()) # batch x L x hidden_size
        #print('state:', state.size())  # num_layers x batch x hidden_size
        output = self.rnn2out(hidden)
        #print('output:', output.size()) # batch x L x output_size/vocab_size
        return output, state


class NGMM(nn.Module):
    '''
    Neural Gaussian mixture model.
    '''
    def __init__(self, input_size, num_components, num_layers, hidden_size, output_size):
        super().__init__()
#        self.input_size = config.get('latent_size')
#        self.num_components = config.get('ngmm_num_components')
#        self.num_layers = config.get('ngmm_num_layers')
#        self.hidden_size = config.get('ngmm_hidden_size')
#        self.output_size = config.get('ngmm_output_size')
        self.input_size = input_size
        self.num_components = num_components
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size # number of dimensions in target
        
        self.mlp=nn.ModuleList()
        for i in range(self.num_layers):
            if i==0:
                in_features=self.input_size
            else:
                in_features=self.hidden_size
            out_features=self.hidden_size
            fc=nn.Linear(in_features, out_features)
            self.mlp.append( fc )
            
        self.linear_alpha = nn.Linear(self.hidden_size, self.num_components)
        self.linear_mu = nn.Linear(self.hidden_size, self.num_components*self.output_size)
        self.linear_logsigma = nn.Linear(self.hidden_size, self.num_components*self.output_size)


class MLP(nn.Module):
    '''
    MLP for regression.
    '''
    def __init__(self, input_size, num_layers, hidden_size, output_size, dropout):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size # number of dimensions in target
        self.dropout = dropout
        
        self.mlp=nn.ModuleList()
        for i in range(self.num_layers):
            if i==0:
                in_features=self.input_size
            else:
                in_features=self.hidden_size
            if i==self.num_layers-1:
                out_features=self.output_size
            else:
                out_features=self.hidden_size
                
            fc=nn.Linear(in_features, out_features)
            self.mlp.append( fc )
            
            
    def forward(self, z):
        for i in range(self.num_layers):
            if i<self.num_layers-1:
                z = F.dropout(F.relu( self.mlp[i](z) ), p=self.dropout )
                #z = F.relu( self.mlp[i](z) )        
            else:
                z = self.mlp[i](z)
        return z
        

class Frag2Mol(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.input_size = vocab.get_size()
        self.embed_size = config.get('embed_size')
        self.hidden_size = config.get('hidden_size')
        self.hidden_layers = config.get('hidden_layers')
        self.latent_size = config.get('latent_size')
        self.dropout = config.get('dropout')
        self.use_gpu = config.get('use_gpu')
        
        #NGMM hyperparameters
        self.predictor_num_layers = config.get('predictor_num_layers')
        self.predictor_hidden_size = config.get('predictor_hidden_size')
        self.predictor_output_size = config.get('predictor_output_size')

        embeddings = self.load_embeddings()
        self.embedder = nn.Embedding.from_pretrained(embeddings)
        
        #self.latent2rnn = nn.Linear(
        #    in_features=self.latent_size,
        #    out_features=self.hidden_size)
        # I changed the above to this:
        self.latent2rnn = nn.Linear(
            in_features=self.latent_size,
            out_features=self.hidden_layers*self.hidden_size)

        self.encoder = Encoder(
            input_size=self.input_size,
            embed_size=self.embed_size,
            hidden_size=self.hidden_size,
            hidden_layers=self.hidden_layers,
            latent_size=self.latent_size,
            dropout=self.dropout,
            use_gpu=self.use_gpu)

        self.decoder = Decoder(
            embed_size=self.embed_size,
            latent_size=self.latent_size,
            hidden_size=self.hidden_size,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            output_size=self.input_size)
        
        # MLP
        self.mlp = MLP(input_size=self.latent_size, num_layers=self.predictor_num_layers, 
                       hidden_size=self.predictor_hidden_size, output_size=self.predictor_output_size,
                       dropout=self.dropout)

    def forward(self, inputs, lengths):
        # INPUTS:
        # inputs: list or tensor of fragment indices
        # lengths: lengths of molecules
        # OUTPUTS:
        # output: batch x L x output_size
        # mu: batch x latent_size
        # sigma: batch x latent_size
        batch_size = inputs.size(0)
        embeddings = self.embedder(inputs)
        #print('self.training:', self.training) # where is it set?
        embeddings1 = F.dropout(embeddings, p=self.dropout, training=self.training) # where is self.training?
        z, mu, logvar = self.encoder(inputs, embeddings1, lengths)
        #state = self.latent2rnn(z) # z: num_layers x batch x latent_size, state: num_layers x batch x hidden_size
        #state = state.view(self.hidden_layers, batch_size, self.hidden_size) #num_layer x batch x hidden_size, maybe not needed
        
        state = self.latent2rnn(z) # batch x num_layers*latent_size
        #state = torch.tanh(state) # I added this, NOTE: should be consistent with sampler.
        state = state.view(batch_size, self.hidden_layers, self.hidden_size) # batch x num_layers x hidden_size
        state = state.transpose(1,0) # now state is num_layer x batch x hidden_state
        state= state.contiguous()
        #print('state:', state.shape)
        
        embeddings2 = F.dropout(embeddings, p=self.dropout, training=self.training)
        output, state = self.decoder(embeddings2, state, lengths)
        
        # the MLP component
        mlp_pred = self.mlp(z)
        #mlp_pred = 0
        
        return output, mu, logvar, mlp_pred


    def load_embeddings(self):
        filename = f'emb_{self.embed_size}.dat'
        path = self.config.path('config') / filename
        embeddings = np.loadtxt(path, delimiter=",")
        return torch.from_numpy(embeddings).float()


class Loss(nn.Module):
    def __init__(self, config, pad):
        super().__init__()
        self.config = config
        self.pad = pad

    def forward(self, output, target, mu, logvar, epoch, idx, properties, mlp_predicted):
        # INPUTS:
        # output: batch x L x output_size (vocab_size), softmax probs
        # target:  batch X L
        # mu: batch x latent_size
        # sigma: batch x latent_size
        
        batch_size=output.shape[0]
        output = F.log_softmax(output, dim=1)
        
        #print('in loss ...')
        #print('target size:', target.shape)
        #print('output size:', output.shape)

        # flatten all predictions and targets
        target = target.view(-1) # 1d tensor now: batch*L
        output = output.view(-1, output.size(2)) # batch*L x vocab_size
        
        #print('target size:', target.shape)
        #print('output size:', output.shape)

        # create a mask filtering out all tokens that ARE NOT the padding token
        mask = (target > self.pad).float() # 1d vector of integers

        # count how many tokens we have
        nb_tokens = int(torch.sum(mask).item()) # total number of tokens in the 1d target

        # pick the values for the label and zero out the rest with the mask
        output = output[range(output.size(0)), target] * mask # obtain the log-probabilities for each target class

        # compute cross entropy loss which ignores all <PAD> tokens
        #CE_loss = -torch.sum(output) / nb_tokens # this is mean over all all targets
        CE_loss = -torch.sum(output) / batch_size # mean over batch
        #CE_loss = -torch.sum(output)

        # compute KL Divergence
        #KL_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KL_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean() # mean over batch
        # alpha = (epoch + 1)/(self.config.get('num_epochs') + 1)
        # return alpha * CE_loss + (1-alpha) * KL_loss
 
        # regression loss, I THINK I NEED TO USE MSE INSTEAD

        MSE_loss = F.mse_loss(mlp_predicted, properties, reduction='sum')/batch_size
        #MSE_loss = F.mse_loss(mlp_predicted, properties, reduction='mean')
        #MSE_loss = 0
        #alpha=0.5
        #alpha = np.max( [(epoch-3)/self.config.get('num_epochs'), 0] )
        #alpha = (epoch/self.config.get('num_epochs'))**2
        #if epoch!=0 and (epoch+1)%2 == 0:
        #    alpha = (epoch+1)/self.config.get('num_epochs')
        #else:
        #    beta = 0
        offset_epoch = self.config.get('offset_epoch')
        T = self.config.get('start_epoch') + self.config.get('num_epochs') - offset_epoch
        t = epoch+1 - offset_epoch
        if T<=0:
            T=1
            t=1
        beta=get_beta(self.config.get('k_beta'), self.config.get('a_beta'), self.config.get('l_beta'), self.config.get('u_beta'), T, t)
        #loss = (1-beta) * ((1-alpha)*CE_loss + alpha*KL_loss) + beta*MSE_loss
        #loss = (1-beta) * (CE_loss + alpha*KL_loss) + beta*MSE_loss
        #loss = (1-beta) * (CE_loss + KL_loss) + beta*MSE_loss
        #loss = CE_loss + KL_loss
        #loss = (1-beta) * (CE_loss + alpha*KL_loss)
        #loss = CE_loss + alpha*KL_loss
        
        # increase alpha along with beta
        #if self.config.get('increase_alpha')>1:
            #alpha = self.config.get('increase_alpha')*alpha
        alpha = get_beta(self.config.get('k_alpha'), self.config.get('a_alpha'), self.config.get('l_alpha'), self.config.get('u_alpha'), T, t)
        
        loss = CE_loss + alpha*MSE_loss + beta*KL_loss
        if idx%100 == 0:
            print('CE_loss:',CE_loss.item())
            print('KL_loss:',KL_loss.item())
            print('MSE_loss:', MSE_loss.item())
        
        return loss, CE_loss.item(), MSE_loss.item(), KL_loss.item(), alpha, beta


def mvgaussian(t, mu,sigma):
    '''
    INPUTS:
    t: batch x K where K is the number of dimensions
    mu: batch x K
    sigma: batch x K
    OUTPUTS:
    pdf: batch
    '''
    #sigma_inv = 1/(sigma + 1e-31)
    K=len(mu)
    norm = torch.prod(sigma, dim=1) * torch.sqrt(2*torch.tensor(np.pi)).pow(K)
    expo = torch.exp( -0.5*( ((t-mu).pow(2)/sigma.pow(2)).sum(dim=1) ) )
    return expo/norm
    

def get_beta(k, a, l, u, T, t):
    """
    Compute the value of beta. When k=0, a is the fixed beta value. Usually we let a=1.
    Special cases:
        when a=0: beta=l
        when k=0: beta=max(a, l)
        when k=0, b=0: beta=a
        when k=0, a=0: beta=l
    INPUTS:
        a, T, t: scalars in formula beta=a*np.exp( k*(1-T/t) ) where a>=0, k>=0.
        l: scalar: l>=0, offset as min value of beta, usually we let l=0.
        u: scalar: u>0. max.
    OUTPUTS:
        beta: scalar.
    """
    beta = a*np.exp( k*(1-T/t) )
    beta = np.max( [beta, l] )
    beta = np.min( [beta, u] )
    return beta
    
