import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        ''' Initialize the layers of this model.'''
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        # Embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # The LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        
        # Final Fully-connected Layer
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        # Initialize the weights
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        self.word_embeddings.weight.data.uniform_(-0.1, 0.1)
        # FC weights as random uniform
        self.linear.weight.data.uniform_(-0.1, 0.1)
        # Set bias tensor to all zeros
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions):
        """Decode image feature vectors and generate captions."""
        #captions = captions[:,:-1]
        embeddings = self.word_embeddings(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        #packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(embeddings)
        hiddens = hiddens[:,:-1,:]
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # List of predicted IDs
        preds = []        
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            # Getting maximum probabilities using Greedy Search
            _, predicted = outputs.max(1)
            # Append prediction with max. prob. to predicted IDs list
            preds.append(predicted.item())
            inputs = self.word_embeddings(predicted).unsqueeze(1)
            
        return preds