#@title AE & VAE class { form-width: "300px" }
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import torchvision
from torch.utils.data import Dataset, DataLoader

class Encoder(nn.Module):
    def __init__(self, orig_dim, inter_dim, code_dim, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=orig_dim, out_features=inter_dim
        )
        self.encoder_output_layer = nn.Linear(
            in_features=inter_dim, out_features=code_dim
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.sigmoid(code)
        #code = torch.relu(code)
        
        return code

class Decoder(nn.Module):


    def __init__(self, code_dim, inter_dim, orig_dim, **kwargs):
        super().__init__()
        self.decoder_hidden_layer = nn.Linear(
            in_features=code_dim, out_features=inter_dim
        )
        self.decoder_output_layer = nn.Linear(
            in_features=inter_dim, out_features=orig_dim
        )

    def forward(self, code):
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.sigmoid(activation)
        
        return reconstructed
        
        
class AE(nn.Module):


    def __init__(self, orig_dim, inter_dim, code_dim):
        super().__init__()
        self.encoder = Encoder(orig_dim = orig_dim, inter_dim = inter_dim, code_dim = code_dim)
        self.decoder = Decoder(orig_dim = orig_dim, inter_dim = inter_dim, code_dim = code_dim)
    
    def forward(self, features):
        code = self.encoder.forward(features)
        reconstructed = self.decoder.forward(code)
        return reconstructed  

    def get_code_embedding(self, dataset):
        Encoder = self.encoder
        input = torch.tensor(dataset.data).float()
        embedding = Encoder.forward(input)
        return embedding.detach().numpy().T


class VAE(nn.Module):

    
    def __init__(self, orig_dim, inter_dim, code_dim):
        super(VAE, self).__init__()

        self.encoder = Encoder(orig_dim = orig_dim, inter_dim = inter_dim, code_dim = 2*code_dim)
        self.decoder = Decoder(orig_dim = orig_dim, inter_dim = inter_dim, code_dim = code_dim)
        self.orig_dim = orig_dim
        self.inter_dim = inter_dim
        self.code_dim = code_dim

    def reparameterization(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
    
    def get_code(self, features):
        x = self.encoder.forward(features)
        
        #print('x shape:', x.shape)
        x = x.view(-1, 2, self.code_dim)

        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance

        #print('mu shape:', mu.shape)
        # get the latent vector through reparameterization
        code = self.reparameterization(mu, log_var)
        #print('code shape:', mu.shape)
        
        '''
        code = x
        mu = x
        log_var = x
        '''
        return code, mu, log_var
        
    def forward(self, features):
        # encoding
        code, mu, log_var = self.get_code(features)

 
        # decoding
        reconstructed = self.decoder.forward(code)
        return reconstructed, mu, log_var

    def get_code_embedding(self, dataset):
        Encoder = self.encoder
        input = torch.tensor(dataset.data).float()
        embedding, mu, log_var = self.get_code(input)
        
        return embedding.detach().numpy().T

class NeuroDataset(Dataset):

    """Neural activity dataset."""

    def __init__(self, data, transform=None):

        self.data = data.T
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'vector': self.data[idx].reshape(-1, 1), 'target': 0}

        if self.transform:
            sample = self.transform(sample)

        return self.data[idx], 0
        #return sample
        
        
        
        
#@title AE learning { form-width: "300px" }
def create_AE_embedding_(d, inter_dim, code_dim, epochs = 50):

    #---------------------------------------------------------------------------
    batch_size = 32
    learning_rate = 1e-2
    
    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_dataset = NeuroDataset(d)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    #---------------------------------------------------------------------------
    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = AE(orig_dim = len(d), inter_dim = inter_dim, code_dim = code_dim).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # mean-squared error loss
    criterion = nn.MSELoss()

    #---------------------------------------------------------------------------

    for epoch in range(epochs):
        loss = 0
        for batch_features, _ in train_loader:
            
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions

            outputs = model(batch_features.float())
            
            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features.float())
            
            # compute accumulated gradients
            train_loss.backward()
            
            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
        
        # compute the epoch training loss
        loss = loss / len(train_loader)
        
        # display the epoch training loss
        if (epoch + 1) % 1 == 0:
            print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))

    
    emb = model.get_code_embedding(train_dataset)

    return emb




#@title VAE learning { form-width: "300px" }
def create_VAE_embedding_(d, inter_dim, code_dim, epochs = 50):

    #---------------------------------------------------------------------------
    batch_size = 32
    learning_rate = 1e-2
    
    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_dataset = NeuroDataset(d)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    #---------------------------------------------------------------------------
    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE(orig_dim = len(d), inter_dim = inter_dim, code_dim = code_dim).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate lr
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # BCE error loss
    # criterion = nn.BCELoss(reduction='sum')
    criterion = nn.MSELoss()

    #---------------------------------------------------------------------------

    for epoch in range(epochs):
        loss = 0
        loss1 = 0
        loss2 = 0
        for batch_features, _ in train_loader:              
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            data = batch_features.float()
            reconstruction, mu, logvar = model(data)
            
            # compute training reconstruction loss
            mse_loss = criterion(reconstruction, data) 
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #* train_dataset.__len__()/batch_size
            train_loss = mse_loss + kld_loss
            
            # compute accumulated gradients
            train_loss.backward()
            
            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
            loss1 += mse_loss.item()
            loss2 += kld_loss.item()
        
        # compute the epoch training loss
        loss = loss / len(train_loader)
        loss1 = loss1 / len(train_loader)
        loss2 = loss2 / len(train_loader)
        
        # display the epoch training loss
        if (epoch + 1) % 1 == 0:
            print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))
            print("epoch : {}/{}, mse loss = {:.8f}".format(epoch + 1, epochs, loss1))
            print("epoch : {}/{}, kld loss = {:.8f}".format(epoch + 1, epochs, loss2))

    emb = model.get_code_embedding(train_dataset)
    
    return emb