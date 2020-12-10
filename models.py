#@title AE & VAE class { form-width: "300px" }
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import torchvision
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from IPython.display import clear_output


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

        if self.transform is not None:
            sample = self.transform(sample)

        return self.data[idx], 0
        #return sample
        
        
#@title AE learning { form-width: "300px" }
def create_AE_embedding_(d, inter_dim, code_dim, epochs = 50, plot=False):

    history = defaultdict(list)

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
    iterator = range(epochs) if plot else tqdm_notebook(range(epochs)) 
    for epoch in iterator:
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
            
            history['loss'].append(train_loss.item())

            clear_output(wait=True)
            fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10,5))
            axes[0].plot(history['loss'])
            axes[0].set_title(f'Loss iter')
            axes[1].plot(history['loss_epoch'])
            axes[1].set_title(f'Loss epoch')

        # compute the epoch training loss
        history['loss_epoch'].append(np.mean(history['loss'][-len(train_loader):]))

        
    emb = model.get_code_embedding(train_dataset)

    return emb, history




#@title VAE learning { form-width: "300px" }
def create_VAE_embedding_(d, inter_dim, code_dim, epochs = 50):

    history = defaultdict(list)

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
            
            history['total_loss'].append(train_loss.item())
            history['mse_loss'].append(mse_loss.item())
            history['kld_loss'].append(kld_loss.item())

            clear_output(wait=True)
            fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(10,5))
            axes[0,0].plot(history['total_loss'])
            axes[0,0].set_title(f'Total loss iter')

            axes[0,1].plot(history['mse_loss'])
            axes[0,1].set_title(f'MSE loss iter')

            axes[0,2].plot(history['kld_loss'])
            axes[0,2].set_title(f'KLD loss iter')

            axes[1,0].plot(history['total_loss_epoch'])
            axes[1,0].set_title(f'Total loss epoch')

            axes[1,1].plot(history['mse_loss_epoch'])
            axes[1,1].set_title(f'MSE epoch')
            
            axes[1,2].plot(history['kld_loss_epoch'])
            axes[1,2].set_title(f'KLD epoch')

        history['total_loss_epoch'].append(np.mean(history['total_loss'][-len(train_loader):]))
        history['mse_loss_epoch'].append(np.mean(history['mse_loss'][-len(train_loader):]))
        history['kld_loss_epoch'].append(np.mean(history['kld_loss'][-len(train_loader):]))
        
    emb = model.get_code_embedding(train_dataset)
    
    return emb, history