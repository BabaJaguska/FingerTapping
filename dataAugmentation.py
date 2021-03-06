# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:40:36 2021

@author: minja
"""

#!/usr/bin/env python
# coding: utf-8

# # Data Augmentation


# -   With smaller datasets, GANs can provide useful data augmentation that substantially 
# [improve classifier performance](https://arxiv.org/abs/1711.04340). 
# -   You have one type of data already labeled and would like to make predictions on 
#[another related dataset for which you have no labels](https://www.nature.com/articles/s41598-019-52737-x). 
#(You'll learn about the techniques for this use case in future notebooks!)
# -   You want to protect the privacy of the people who provided their information so you can provide
# access to a [generator instead of real data](https://www.ahajournals.org/doi/full/10.1161/CIRCOUTCOMES.118.005122). 
# -   You have [input data with many missing values](https://arxiv.org/abs/1806.02920), where 
#the input dimensions are correlated and you would like to train a model on complete inputs. 
# -   You would like to be able to identify a real-world abnormal feature in an image 
#[for the purpose of diagnosis](https://link.springer.com/chapter/10.1007/978-3-030-00946-5_11), 
#but have limited access to real examples of the condition. 


#%%

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset
import Parameters
from torch.autograd import Variable
# torch.manual_seed(0) # Set for our testing purposes, please do not change!

def show_tensor_images(image_tensor, num_images=25, size=(3, 32, 32), nrow=5, show=True):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    # image_unflat = image_unflat[:,:,:,None]
    # image_unflat = image_unflat.permute(0,3,1,2)
    # image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    # plt.imshow(image_grid.permute(1, 2, 0))
    # plt.imshow(image_grid)
    
    plt.subplot(5,1,1)
    plt.plot(image_unflat[1].T)
        
    plt.subplot(5,1,2)
    plt.plot(image_unflat[10].T)
        
    plt.subplot(5,1,3)
    plt.plot(image_unflat[-1].T)
    
    plt.subplot(5,1,4)
    plt.plot(image_unflat[-10].T)
    
    plt.subplot(5,1,5)
    plt.plot(image_unflat[25].T)
    
    if show:
        plt.show()



class DatasetTapping(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, Y):
      self.X = X
      self.Y = Y
      if len(self.X) != len(self.Y):
        raise Exception("The length of X does not match the length of Y")
    
  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
    _x = self.X[index]
    _y = self.Y[index]

    return _x, _y



#%%
# #### Generator


class Generator(nn.Module):
    '''
    Generator Class
    Values:
        input_dim: the dimension of the input vector, a scalar
        im_chan: the number of channels of the output image, a scalar
              (6 is your default for tapping data) 
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, gen_type, input_dim=64, im_chan=6, hidden_dim=16):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        # Build the neural network
        if gen_type == 'conv':
            self.gen = nn.Sequential(
                self.make_gen_block(input_dim, hidden_dim * 16, kernel_size = 3, stride = 1),
                self.make_gen_block(hidden_dim * 16, hidden_dim * 8, kernel_size=7, stride=2),
                self.make_gen_block(hidden_dim * 8, hidden_dim * 8, kernel_size=5, stride=1),
                self.make_gen_block(hidden_dim * 8, hidden_dim * 4, kernel_size=5, stride = 1, dropout = 1),
                self.make_gen_block(hidden_dim * 4, hidden_dim * 4, kernel_size=5, stride = 1),
                self.make_gen_block(hidden_dim * 4, hidden_dim * 4, kernel_size=7, stride=1),
                self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=7, stride = 1),
                self.make_gen_block(hidden_dim * 2, hidden_dim * 2, kernel_size=7, stride = 1, dropout = 1),
                self.make_gen_block(hidden_dim * 2, hidden_dim * 2, kernel_size=7, stride = 2),
                # self.make_gen_block(hidden_dim * 2, hidden_dim * 2, kernel_size=7, stride = 2),
                self.make_gen_block(hidden_dim * 2, hidden_dim * 2, kernel_size=7, stride = 2),
                self.make_gen_block(hidden_dim * 2, hidden_dim * 2, kernel_size=7, stride = 2, dropout = 1),
                self.make_gen_block(hidden_dim * 2, hidden_dim * 1, kernel_size=10, stride = 2),
                self.make_gen_block(hidden_dim * 1, hidden_dim * 1, kernel_size=5, stride = 1),
                self.make_gen_block(hidden_dim, im_chan, kernel_size=3, final_layer=True),
            )
            
        if gen_type == 'LSTM':
            self.gen = nn.Sequential(
                nn.LSTM(input_dim, hidden_dim, num_layers=3, bidirectional = True),
                nn.Linear(hidden_dim)
            )
            
    def LSTM_step(self, x,h,c):
        
        """
        Args:
            x: (batch_size,  1), sequence of tokens generated by generator
            h: (1, batch_size, hidden_dim), lstm hidden state
            c: (1, batch_size, hidden_dim), lstm cell state
        """

        output, (h, c) = self.lstm(x, (h, c))
        # pred = F.softmax(self.lin(output.view(-1, self.hidden_dim)), dim=1)
        
        return output, h, c
    
    def init_hidden(self, batch_size, hidden_dim, device='cuda'):
        
        h = Variable(torch.zeros((1, batch_size, hidden_dim)))
        c = Variable(torch.zeros((1, batch_size, hidden_dim)))
        h, c = h.to(device), c.to(device)
        
        return h, c
    
    
    def sample(self, batch_size, seq_len, x=None):
        samples = [];
        h, c = self.init_hidden(batch_size)
        for i in range(seq_len):
            output, h,c = self.step(x, h,c)
            x = output.multinomial(1)
        
    

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, dropout = 0, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            
            if dropout:
                return nn.Sequential(
                    nn.ConvTranspose1d(input_channels, output_channels, kernel_size, stride),
                    nn.BatchNorm1d(output_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.1)

                )
            return nn.Sequential(
                nn.ConvTranspose1d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm1d(output_channels),
                nn.ReLU(inplace=True)

            )
        else:
            return nn.Sequential(
                nn.ConvTranspose1d(input_channels, output_channels, kernel_size, stride),
                # nn.Tanh(),
            )

    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, input_dim)
        '''
        x = noise.view(len(noise), self.input_dim, 1) # sklonila jos jednu "1" sa kraja 
        return self.gen(x)


def get_noise(n_samples, input_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, input_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        input_dim: the dimension of the input vector, a scalar
        device: the device type
    '''
    return torch.randn(n_samples, input_dim, device=device)

def combine_vectors(x, y):
    '''
    Function for combining two vectors with shapes (n_samples, ?) and (n_samples, ?)
    Parameters:
    x: (n_samples, ?) the first vector. 
        In this assignment, this will be the noise vector of shape (n_samples, z_dim), 
        but you shouldn't need to know the second dimension's size.
    y: (n_samples, ?) the second vector.
        Once again, in this assignment this will be the one-hot class vector 
        with the shape (n_samples, n_classes), but you shouldn't assume this in your code.
    '''
    return torch.cat([x, y], 1)

def get_one_hot_labels(labels, n_classes):
    '''
    labels: (n_samples, 1) 
    n_classes: a single integer corresponding to the total number of classes in the dataset
    tapping vec ima one hot labels?
    
    # radi samo sa LONG tipom. Ako ne radi, probaj: (Labels = labels.type(torch.long))
    '''
    
    return F.one_hot(labels, n_classes)

#%%
# ## Training


tapping_shape = (6, 1, Parameters.samples)
n_classes = 4
 
#   *   criterion: the loss function
#   *   n_epochs: the number of times you iterate through the entire dataset when training
#   *   z_dim: the dimension of the noise vector
#   *   display_step: how often to display/visualize the images
#   *   batch_size: the number of images per forward/backward pass
#   *   lr: the learning rate
#   *   device: the device type

# n_epochs = 300
# z_dim = 128
# display_step = 50
# batch_size = 64
# lr = 0.0002
# device = 'cuda'


# Then, you want to set your generator's input dimension. Recall that for 
#conditional GANs, the generator's input is the noise vector concatenated with the class vector.



# %%
# #### Classifier


class Classifier(nn.Module):
    '''
    Classifier Class
    Values:
        im_chan: the number of channels of the output image, a scalar
        n_classes: the total number of classes in the dataset, an integer scalar
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_chan, n_classes, hidden_dim=32):
        super(Classifier, self).__init__()
        self.disc = nn.Sequential(
            self.make_classifier_block(im_chan, hidden_dim),
            self.make_classifier_block(hidden_dim, hidden_dim * 2),
            self.make_classifier_block(hidden_dim * 2, hidden_dim * 4),
            self.make_classifier_block(hidden_dim * 4, n_classes, final_layer=True),
        )

    def make_classifier_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a classifier block; 
        a convolution, a batchnorm (except in the final layer), and an activation (except in the final
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm1d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the classifier: Given an image tensor, 
        returns an n_classes-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with im_chan channels
        '''
        class_pred = self.disc(image)
        return class_pred.view(len(class_pred), -1)

#%%
 
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
      im_chan: the number of channels of the output image, a scalar
            6 za tapping default
      hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_chan=6, hidden_dim=32):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim, stride=1, kernel_size = 3, depthwise = True),
            self.make_disc_block(hidden_dim, hidden_dim * 2, kernel_size = 5),
            self.make_disc_block(hidden_dim * 2, hidden_dim * 4, kernel_size = 3),
            self.make_disc_block(hidden_dim * 4, hidden_dim * 4),
            self.make_disc_block(hidden_dim * 4, hidden_dim * 4, kernel_size = 3),
            self.make_disc_block(hidden_dim * 4, hidden_dim * 4),
            self.make_disc_block(hidden_dim * 4, hidden_dim * 8),
            # self.make_disc_block(hidden_dim * 8, hidden_dim * 8),
            self.make_disc_block(hidden_dim * 8, 1, final_layer=True)
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, depthwise = False, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a discriminator block of the DCGAN; 
        a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            if depthwise:
                return nn.Sequential(
                    nn.Conv1d(input_channels, input_channels, kernel_size, stride, groups = input_channels),
                    nn.Conv1d(input_channels, output_channels, kernel_size = 1, stride = 1),
                    nn.BatchNorm1d(output_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    # nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Dropout(0.2)
                )
            else:
                return nn.Sequential(
                    nn.Conv1d(input_channels, output_channels, kernel_size, stride),
                    nn.BatchNorm1d(output_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.2)
                )
        else:
            return nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        '''
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)
    
    
def get_gradient(crit, real, fake, epsilon):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)
    
    # Calculate the critic's scores on the mixed images
    mixed_scores = crit(mixed_images)
    
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        # Note: You need to take the gradient of outputs with respect to inputs.
        # This documentation may be useful, but it should not be necessary:
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
        #### START CODE HERE ####
        inputs=mixed_images,
        outputs=mixed_scores,
        #### END CODE HERE ####
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient

def gradient_penalty(gradient):
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)
    
    # Penalize the mean squared distance of the gradient norms from 1
    #### START CODE HERE ####
    penalty = torch.mean((gradient_norm - 1)**2)
    #### END CODE HERE ####
    return penalty

def get_gen_loss(crit_fake_pred):
    '''
    Return the loss of a generator given the critic's scores of the generator's fake images.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''
    #### START CODE HERE ####
    gen_loss = -1. * torch.mean(crit_fake_pred)
    #### END CODE HERE ####
    return gen_loss

def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    '''
    Return the loss of a critic given the critic's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
        crit_real_pred: the critic's scores of the real images
        gp: the unweighted gradient penalty
        c_lambda: the current weight of the gradient penalty 
    Returns:
        crit_loss: a scalar for the critic's loss, accounting for the relevant factors
    '''
    #### START CODE HERE ####
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp
    #### END CODE HERE ####
    return crit_loss
    

def train_generator(dataX, dataY):
    # and discriminator
    device = Parameters.device
    display_step = Parameters.display_step
    tapping_shape = dataX.shape[1:]
    generator_input_dim = Parameters.z_dim + Parameters.n_classes
    gen = Generator(generator_input_dim, im_chan = tapping_shape[0]).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=Parameters.lr)
    discriminator_input_dim = tapping_shape[0] + n_classes
    disc = Discriminator(discriminator_input_dim).to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=Parameters.lr)

    def weights_init(m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm1d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
            
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

    # criterion = nn.BCEWithLogitsLoss()

    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    
    c_lambda = 5
    crit_repeats = 3
    
    dataloader = DataLoader(DatasetTapping(dataX, dataY),
                            batch_size = Parameters.batch_size,
                            shuffle = True)
    all_gen_losses = []
    all_disc_losses = []
    
    print('Training...')
    for epoch in range(Parameters.n_epochs):
        # Dataloader returns the batches and the labels  
        for real, one_hot_labels in dataloader:
            cur_batch_size = len(real)
            # Flatten the batch of real images from the dataset
            real = real.to(device).float()
            one_hot_labels = one_hot_labels.to(device).float()
            # Convert the labels from the dataloader into one-hot versions of those labels
            #one_hot_labels = get_one_hot_labels(labels.to(device), n_classes).float()

            image_one_hot_labels = one_hot_labels[:, :, None]
            image_one_hot_labels = image_one_hot_labels.repeat(1, 1, tapping_shape[1])
            # prosirila mu dimenzije za dimenzije signala. Lepis ovo na sliku?

            ### Update discriminator ###
            mean_iteration_critic_loss = 0
            for _ in range(crit_repeats):
                # Zero out the discriminator gradients
                disc_opt.zero_grad()
                # Get noise corresponding to the current batch_size 
                fake_noise = get_noise(cur_batch_size, Parameters.z_dim, device=Parameters.device)
            
                # Combine the vectors of the noise and the one-hot labels for the generator
                noise_and_labels = combine_vectors(fake_noise, one_hot_labels) # pazi one hot labels, not image one hot labels
                fake = gen(noise_and_labels)
                # Combine the vectors of the images and the one-hot labels for the discriminator
                fake_image_and_labels = combine_vectors(fake.detach(), image_one_hot_labels)
                real_image_and_labels = combine_vectors(real, image_one_hot_labels)
                disc_fake_pred = disc(fake_image_and_labels)
                disc_real_pred = disc(real_image_and_labels)
                
                
                epsilon = torch.rand(len(real), 1, 1, device=device, requires_grad=True)
                gradient = get_gradient(disc,real_image_and_labels, fake_image_and_labels, epsilon)
                gp = gradient_penalty(gradient)
                disc_loss = get_crit_loss(disc_fake_pred, disc_real_pred, gp, c_lambda)
    
                # disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred) + (0.3 * torch.rand(1).item()))
                # disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred) * (0.3 * torch.rand(1).item() + 0.7))
                # disc_loss = (disc_fake_loss + disc_real_loss) / 2
                disc_loss.backward(retain_graph=True)
                disc_opt.step() 
                mean_iteration_critic_loss  += disc_loss.item() / crit_repeats;

                # Keep track of the average discriminator loss
            mean_discriminator_loss += mean_iteration_critic_loss / display_step
            
            
            ##################

            ### Update generator ###
            # Zero out the generator gradients
            gen_opt.zero_grad()

            # Pass the discriminator the combination of the fake images and the one-hot labels
            fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)

            disc_fake_pred = disc(fake_image_and_labels)
            gen_loss = get_gen_loss(disc_fake_pred)
            gen_loss.backward()
            gen_opt.step()
            
            
        ################## jos jedan korak za generator mozda???? #####
        
            # gen_opt.zero_grad()
        
            # fake_noise = get_noise(cur_batch_size, Parameters.z_dim, device=Parameters.device)
        
            # # Combine the vectors of the noise and the one-hot labels for the generator
            # noise_and_labels = combine_vectors(fake_noise, one_hot_labels) # pazi one hot labels, not image one hot labels
            # fake = gen(noise_and_labels)
            # fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
            # disc_fake_pred = disc(fake_image_and_labels)
            # gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            # gen_loss.backward()
            # gen_opt.step()
            
            
        ####
        
        

            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step
            all_gen_losses.append(gen_loss.item())
            all_disc_losses.append(disc_loss.item())

            if cur_step % display_step == 0 and cur_step > 0:
                print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                show_tensor_images(real, size = real.shape)
                show_tensor_images(fake, size = fake.shape)
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1
    
    
    plt.plot(all_disc_losses)
    plt.plot(all_gen_losses)
    plt.legend(('disc loss', 'gen loss'))
    plt.show()
            
    return gen, disc
            
            
# %% Get that generator

# train_generator(data)



#%%

def train_classifier(dataTrain, dataVal):
    criterion = nn.CrossEntropyLoss()
    n_epochs = 10

    validation_dataloader = DataLoader(
         dataVal,
         batch_size=Parameters.batch_size)

    display_step = 10
    batch_size = 512
    lr = 0.0002
    device = 'cuda'
    classifier = Classifier(tapping_shape[0], n_classes).to(device)
    classifier_opt = torch.optim.Adam(classifier.parameters(), lr=lr)
    cur_step = 0
    for epoch in range(n_epochs):
        for real, labels in tqdm(dataloader):
            cur_batch_size = len(real)
            real = real.to(device).float()
            labels = labels.to(device)

            ### Update classifier ###
            # Get noise corresponding to the current batch_size
            classifier_opt.zero_grad()
            labels_hat = classifier(real.detach())
            classifier_loss = criterion(labels_hat, labels)
            classifier_loss.backward()
            classifier_opt.step()

            if cur_step % display_step == 0:
                classifier_val_loss = 0
                classifier_correct = 0
                num_validation = 0
                for val_example, val_label in validation_dataloader:
                    cur_batch_size = len(val_example)
                    num_validation += cur_batch_size
                    val_example = val_example.to(device)
                    val_label = val_label.to(device)
                    labels_hat = classifier(val_example)
                    classifier_val_loss += criterion(labels_hat, val_label) * cur_batch_size
                    classifier_correct += (labels_hat.argmax(1) == val_label).float().sum()

                print(f"Step {cur_step}: "
                        f"Classifier loss: {classifier_val_loss.item() / num_validation}, "
                        f"classifier accuracy: {classifier_correct.item() / num_validation}")
            cur_step += 1


# ## Tuning the Classifier
# After two courses, you've probably had some fun debugging your GANs 
# and have started to consider yourself a bug master.
# For this assignment, your mastery will be put to the test on some interesting bugs... 
# well, bugs as in insects.
# 
# As a bug master, you want a classifier capable of classifying different species of bugs:
    # bees, beetles, butterflies, caterpillar, and more. 
    #Luckily, you found a great dataset with a lot of animal species and objects,
    #and you trained your classifier on that.
# 
# But the bug classes don't do as well as you would like. 
#Now your plan is to train a GAN on the same data so it can generate new bugs to make your 
#classifier better at distinguishing between all of your favorite bugs!
# 
# You will fine-tune your model by augmenting the original real data with 
#fake data and during that process, observe how to increase the accuracy of your
#  classifier with these fake, GAN-generated bugs. After this, you will prove your worth as a bug master.

# #### Sampling Ratio
# 
# Suppose that you've decided that although you have this pre-trained general
#  generator and this general classifier, capable of identifying 100 classes with some accuracy (~17%),
# what you'd really like is a model that can classify the five different kinds of bugs in the dataset.
# You'll fine-tune your model by augmenting your data with the generated images. 
# Keep in mind that both the generator and the classifier were trained on the same images: 
    # the 40 images per class you painstakingly found so your generator may not be great. 
    # This is the caveat with data augmentation, ultimately you are still bound by the real
    # data that you have but you want to try and create more. To make your models even better, 
    # you would need to take some more bug photos, label them, and add them to your training set 
    # and/or use higher quality photos.
# 
# To start, you'll first need to write some code to sample a combination of real and generated images.
#  Given a probability, `p_real`, you'll need to generate a combined tensor where roughly `p_real` 
# of the returned images are sampled from the real images. Note that you should not interpolate 
# the images here: you should choose each image from the real or fake set with a given probability. 
# For example, if your real images are a tensor of `[[1, 2, 3, 4, 5]]` 
# and your fake images are a tensor of `[[-1, -2, -3, -4, -5]]`, and `p_real = 0.2`,
# two potential return values are `[[1, -2, 3, -4, -5]]` or `[[-1, 2, -3, -4, -5]]`
# 
# In addition, we will expect the images to remain in the same order to maintain
#  their alignment with their labels (this applies to the fake images too!). 
# 
# <details>
# <summary>
# <font size="3" color="green">
# <b>Optional hints for <code><font size="4">combine_sample</font></code></b>
# </font>
# </summary>
# 
# 1.   This code probably shouldn't be much longer than 3 lines
# 2.   You can index using a set of booleans which have the same length as your tensor
# 3.   You want to generate an unbiased sample, which you can do (for example) 
# with `torch.rand(length_reals) > p`.
# 4.   There are many approaches here that will give a correct answer here. 
# You may find [`torch.rand`](https://pytorch.org/docs/stable/generated/torch.rand.html) 
# or [`torch.bernoulli`](https://pytorch.org/docs/master/generated/torch.bernoulli.html) useful. 
# 5.   You don't want to edit an argument in place, so you may find [`cur_tensor.clone()`](
# https://pytorch.org/docs/stable/tensors.html) useful too, which makes a copy of `cur_tensor`. 
# 
# </details>

def combine_sample(real, fake, p_real):
    '''
    Function to take a set of real and fake images of the same length (x)
    and produce a combined tensor with length (x) and sampled at the target probability
    Parameters:
        real: a tensor of real images, length (x)
        fake: a tensor of fake images, length (x)
        p_real: the probability the images are sampled from the real set
    '''
    #### START CODE HERE ####
    mask = torch.rand(len(real)) > p_real
    cloneImages = real.clone()
    cloneImages[mask] = fake[mask]
    target_images = cloneImages
    #### END CODE HERE ####
    return target_images

#%%
# unit testic

n_test_samples = 9999
test_combination = combine_sample(
    torch.ones(n_test_samples, 1), 
    torch.zeros(n_test_samples, 1), 
    0.3
)
# Check that the shape is right
assert tuple(test_combination.shape) == (n_test_samples, 1)
# Check that the ratio is right
assert torch.abs(test_combination.mean() - 0.3) < 0.05
# Make sure that no mixing happened
assert test_combination.median() < 1e-5

test_combination = combine_sample(
    torch.ones(n_test_samples, 10, 10), 
    torch.zeros(n_test_samples, 10, 10), 
    0.8
)
# Check that the shape is right
assert tuple(test_combination.shape) == (n_test_samples, 10, 10)
# Make sure that no mixing happened
assert torch.abs((test_combination.sum([1, 2]).median()) - 100) < 1e-5

test_reals = torch.arange(n_test_samples)[:, None].float()
test_fakes = torch.zeros(n_test_samples, 1)
test_saved = (test_reals.clone(), test_fakes.clone())
test_combination = combine_sample(test_reals, test_fakes, 0.3)
# Make sure that the sample isn't biased
assert torch.abs((test_combination.mean() - 1500)) < 100
# Make sure no inputs were changed
assert torch.abs(test_saved[0] - test_reals).sum() < 1e-3
assert torch.abs(test_saved[1] - test_fakes).sum() < 1e-3

test_fakes = torch.arange(n_test_samples)[:, None].float()
test_combination = combine_sample(test_reals, test_fakes, 0.3)
# Make sure that the order is maintained
assert torch.abs(test_combination - test_reals).sum() < 1e-4
if torch.cuda.is_available():
    # Check that the solution matches the input device
    assert str(combine_sample(
        torch.ones(n_test_samples, 10, 10).cuda(), 
        torch.zeros(n_test_samples, 10, 10).cuda(),
        0.8
    ).device).startswith("cuda")
print("Success!")

# %% 
# Now you have a challenge: find a `p_real` and a generator image 
# such that your classifier gets an average of a 51% accuracy or higher on the insects,
#  when evaluated with the `eval_augmentation` function.
#  **You'll need to fill in `find_optimal` to find these parameters to solve this part!**
#  Note that if your answer takes a very long time to run, you may need to hard-code the solution it finds. 
# 
# When you're training a generator, you will often have to look at different 
# checkpoints and choose one that does the best (either empirically or using some evaluation method).
# Here, you are given four generator checkpoints: `gen_1.pt`, `gen_2.pt`, `gen_3.pt`, `gen_4.pt`.
#  You'll also have some scratch area to write whatever code you'd like to solve this problem, 
# but you must return a `p_real` and an image name of your selected generator checkpoint. 
# You can hard-code/brute-force these numbers if you would like, but you are encouraged to try 
# to solve this problem in a more general way. 
# In practice, you would also want a test set (since it is possible to overfit on a validation set),
# but for simplicity you can just focus on the validation set.

def find_optimal():
    # In the following section, you can write the code to choose your optimal answer
    # You can even use the eval_augmentation function in your code if you'd like!
    gen_names = [
        "gen_1.pt",
        "gen_2.pt",
        "gen_3.pt",
        "gen_4.pt"
    ]

    #### START CODE HERE #### 
    best_p_real, best_gen_name = None, None
    
    # hard coded:  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<!
    best_p_real = 0.6
    best_gen_name= gen_names[3]
    
    #### END CODE HERE ####
    return best_p_real, best_gen_name

def generate_batch_of_fakes(gen, one_hot_labels, batch_size):  
    
    fake_noise = get_noise(batch_size, Parameters.z_dim, device=Parameters.device)
    one_hot_labels = one_hot_labels.to(Parameters.device).float()
    # Combine the vectors of the noise and the one-hot labels for the generator
    noise_and_labels = combine_vectors(fake_noise, one_hot_labels)
    
    batch_of_fakes = gen(noise_and_labels)
    
    return batch_of_fakes
    



def augmented_train(p_real, gen_name):
    gen = Generator(generator_input_dim).to(device)
    gen.load_state_dict(torch.load(gen_name))

    classifier = Classifier(tapping_shape[0], n_classes).to(device)
    classifier.load_state_dict(torch.load("class.pt"))
    criterion = nn.CrossEntropyLoss()
    batch_size = 256

    train_set = torch.load("insect_train.pt")
    val_set = torch.load("insect_val.pt")
    dataloader = DataLoader(
        torch.utils.data.TensorDataset(train_set["images"], train_set["labels"]),
        batch_size=batch_size,
        shuffle=True
    )
    validation_dataloader = DataLoader(
        torch.utils.data.TensorDataset(val_set["images"], val_set["labels"]),
        batch_size=batch_size
    )

    display_step = 1
    lr = 0.0002
    n_epochs = 20
    classifier_opt = torch.optim.Adam(classifier.parameters(), lr=lr)
    cur_step = 0
    best_score = 0
    for epoch in range(n_epochs):
        for real, labels in dataloader:
            real = real.to(device)
            # Flatten the image
            labels = labels.to(device)
            one_hot_labels = get_one_hot_labels(labels.to(device), n_classes).float()

            ### Update classifier ###
            # Get noise corresponding to the current batch_size
            classifier_opt.zero_grad()
            cur_batch_size = len(labels)
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            noise_and_labels = combine_vectors(fake_noise, one_hot_labels)
            fake = gen(noise_and_labels)

            target_images = combine_sample(real.clone(), fake.clone(), p_real)
            labels_hat = classifier(target_images.detach())
            classifier_loss = criterion(labels_hat, labels)
            classifier_loss.backward()
            classifier_opt.step()

            # Calculate the accuracy on the validation set
            if cur_step % display_step == 0 and cur_step > 0:
                classifier_val_loss = 0
                classifier_correct = 0
                num_validation = 0
                with torch.no_grad():
                    for val_example, val_label in validation_dataloader:
                        cur_batch_size = len(val_example)
                        num_validation += cur_batch_size
                        val_example = val_example.to(device)
                        val_label = val_label.to(device)
                        labels_hat = classifier(val_example)
                        classifier_val_loss += criterion(labels_hat, val_label) * cur_batch_size
                        classifier_correct += (labels_hat.argmax(1) == val_label).float().sum()
                    accuracy = classifier_correct.item() / num_validation
                    if accuracy > best_score:
                        best_score = accuracy
            cur_step += 1
    return best_score



# def eval_augmentation(p_real, gen_name, n_test=20):
#     total = 0
#     for i in range(n_test):
#         total += augmented_train(p_real, gen_name)
#     return total / n_test

# best_p_real, best_gen_name = find_optimal()
# performance = eval_augmentation(best_p_real, best_gen_name)
# print(f"Your model had an accuracy of {performance:0.1%}")
# assert performance > 0.51
# print("Success!")

#%%
# You'll likely find that the worst performance is when the generator is performing alone:
# this corresponds to the case where you might be trying to hide 
# the underlying examples from the classifier.
# Perhaps you don't want other people to know about your specific bugs!


# accuracies = []
# p_real_all = torch.linspace(0, 1, 21)
# for p_real_vis in tqdm(p_real_all):
#     accuracies += [eval_augmentation(p_real_vis, best_gen_name, n_test=4)]
# plt.plot(p_real_all.tolist(), accuracies)
# plt.ylabel("Accuracy")
# _ = plt.xlabel("Percent Real Images")

#%%
# Here's a visualization of what the generator is actually generating,
# with real examples of each class above the corresponding generated image.  

# examples = [4, 41, 80, 122, 160]
# train_images = torch.load("insect_train.pt")["images"][examples]
# train_labels = torch.load("insect_train.pt")["labels"][examples]

# one_hot_labels = get_one_hot_labels(train_labels.to(device), n_classes).float()
# fake_noise = get_noise(len(train_images), z_dim, device=device)
# noise_and_labels = combine_vectors(fake_noise, one_hot_labels)
# gen = Generator(generator_input_dim).to(device)
# gen.load_state_dict(torch.load(best_gen_name))

# fake = gen(noise_and_labels)
# show_tensor_images(torch.cat([train_images.cpu(), fake.cpu()]))







