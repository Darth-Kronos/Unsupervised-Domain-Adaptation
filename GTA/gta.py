import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import itertools
import numpy as np
import utils

# Generator network
class Generator(nn.Module):
    def __init__(self, opt, nclasses) -> None:
        super().__init__()

        self.ndim = opt.ndim
        self.ngf = opt.ngf
        self.nz = opt.nz
        self.gpu = opt.gpu
        self.nclasses = nclasses

        self.block = nn.Sequential(
            nn.ConvTranspose2d(self.nz+self.ndim+nclasses+1, self.ngf*8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*8, self.ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, input):   
        batch_size = input.size()[0]
        input = input.view(-1, self.ndim+self.nclasses+1, 1, 1)
        noise = torch.FloatTensor(batch_size, self.nz, 1, 1).normal_(0, 1)    
        if self.gpu>=0:
            noise = noise.cuda()
        noise = Variable(noise)
        output = self.main(torch.cat((input, noise),1))
        return output
    
# Discriminator 
class Discriminator(nn.Module):
    def __init__(self, opt, nclasses):
        super().__init__()
        
        self.ndf = opt.ndim//2
        self.feature = nn.Sequential(
            nn.Conv2d(3, 128, 5, 1, 1),            
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128, 128, 5, 1, 1),         
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,2),
            

            nn.Conv2d(128, 128, 5, 1, 1),           
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 128, 5, 1, 1),           
            nn.LeakyReLU(0.2, inplace=True),

            nn.Unflatten(1,())         
        )        

        # aux-classifier fc
        self.aux_classifier = nn.Linear(self.ndf*2, nclasses)
        # discriminator fc
        self.source_classifier = nn.Sequential(
        						nn.Linear(self.ndf*2, 1), 
        						nn.Sigmoid())              

    def forward(self, input):       
        output = self.feature(input)
        output_s = self.source_classifier(output.view(-1, self.ndf*2))
        output_s = output_s.view(-1)
        output_c = self.aux_classifier(output.view(-1, self.ndf*2))
        return output_s, output_c

# Pretrainied Resnet50
class FeatureExtractor(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.ndf = opt.ndim//2
        self.feature = nn.Sequential(
            nn.Conv2d(3, self.ndf, 5, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(self.ndf, self.ndf, 5, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
                    
            nn.Conv2d(self.ndf, self.ndf*2, 5, 1,0),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):   
        output = self.feature(input)
        return output.view(-1, 2*self.ndf)

class Classifier(nn.Module):
    def __init__(self, opt, nclasses):
        super().__init__()
        self.ndf = opt.ndim//2
        self.main = nn.Sequential(          
            nn.Linear(2*self.ndf, 2*self.ndf),
            nn.ReLU(inplace=True),
            nn.Linear(2*self.ndf, nclasses),                         
        )

    def forward(self, input):       
        output = self.main(input)
        return output

class gta:
    def __init__(self, opt, nclasses, mean, std, source_loader, target_loader):

        self.source_loader = source_loader
        self.target_loader = target_loader
        self.opt = opt
        self.mean = mean
        self.std = std
        self.best_val = 0
        
        # Defining networks
        self.nclasses = nclasses
        self.generator = Generator(opt, nclasses)
        self.discriminator = Discriminator(opt, nclasses)
        self.featureExtractor = FeatureExtractor(opt)
        self.classifier = Classifier(opt, nclasses)

        # Weight initialization
        self.generator.apply(utils.weights_init)
        self.discriminator.apply(utils.weights_init)
        self.featureExtractor.apply(utils.weights_init)
        self.classifier.apply(utils.weights_init)

        # Loss functions
        self.aux_loss = nn.CrossEntropyLoss()
        self.source_loss = nn.BCELoss()

        if opt.gpu>=0:
            self.discriminator.cuda()
            self.generator.cuda()
            self.featureExtractor.cuda()
            self.classifier.cuda()
            self.aux_loss.cuda()
            self.source_loss.cuda()

        # optimizers
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerF = optim.Adam(self.featureExtractor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerC = optim.Adam(self.classifier.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        self.real_label_val = 1
        self.fake_label_val = 0

    # validation function
    def validate(self, epoch):
        
        self.featureExtractor.eval()
        self.classifier.eval()
        total = 0
        correct = 0
    
        # Testing the model
        for i, source_data in enumerate(self.source_valloader):
            inputs, labels = source_data         
            inputv, labelv = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda()) 

            outC = self.classifier(self.featureExtractor(inputv))
            _, predicted = torch.max(outC.data, 1)        
            total += labels.size(0)
            correct += ((predicted == labels.cuda()).sum())
            
        val_acc = 100*float(correct)/total
        print(f'Epoch: {epoch}, Val Accuracy: {val_acc}')
    
        # Saving checkpoints
        torch.save(self.featureExtractor.state_dict(), '%s/models/featureExtractor.pth' %(self.opt.outf))
        torch.save(self.classifier.state_dict(), '%s/models/classifier.pth' %(self.opt.outf))
        
        if val_acc>self.best_val:
            self.best_val = val_acc
            torch.save(self.featureExtractor.state_dict(), '%s/models/model_best_featureExtractor.pth' %(self.opt.outf))
            torch.save(self.classifier.state_dict(), '%s/models/model_best_classifier.pth' %(self.opt.outf))


    """
    Train function
    """
    def train(self):
        
        curr_iter = 0
        
        real_label = torch.FloatTensor(self.opt.batch_size).fill_(self.real_label_val)
        fake_label = torch.FloatTensor(self.opt.batch_size).fill_(self.fake_label_val)
        if self.opt.gpu>=0:
            real_label, fake_label = real_label.cuda(), fake_label.cuda()
        real_label = Variable(real_label) 
        fake_label = Variable(fake_label) 
        
        for epoch in range(self.opt.epochs):
            
            # set the networks to training mode
            self.generator.train()    
            self.featureExtractor.train()    
            self.classifier.train()    
            self.discriminator.train()    
        
            for i, (source_data, target_data) in enumerate(zip(self.source_loader, self.target_loader)):
                          
                source_images, source_labels = source_data
                target_images, __ = target_data       
                source_inputs_unnorm = (((source_images*self.std[0]) + self.mean[0]) - 0.5)*2

                # One hot encoding for labels
                labels_onehot = np.zeros((self.opt.batch_size, self.nclasses+1), dtype=np.float32)
                for num in range(self.opt.batch_size):
                    labels_onehot[num, source_labels[num]] = 1
                source_labels_onehot = torch.from_numpy(labels_onehot)

                labels_onehot = np.zeros((self.opt.batch_size, self.nclasses+1), dtype=np.float32)
                for num in range(self.opt.batch_size):
                    labels_onehot[num, self.nclasses] = 1
                target_labels_onehot = torch.from_numpy(labels_onehot)
                
                if self.opt.gpu>=0:
                    source_images, source_labels = source_images.cuda(), source_labels.cuda()
                    source_inputs_unnorm = source_inputs_unnorm.cuda() 
                    target_images = target_images.cuda()
                    source_labels_onehot = source_labels_onehot.cuda()
                    target_labels_onehot = target_labels_onehot.cuda()
                
                # Wrapping in variable
                source_images, source_labels = Variable(source_images), Variable(source_labels)
                source_inputs_unnorm = Variable(source_inputs_unnorm)
                target_images = Variable(target_images)
                source_labels_onehot = Variable(source_labels_onehot)
                target_labels_onehot = Variable(target_labels_onehot)
                
                
                # Training D network
                
                self.discriminator.zero_grad()
                source_embedds = self.featureExtractor(source_images) # 2048 size embedds
                source_embedds_label = torch.cat((source_labels_onehot, source_embedds), 1)
                source_gen = self.generator(source_embedds_label)

                target_embedds = self.featureExtractor(target_images)
                target_embedds_label = torch.cat((target_labels_onehot, target_embedds),1)
                target_gen = self.generator(target_embedds_label)

                source_real_D_s, source_real_D_c = self.discriminator(source_inputs_unnorm)   
                errD_src_real_s = self.source_loss(source_real_D_s, real_label) 
                errD_src_real_c = self.aux_loss(source_real_D_c, source_labels) 

                source_fake_D_s, source_fake_D_c = self.discriminator(source_gen)
                errD_src_fake_s = self.source_loss(source_fake_D_s, fake_label)

                target_fake_D_s, target_fake_D_c = self.discriminator(target_gen)          
                errD_tgt_fake_s = self.source_loss(target_fake_D_s, fake_label)

                errD = errD_src_real_c + errD_src_real_s + errD_src_fake_s + errD_tgt_fake_s
                errD.backward(retain_graph=True)    
                self.optimizerD.step()
                

                # Training G network
                
                self.generator.zero_grad()       
                source_fake_D_s, source_fake_D_c = self.discriminator(source_gen)
                errG_c = self.aux_loss(source_fake_D_c, source_labels)
                errG_s = self.source_loss(source_fake_D_s, real_label)
                errG = errG_c + errG_s
                errG.backward(retain_graph=True)
                self.optimizerG.step()
                

                # Training C network
                
                self.classifier.zero_grad()
                outC = self.classifier(source_embedds)   
                errC = self.aux_loss(outC, source_labels)
                errC.backward(retain_graph=True)    
                self.optimizerC.step()

                
                # Training F network

                self.featureExtractor.zero_grad()
                errF_fromC = self.aux_loss(outC, source_labels)        

                source_fake_D_s, source_fake_D_c = self.discriminator(source_gen)
                errF_src_fromD = self.aux_loss(source_fake_D_c, source_labels)*(self.opt.adv_weight)

                target_fake_D_s, target_fake_D_c = self.discriminator(target_gen)
                errF_tgt_fromD = self.source_loss(target_fake_D_s, real_label)*(self.opt.adv_weight*self.opt.alpha)
                
                errF = errF_fromC + errF_src_fromD + errF_tgt_fromD
                errF.backward()
                self.optimizerF.step()        
                
                curr_iter += 1
                
                    
                # Learning rate scheduling
                if self.opt.lrd:
                    self.optimizerD = utils.exp_lr_scheduler(self.optimizerD, epoch, self.opt.lr, self.opt.lrd, curr_iter)    
                    self.optimizerF = utils.exp_lr_scheduler(self.optimizerF, epoch, self.opt.lr, self.opt.lrd, curr_iter)
                    self.optimizerC = utils.exp_lr_scheduler(self.optimizerC, epoch, self.opt.lr, self.opt.lrd, curr_iter)                  
            
            # Validate every epoch
            self.validate(epoch+1)

    
    