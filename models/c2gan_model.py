import torch
import numpy as np
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
# import extract_facial_landmark
# from train import *

class C2GANModel(BaseModel):
    def name(self):
        return 'C2GANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc-3,opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.input_nc-3,opt.fineSize, opt.fineSize)
        self.input_C = self.Tensor(opt.batchSize, opt.input_nc-3,opt.fineSize, opt.fineSize)
        self.input_D = self.Tensor(opt.batchSize, opt.input_nc-3,opt.fineSize, opt.fineSize)

        # load/define networks
        # self.netG_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        # self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_L = networks.define_G(3, 3, opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        # self.netG2 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_L = networks.define_D(6, opt.ndf, opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            # self.load_network(self.netG_A, 'G_A', opt.which_epoch)
            # self.load_network(self.netG_B, 'G_B', opt.which_epoch)
            self.load_network(self.netG, 'G', opt.which_epoch)
            self.load_network(self.netG_L, 'G_L', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
                self.load_network(self.netD_L, 'D_L', opt.which_epoch)

        if self.isTrain:

            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.fake_L_pool = ImagePool(opt.pool_size)
            # self.fake_A_pool = ImagePool(opt.pool_size)
            # self.fake_B_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            # self.criterionSmoothL1 = torch.nn.SmoothL1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterion_cycle = torch.nn.L1Loss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_L = torch.optim.Adam(self.netG_L.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
            #                                     lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(), self.netG_L.parameters()),
            #                                     lr=opt.lr, betas=(opt.beta1, 0.999))
            
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_L = torch.optim.Adam(self.netD_L.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_G_L)
            self.optimizers.append(self.optimizer_D_L)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        networks.print_network(self.netG_L)
        # networks.print_network(self.netG_A)
        # networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD)
            networks.print_network(self.netD_L)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_C = input['C']
        input_D = input['D']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
            input_C = input_C.cuda(self.gpu_ids[0], async=True)
            input_D = input_D.cuda(self.gpu_ids[0], async=True)

        self.input_A=input_A
        self.input_B=input_B
        self.input_C=input_C
        self.input_D=input_D
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def forward(self):

        self.real_A = Variable(self.input_A)
        # combine_AC=torch.cat((self.input_A, self.input_C), 1)
        # print('combina_AC',combine_AC)
        # self.fake_B = self.netG.forward(self.real_A)
        # self.fake_B = self.netG.forward(Variable(combine_AC))
        
        self.real_B = Variable(self.input_B)
        self.input_C = Variable(self.input_C)
        self.input_D = Variable(self.input_D)
        # self.input_D = Variable(self.input_D)
        # combine_realB_D=torch.cat((self.real_B, self.input_D), 1)
        # self.recovered_A = self.netG.forward(combine_realB_D)

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)    
        self.input_C = Variable(self.input_C, volatile=True) 
        combine_realA_C=torch.cat((self.real_A, self.input_C), 1)
        self.fake_B = self.netG.forward(combine_realA_C)
        # self.fake_B = self.netG_A.forward(combine_realA_C)
        self.fake_B_L = self.netG_L.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)
        # combine_BD=torch.cat((self.input_B, self.input_D), 1)
        # self.fake_A = self.netG.forward(Variable(combine_BD))
        self.input_D = Variable(self.input_D, volatile=True)
        combine_fakeB_D=torch.cat((self.fake_B, self.input_D), 1)
        # self.recovered_A = self.netG_B(combine_fakeB_D)
        self.recovered_A = self.netG(combine_fakeB_D)

        self.fake_A_L =self.netG_L(self.real_A)
        # combine_realB_D=torch.cat((self.real_B, self.input_D), 1)
        # self.fake_A = self.netG_B(combine_realB_D)
        # self.fake_A = self.netG(combine_realB_D)

        # combine_fakeA_C=torch.cat((self.fake_A, self.input_C), 1)
        # self.recovered_B = self.netG_A(combine_fakeA_C)
        # self.recovered_B = self.netG(combine_fakeA_C)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_G(self):

        combine_realA_C=torch.cat((self.real_A, self.input_C), 1)
        # self.fake_B = self.netG_A.forward(combine_realA_C)
        self.fake_B = self.netG.forward(combine_realA_C)
        
        # First, G(A) should fake the discriminator
        combine_AC_fakeB = torch.cat((self.real_A, self.input_C, self.fake_B), 1)
        pred_combine_AC_fakeB = self.netD.forward(combine_AC_fakeB)
        self.loss_G_GAN_A2B = self.criterionGAN(pred_combine_AC_fakeB, True)


        combine_fakeB_D=torch.cat((self.fake_B, self.input_D), 1)
        # self.fake_A = self.netG_B.forward(combine_realB_D)
        self.recovered_A = self.netG.forward(combine_fakeB_D)

        combine_BD_recoveredA = torch.cat((self.real_B, self.input_D, self.recovered_A), 1)
        pred_combine_BD_recoveredA = self.netD.forward(combine_BD_recoveredA)
        self.loss_G_GAN_B2A = self.criterionGAN(pred_combine_BD_recoveredA, True)

        self.loss_G_GAN = self.loss_G_GAN_A2B + self.loss_G_GAN_B2A   


        # Second, constraction loss, G(A) = B
        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_recon
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_recon

        # self.loss_G_L1 = (self.loss_G_L1_B + self.loss_G_L1_A) * 0.5
          
        # Cycle loss
        # self.input_D = Variable(self.input_D)
        # combine_fakeB_D=torch.cat((self.fake_B, self.input_D), 1)
        # self.recovered_A = self.netG_B.forward(combine_fakeB_D)
        # self.recovered_A = self.netG.forward(combine_fakeB_D)

        # combine_fakeA_C=torch.cat((self.fake_A, self.input_C), 1)
        # self.recovered_B = self.netG_A.forward(combine_fakeA_C)
        # self.recovered_B = self.netG.forward(combine_fakeA_C)

        self.loss_G_cyc = self.criterion_cycle(self.recovered_A, self.real_A) *self.opt.lambda_cycle
        # self.loss_cycle_B = self.criterion_cycle(self.recovered_B, self.real_B)*self.opt.lambda_cycle

        # self.loss_G_cyc = (self.loss_cycle_A + self.loss_cycle_B) * 0.5
        # landmark loss

        # total      
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_cyc

        # self.loss_G_L2 = self.criterionL2(self.fake_B, self.real_B) * self.opt.lambda_A
        # self.loss_G = self.loss_G_GAN + self.loss_G_L2

        # self.loss_G_SmoothL1 = self.criterionSmoothL1(self.fake_B, self.real_B) * self.opt.lambda_A
        # self.loss_G = self.loss_G_GAN + self.loss_G_SmoothL1

        self.loss_G.backward(retain_graph=True)

    def backward_G_L(self):

        self.fake_B_L = self.netG_L.forward(self.fake_B)
        
        # First, G(A) should fake the discriminator
        combine_fake_B_L = torch.cat((self.fake_B, self.fake_B_L), 1)
        pred_combine_fake_B_L = self.netD_L.forward(combine_fake_B_L)
        self.loss_G_L_GAN_1 = self.criterionGAN(pred_combine_fake_B_L, True)


        # combine_real_B_L = torch.cat((self.fake_B, self.input_C), 1)
        # pred_combine_real_B_L = self.netD_L.forward(combine_real_B_L)
        # self.loss_G_L_GAN_2 = self.criterionGAN(pred_combine_real_B_L, True)

        self.fake_A_L = self.netG_L.forward(self.recovered_A)
        
        # First, G(A) should fake the discriminator
        combine_fake_A_L = torch.cat((self.recovered_A, self.fake_A_L), 1)
        pred_combine_fake_A_L = self.netD_L.forward(combine_fake_A_L)
        self.loss_G_L_GAN_3 = self.criterionGAN(pred_combine_fake_A_L, True)


        # combine_real_A_L = torch.cat((self.recovered_A, self.input_D), 1)
        # pred_combine_real_A_L = self.netD_L.forward(combine_real_A_L)
        # self.loss_G_L_GAN_4 = self.criterionGAN(pred_combine_real_A_L, True)


        self.loss_G_L_GAN = self.loss_G_L_GAN_1 + self.loss_G_L_GAN_3


        # Second, constraction loss, G(A) = B
        self.loss_G_L_cyc = (self.criterion_cycle(self.fake_B_L, self.input_C) + self.criterion_cycle(self.fake_A_L, self.input_D)) * self.opt.lambda_cycle
        # self.loss_G_Lamk = extract_facial_landmark.extract_landmark(util.tensor2im(self.real_B.data), util.tensor2im(self.fake_B.data)) + extract_facial_landmark.extract_landmark(util.tensor2im(self.real_A.data), util.tensor2im(self.recovered_A.data)) + extract_facial_landmark.extract_landmark(util.tensor2im(self.real_A.data), util.tensor2im(self.fake_A.data)) + extract_facial_landmark.extract_landmark(util.tensor2im(self.real_B.data), util.tensor2im(self.recovered_B.data))
        # self.loss_G_Lamk = extract_facial_landmark.extract_landmark(util.tensor2im(self.real_B.data), util.tensor2im(self.fake_B.data)) + extract_facial_landmark.extract_landmark(util.tensor2im(self.real_A.data), util.tensor2im(self.recovered_A.data)) 

        # if epoch<50:
        #     print('Using pretrained model')
        #     self.loss_G_L = self.loss_G_L_GAN + self.loss_G_L_cyc
        # else:
        #     print('Learning landmark')
        self.loss_G_L = self.loss_G_L_GAN + self.loss_G_L_cyc

        # self.loss_G_L2 = self.criterionL2(self.fake_B, self.real_B) * self.opt.lambda_A
        # self.loss_G = self.loss_G_GAN + self.loss_G_L2

        # self.loss_G_SmoothL1 = self.criterionSmoothL1(self.fake_B, self.real_B) * self.opt.lambda_A
        # self.loss_G = self.loss_G_GAN + self.loss_G_SmoothL1

        self.loss_G_L.backward(retain_graph=True)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # self.input_C = Variable(self.input_C)
        combine_AC_fakeB = self.fake_AB_pool.query(torch.cat((self.real_A, self.input_C, self.fake_B), 1))
        pred_combine_AC_fakeB = self.netD.forward(combine_AC_fakeB.detach())
        self.loss_D_combine_AC_fakeB = self.criterionGAN(pred_combine_AC_fakeB, False)

        # Real
        self.real_B = Variable(self.input_B)
        combine_AC_realB = torch.cat((self.real_A, self.input_C, self.real_B), 1)
        pred_combine_AC_realB = self.netD.forward(combine_AC_realB)
        self.loss_D_combine_AC_realB = self.criterionGAN(pred_combine_AC_realB, True)

        # stop backprop to the generator by detaching fake_B
        # self.input_D = Variable(self.input_D)
        combine_BD_recovered_A = self.fake_AB_pool.query(torch.cat((self.real_B, self.input_D, self.recovered_A), 1))
        # print(combine_BD_recovered_A.size())
        pred_combine_BD_recovered_A= self.netD.forward(combine_BD_recovered_A.detach())
        self.loss_D_combine_BD_fakeA = self.criterionGAN(pred_combine_BD_recovered_A, False)

        # Real
        combine_BD_realA = torch.cat((self.real_B, self.input_D, self.real_A), 1)
        pred_combine_BD_realA = self.netD.forward(combine_BD_realA)
        self.loss_D_combine_BD_realA = self.criterionGAN(pred_combine_BD_realA, True)

        self.loss_D_real = (self.loss_D_combine_AC_realB + self.loss_D_combine_BD_realA ) * 0.5
        self.loss_D_fake = (self.loss_D_combine_BD_fakeA + self.loss_D_combine_AC_fakeB ) * 0.5

        # Combined loss
        self.loss_D = self.loss_D_fake + self.loss_D_real

        self.loss_D.backward(retain_graph=True)

    def backward_D_L(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # self.input_C = Variable(self.input_C)
        combine_fake_B_L = self.fake_L_pool.query(torch.cat((self.fake_B, self.fake_B_L), 1))
        pred_combine_fake_B_L = self.netD_L.forward(combine_fake_B_L.detach())
        self.loss_pred_combine_fake_B_L = self.criterionGAN(pred_combine_fake_B_L, False)

        # Real
        combine_real_B_L = torch.cat((self.fake_B, self.input_C), 1)
        pred_combine_real_B_L= self.netD_L.forward(combine_real_B_L)
        self.loss_pred_combine_real_B_L = self.criterionGAN(pred_combine_real_B_L, True)

        # stop backprop to the generator by detaching fake_B
        # self.input_D = Variable(self.input_D)
        combine_fake_A_L = self.fake_L_pool.query(torch.cat((self.recovered_A, self.fake_A_L), 1))
        pred_combine_fake_A_L = self.netD_L.forward(combine_fake_A_L.detach())
        self.loss_pred_combine_fake_A_L = self.criterionGAN(pred_combine_fake_A_L, False)

        # Real
        combine_real_A_L = torch.cat((self.recovered_A, self.input_D), 1)
        pred_combine_real_A_L= self.netD_L.forward(combine_real_A_L)
        self.loss_pred_combine_real_A_L = self.criterionGAN(pred_combine_real_A_L, True)

        self.loss_D_L_real = (self.loss_pred_combine_real_B_L + self.loss_pred_combine_real_A_L ) * 0.5
        self.loss_D_L_fake = (self.loss_pred_combine_fake_B_L + self.loss_pred_combine_fake_A_L ) * 0.5

        # Combined loss
        self.loss_D_L = self.loss_D_L_fake + self.loss_D_L_real

        self.loss_D_L.backward()

    def optimize_parameters(self):
        self.forward()
        
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G_L.zero_grad()
        self.backward_G_L()
        self.optimizer_G_L.step()

        self.optimizer_D_L.zero_grad()
        self.backward_D_L()
        self.optimizer_D_L.step()

    # def get_current_errors(self):
    #     return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]), ('G_L1', self.loss_G_L1.data[0]),
    #         ('G_Cyc', self.loss_G_cyc.data[0]), ('G_Lamd', self.loss_G_Lamk), ('G_Total', self.loss_G.data[0]), ('D_real', self.loss_D_real.data[0]), ('D_fake', self.loss_D_fake.data[0]), ('D_Total', self.loss_D.data[0])])

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]), ('G_L1', self.loss_G_L1.data[0]),
            ('G_Cyc', self.loss_G_cyc.data[0]), ('G_Total', self.loss_G.data[0]), ('D_real', self.loss_D_real.data[0]), 
            ('D_fake', self.loss_D_fake.data[0]), ('D_Total', self.loss_D.data[0]),  ('loss_G_L_GAN', self.loss_G_L_GAN.data[0]), 
            ('loss_G_L_cyc', self.loss_G_L_cyc.data[0]), ('loss_G_L', self.loss_G_L.data[0]), ('loss_D_L_real', self.loss_D_L_real.data[0]),
            ('loss_D_L_fake', self.loss_D_L_fake.data[0]), ('loss_D_L_Total', self.loss_D_L.data[0])])


    def get_current_visuals(self):
        # self.input_C = Variable(self.input_C)
        # self.input_D = Variable(self.input_D)
        real_A = util.tensor2im(self.real_A.data)
        input_C = util.tensor2im(self.input_C.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        # fake_A = util.tensor2im(self.fake_A.data)
        input_D = util.tensor2im(self.input_D.data)
        recovered_A = util.tensor2im(self.recovered_A.data)
        # recovered_B = util.tensor2im(self.recovered_B.data)
        fake_B_L = util.tensor2im(self.fake_B_L.data)
        fake_A_L = util.tensor2im(self.fake_A_L.data)

        return OrderedDict([('real_A', real_A), ('input_C', input_C), ('fake_B', fake_B), ('real_B', real_B), ('input_D', input_D), ('recovered_A', recovered_A), ('fake_A_L', fake_A_L), ('fake_B_L', fake_B_L)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netG_L, 'G_L', label, self.gpu_ids)
        # self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        # self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
        self.save_network(self.netD_L, 'D_L', label, self.gpu_ids)
