import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

import tflib as lib
import tflib.save_images
import tflib.mnist
import tflib.plot

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

import os
from tqdm import tqdm


torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0

DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 10000 #40000 # How many generator iterations to train for; 10000/200=50 epochs
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)
img_shape = (1, 28, 28)
rank_margin = 1
fb_percentage = 1 
num_add = int(50000*fb_percentage)
use_tensorboard = True
num_check = 50000

output_path = 'results'
run = 0 
output_path = os.path.join(output_path, 'dicgan_small_digits', str(run))
sample_path = os.path.join(output_path, 'samples')
log_path = os.path.join(output_path, 'logs')
if not os.path.exists(sample_path):
    os.makedirs(sample_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)

if use_tensorboard:
    writer = SummaryWriter(log_path)

lib.print_model_settings(locals().copy())

# ==================Definition Start======================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        preprocess = nn.Sequential(
            nn.Linear(128, 4*4*4*DIM),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*DIM, 2*DIM, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*DIM, DIM, 5),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 1, 8, stride=2)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*DIM, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        return output.view(-1, OUTPUT_DIM)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Conv2d(1, DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*DIM)
        out = self.output(out)
        return out.view(-1)

def generate_image(frame, netG):
    noise = torch.randn(BATCH_SIZE, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise, volatile=True)
    samples = netG(noisev)
    samples = samples.view(BATCH_SIZE, 28, 28)
    samples = samples.cpu().data.numpy()

    lib.save_images.save_images(
        samples,
        os.path.join(sample_path, 'samples_{}.png'.format(frame))
    )

# Dataset iterator
train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
data, labels = lib.mnist.load_train_data()
mask_labels = np.zeros(len(data))
rand_idx = np.random.choice(len(data), BATCH_SIZE)
print(labels[rand_idx])
samples = data[rand_idx].reshape(BATCH_SIZE, 28, 28)
lib.save_images.save_images(
        samples,
        os.path.join(sample_path, 'real_samples.png')
    )

def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# ==================Definition End======================

netG = Generator()
netD = Discriminator()
print(netG)
print(netD)

if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

# load pretrained model
load_model_path = './results/pretrain'
model_iter = 199999
D_file = os.path.join(load_model_path, 'D_{}.pt'.format(model_iter))
G_file = os.path.join(load_model_path, 'G_{}.pt'.format(model_iter))
netG.load_state_dict(torch.load(G_file))
netD.load_state_dict(torch.load(D_file))
# load cls
from cls_mnist.cls_mnist import Net
mnist_cls = Net().cuda()
checkpoint = torch.load('cls_mnist/mnist_cnn.pt')
mnist_cls.load_state_dict(checkpoint)
mnist_cls.eval()

counter = 0
margin_rank_loss = torch.nn.MarginRankingLoss(margin=rank_margin, reduction='none')
counter_epoch = 0
fb_flag = True
data_len = len(data)
count_eff_pair = 0
for iteration in tqdm(range(ITERS)):
    start_time = time.time()
    if iteration % 200 == 0:
        with open(output_path + "/eff_pair.txt",'a+') as f:
            f.write("Iteration: {} \t Num: {} \n ".format(iteration, count_eff_pair))
        count_eff_pair = 0
        torch.save(netG.state_dict(), os.path.join(output_path, 'G_{}.pt'.format(iteration-1)))
        torch.save(netD.state_dict(), os.path.join(output_path, 'D_{}.pt'.format(iteration-1))) 
        preds, d_preds = [], []
        add_data = []
        netG.eval()
        netD.eval()
        for k in range(num_add//BATCH_SIZE):
            noise = torch.randn(BATCH_SIZE, 128)
            if use_cuda:
                noise = noise.cuda(gpu)
            noisev = autograd.Variable(noise)  # totally freeze netG
            fake = netG(noisev).detach()
            gen_imgs = fake
            gen_imgs = gen_imgs.view(-1, 1, 28, 28)
            d_pred = netD(gen_imgs)
            d_preds.append(d_pred.data.cpu().numpy())
            pred = mnist_cls(gen_imgs)
            pred = pred.argmax(dim=1)
            preds.append(pred.data.cpu().numpy())
            imgs = gen_imgs.view(gen_imgs.size(0), *img_shape)
            add_data.append(imgs.data.cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        add_data = np.concatenate(add_data, axis=0).reshape(num_add, -1)
        if np.sum(preds<1)/float(num_check) == 1:
            fb_flag = False
        d_preds = np.concatenate(d_preds)
        len_goods = np.sum(preds < 1)
        with open(os.path.join(output_path,'possitive_1.txt'), 'a+') as f:
            f.write("Epoch: {} \t Pos: {}\n".format((iteration)//200, float(len_goods)/float(num_check)))
        len_goods = np.sum(preds < 5)
        with open(os.path.join(output_path,'possitive_5.txt'), 'a+') as f:
            f.write("Epoch: {} \t Pos: {}\n".format((iteration)//200, float(len_goods)/float(num_check)))
        with open(os.path.join(output_path,'d_preds.txt'), 'a+') as f:
            f.write("Epoch: {} \t Pos: {} \t Neg: {}\n".format((iteration)//200, np.mean(d_preds[preds<5]), np.mean(d_preds[preds>=5])))
        # remove old data and add new data
        if fb_flag and iteration > 0:
            toRemove = np.argsort(mask_labels)[:num_add]
            data = np.delete(data, toRemove, axis=0)
            labels = np.delete(labels, toRemove)
            mask_labels = np.delete(mask_labels, toRemove)
            data = np.concatenate([data, add_data], axis=0)
            labels = np.concatenate([labels, preds])
            mask_labels = np.concatenate([labels, np.repeat(counter_epoch, len(add_data))])
            perm = np.random.permutation(len(data))
            data = data[perm]
            labels = labels[perm]
            mask_labels = mask_labels[perm]
        rand_idx = np.random.choice(len(data), BATCH_SIZE)
        samples = data[rand_idx].reshape(BATCH_SIZE, 28, 28)
        lib.save_images.save_images(
                samples,
                os.path.join(sample_path, 'train_samples.png')
            )
    ############################
    # (1) Update D network
    ###########################
    netG.train()
    netD.train()
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update
    for iter_d in range(CRITIC_ITERS):
        _data = data[(counter%(data_len//BATCH_SIZE))*BATCH_SIZE:(counter%(data_len//BATCH_SIZE)+1)*BATCH_SIZE] #next(data) #data.next()
        real_data = torch.Tensor(_data)
        if use_cuda:
            real_data = real_data.cuda(gpu)
        real_data_v = autograd.Variable(real_data)

        netD.zero_grad()

        # train with real
        D_real = netD(real_data_v)
        D_real = D_real.mean()
        # print D_real
        D_real.backward(mone)

        # train with fake
        noise = torch.randn(BATCH_SIZE, 128)
        if use_cuda:
            noise = noise.cuda(gpu)
        with torch.no_grad():
            noisev = autograd.Variable(noise)  # totally freeze netG
            fake = autograd.Variable(netG(noisev).data)
        inputv = fake
        D_fake = netD(inputv)
        D_fake = D_fake.mean()
        D_fake.backward(one)

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
        gradient_penalty.backward()
        # rank loss
        ls = labels[(counter%(data_len//BATCH_SIZE))*BATCH_SIZE:(counter%(data_len//BATCH_SIZE)+1)*BATCH_SIZE]
        first_labels = ls[:int(BATCH_SIZE/2)]
        second_labels = ls[int(BATCH_SIZE/2):]
        comps = np.greater(first_labels, second_labels)
        r_labels = torch.ones(int(BATCH_SIZE/2)).cuda()
        r_labels[comps] = -1
        comps = np.equal(first_labels, second_labels)
        count_eff_pair += np.sum(~comps)
        #print(count_eff_pair, comps, first_labels, second_labels)
        r_weights = torch.ones(int(BATCH_SIZE/2)).cuda()
        r_weights[comps] = 0
        first_samples = real_data_v[:int(BATCH_SIZE/2)]
        second_samples = real_data_v[int(BATCH_SIZE/2):]
        r_out_first = netD(first_samples)
        r_out_second = netD(second_samples)
        r_loss_ds = margin_rank_loss(r_out_first, r_out_second, r_labels)
        r_loss_d = torch.mean(r_loss_ds*r_weights)
        r_loss_d.backward()
        r_loss_good = torch.mean(r_out_first*((r_labels+1)/2)+r_out_second*((r_labels-1)/(-2)))
        r_loss_bad = torch.mean(r_out_second*((r_labels+1)/2)+r_out_first*((r_labels-1)/(-2))) 
        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        optimizerD.step()
        counter += 1

    ############################
    # (2) Update G network
    ###########################
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation
    netG.zero_grad()

    noise = torch.randn(BATCH_SIZE, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise)
    fake = netG(noisev)
    G = netD(fake)
    G = G.mean()
    G.backward(mone)
    G_cost = -G
    optimizerG.step()

    # Write logs and save samples
    lib.plot.plot(os.path.join(output_path, 'time'), time.time() - start_time)
    lib.plot.plot(os.path.join(output_path, 'train_disc_cost'), D_cost.cpu().data.numpy())
    lib.plot.plot(os.path.join(output_path, 'train_gen_cost'), G_cost.cpu().data.numpy())
    lib.plot.plot(os.path.join(output_path, 'wasserstein_distance'), Wasserstein_D.cpu().data.numpy())
    if use_tensorboard:
        writer.add_scalars('d_out', {'real': D_real, 'fake': D_fake, 'good': r_loss_good, 'bad': r_loss_bad}, iteration)
        writer.add_scalars('loss', {'d_loss': (D_fake-D_real), 'r_loss':r_loss_d}, iteration)

    # Calculate dev loss and generate samples every 100 iters
    if iteration % 200 == 199:
        dev_disc_costs = []
        for images,_ in dev_gen():
            imgs = torch.Tensor(images)
            if use_cuda:
                imgs = imgs.cuda(gpu)
            imgs_v = autograd.Variable(imgs, volatile=True)

            D = netD(imgs_v)
            _dev_disc_cost = -D.mean().cpu().data.numpy()
            dev_disc_costs.append(_dev_disc_cost)
        lib.plot.plot(os.path.join(output_path, 'dev_disc_cost'), np.mean(dev_disc_costs))

        generate_image(iteration, netG)
             
    # Write logs every 100 iters
    if (iteration < 5) or (iteration % 100 == 99):
        lib.plot.flush()

    lib.plot.tick()
