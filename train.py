import torch
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from network import NetG, NetD
from tqdm import tqdm
import matplotlib.pyplot as plt
from ImageProcessing import CelebaHQ
import numpy as np
import utils

netG = NetG()
netD = NetD(bn=False)
netG.cuda()
netD.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))

load = False
if load:
    checkpoint = torch.load("/net/girslchool/besnier/model/model_pggan.pytorch")
    netG.load_state_dict(checkpoint['generator']['state_dict'])
    netG.cuda()
    optimizerG.load_state_dict(checkpoint['generator']['optimizer'])
    netD.load_state_dict(checkpoint['discriminator']['state_dict'])
    netD.cuda()
    optimizerD.load_state_dict(checkpoint['discriminator']['optimizer'])

noise_fixe = torch.randn(10, 100, 1, 1).cuda()
alpha = False
cpt = 1

for k, batchsize in zip([8, 16, 32, 64, 128, 256, 512], [64, 64, 32, 32, 32, 32, 32]):
    dTrue = []
    dFalse = []
    ldf = 0
    ldt = 0
    dataset = CelebaHQ(h5file="/net/girlschool/besnier/CelebA-HQ/celebA-HQ",
                       datasize="data" + str(k) + "x" + str(k))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize,
                                             shuffle=True, num_workers=1)
    bar = tqdm(range(int(k/2)), ascii=True)
    for epoch in bar:
        if epoch > 10:
            alpha = False
        total = 10*len(dataloader)
        noise = torch.randn(batchsize, 100, 1, 1).cuda()
        for i, (x, y) in zip(tqdm(range(len(dataloader)), ascii=True), dataloader):
            x_cuda = x.cuda()
            real_label = torch.cuda.FloatTensor(batchsize).fill_(.9)
            fake_label = torch.cuda.FloatTensor(batchsize).fill_(.1)

            alpha_value = (epoch * len(dataloader) + (i+1)) / total

            # Discriminateur D
            optimizerD.zero_grad()
            outputTrue = netD(x_cuda, alpha=(1-alpha_value) if alpha else -1)
            # lossDT = F.binary_cross_entropy_with_logits(outputTrue, real_label)
            lossDT = -torch.mean(outputTrue)

            # with false label
            outputG = netG(Variable(noise))
            outputFalse = netD(outputG.detach(), alpha=(1-alpha_value) if alpha else -1)

            # lossDF = F.binary_cross_entropy_with_logits(outputFalse, fake_label)
            lossDF = torch.mean(outputFalse)
            dTrue.append(F.sigmoid(outputTrue).data.mean())
            dFalse.append(F.sigmoid(outputFalse).data.mean())

            gradient_penalty = utils.calc_gradient_penalty(netD, x_cuda, outputG, batch_size=batchsize, lda=10, view=x_cuda.size())
            (lossDT+lossDF+gradient_penalty).backward()
            optimizerD.step()

            ldf += lossDF
            ldt += lossDT

            # Generateur
            optimizerG.zero_grad()
            outputG = netG(noise, alpha=(1-alpha_value) if alpha else -1)
            outputD = netD(outputG, alpha=(1-alpha_value) if alpha else -1)
            # lossG = F.binary_cross_entropy_with_logits(outputD, real_label)
            lossG = -torch.mean(outputD)
            lossG.backward()
            optimizerG.step()

            if i == len(dataloader)-2:
                break
            bar.set_postfix({"Dataset": np.array(dTrue).mean(), "G": np.array(dFalse).mean(), "taille": k, "img": cpt-1})
        with open("/net/girlschool/besnier/image_HQ/res.csv", 'a') as f:
            a = ldf/(len(dataloader)-2)*1.
            b = ldt/(len(dataloader)-2)*1.
            f.write(str(a) + '\t' + str(b) + '\n')
        ldt = 0
        ldf = 0
        if epoch % 2 == 1:
            netG.eval()
            img = netG(noise, alpha=(1 - alpha_value) if alpha else -1).data.cpu()
            tensor_back = torch.zeros(k * 5 + 6, k * 5 + 6, 3)
            index = 0
            netG.train()

            for w in range(5):
                for h in range(5):
                    tensor_back[w * k + w + 1:(w + 1) * k + w + 1, h * k + h + 1:(h + 1) * k + h + 1] = img[index].transpose(0, 2).transpose(0, 1) / 2 + 0.5
                    index += 1
            plt.imsave("/net/girlschool/besnier/image_HQ/img{}".format(cpt), tensor_back)

            cpt += 1
        dTrue = []
        dFalse = []
        if k >= 128:
            torch.save({
                "generator":
                    {
                        'epoch': i + 1,
                        'state_dict': netG.state_dict(),
                        'optimizer': optimizerG.state_dict(),
                    },
                "discriminator":
                    {
                        'epoch': i + 1,
                        'state_dict': netD.state_dict(),
                        'optimizer': optimizerD.state_dict(),
                    }
            }, "/net/girlschool/besnier/model/model_pggan-" + str(k) + "x" + str(k) + ".pytorch")

    netG.add_layer()
    netD.add_layer()
    netG.cuda()
    netD.cuda()
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
    alpha = True
