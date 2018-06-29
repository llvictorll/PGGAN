import torch
from torch import optim
import torch.nn.functional as F
from network import NetG, NetD
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from dataloader import dataload
netG = NetG()
netD = NetD()

netG.cuda()
netD.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=4e-5, betas=(0.5,0.999))
optimizerG = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5,0.999))

noise_fixe = noise = torch.randn(64, 100, 1, 1).cuda()
alpha = False
cpt = 1

for k in [8, 16, 32, 64]:
    batch_size = 64
    dataset = dataload(k)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    bar = tqdm(range(1))

    for epoch in bar:
        if epoch > 10:
            alpha = False
        total = 10*len(dataloader)

        for i, x in zip(tqdm(range(len(dataloader))), dataloader):
            x = x.cuda()
            real_label = torch.cuda.FloatTensor(batch_size).fill_(.9)
            fake_label = torch.cuda.FloatTensor(batch_size).fill_(.1)
            noise = torch.randn(batch_size, 100, 1, 1).cuda()

            alpha_value = ((epoch*len(dataloader) + (i+1))/total)

            # Discriminateur D
            optimizerD.zero_grad()
            #print(x.size())
            #with real label
            outputTrue = netD(x, alpha=(1-alpha_value) if alpha else -1)
            lossDT = F.binary_cross_entropy_with_logits(outputTrue, real_label)

            #with false label
            outputG = netG(noise).detach()
            #print(outputG.size())
            outputFalse = netD(outputG, alpha=(1-alpha_value) if alpha else -1)
            #print(outputFalse.size())
            #print(fake_label.size())

            lossDF = F.binary_cross_entropy_with_logits(outputFalse, fake_label)

            (lossDT+lossDF).backward()
            optimizerD.step()

            #Generateur
            outputG = netG(noise, alpha=(1-alpha_value) if alpha else -1)
            outputD = netD(outputG, alpha=(1-alpha_value) if alpha else -1)
            lossG = F.binary_cross_entropy_with_logits(outputD, real_label)

            lossG.backward()
            optimizerG.step()
            break
            if(i%250 == 0):
                netG.eval()
                img = netG(noise_fixe, alpha=(1-alpha_value) if alpha else -1).data.cpu()
                tensor_back = torch.zeros(k*5+6, k*5+6, 3)
                index = 0
                netG.train()
                for w in range(5):
                    for h in range(5):
                        tensor_back[w * k + w + 1:(w + 1) * k + w + 1, h * k + h + 1:(h + 1) * k + h + 1] = img[index].transpose(0, 2).transpose(0, 1)/2+0.5
                        index += 1
                plt.imsave("/local/besnier/image_HQ/img{}".format(cpt),tensor_back)
                cpt+=1

    netG.add_layer()
    netD.add_layer()
    netG.cuda()
    netD.cuda()

    optimizerD = optim.Adam(netD.parameters(), lr=4e-5, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))

    alpha = True
















