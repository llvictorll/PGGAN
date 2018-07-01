import torch.nn as nn
import torch.nn.functional as F


class NetG(nn.Module):
    def __init__(self, nz=100, nc=3, ngf=64):
        super(NetG, self).__init__()
        self.nz = nz  # dimension du bruit en entrée
        self.nc = nc  # dim de sortie en RGB =3
        self.ngf = ngf  # dimension en sortie de G
        self.cngf = ngf  # dimension "progressive"
        self.f_block = self._first_block()  # initialisation du 1er block
        self.mlist = nn.ModuleList()  # init liste de blocks intermediaires
        self.mlist.append(self.f_block)
        self.mlist.append(self._intermediate_block())
        self.block_to_image = self._block_to_RGB(self.cngf//2)  # On met ici le block chargé de transfo le block en image
        self.prev_b2img = self.block_to_image  # sauvegarde la "block2img" qui precede

    def _first_block(self):
        # 1er block qui prend en entrée Z
        block = nn.Sequential(
            nn.ConvTranspose2d(self.nz, self.cngf, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.cngf),
            nn.ReLU()
        )
        return block

    def _intermediate_block(self):
        # block a ajouter apres chaque chgmt de taille
        block = nn.Sequential(
            nn.ConvTranspose2d(self.cngf, self.cngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.cngf // 2),
            nn.ReLU()
        )
        return block

    def _block_to_RGB(self, c_in):
        # ce block permet de prendre en entrée un Tensor de taille c_in
        # et de retourner un image
        block = nn.Sequential(
            nn.ConvTranspose2d(c_in, self.nc, 1, 1, 0, bias=False),
            nn.ReLU(),
            nn.Tanh()
        )
        return block

    def add_layer(self):
        self.mlist.append(self._intermediate_block())  # on ajoute un block intermedaire
        self.cngf = self.cngf // 2  # maj de la taille de sortie
        self.prev_b2img = self.block_to_image  # maj de l'ancienne b2img
        self.block_to_image = self._block_to_RGB(self.cngf)  # on créer un nouveau block2img avec les bonnes dim

    def forward(self, x, alpha=-1):
        x_copy = x

        for i, module in enumerate(self.mlist):
            #print(x_copy.size())
            outputx = module(x_copy)
            #print(outputx.size())
            is_last_block = i == len(self.mlist) - 1

            if is_last_block and alpha > 0:
                x_copy.detach()
                imgx = self.block_to_image(outputx)
                x_copy = alpha * self.prev_b2img(F.upsample(x_copy, scale_factor=2)).detach() + (1 - alpha) * imgx

            elif (is_last_block):
                x_copy = self.block_to_image(outputx)
            else:
                x_copy = outputx
            #print("end of module")

        return x_copy


class NetD(nn.Module):
    def __init__(self, nc=3, ngf=64):
        super(NetD, self).__init__()
        self.nc = nc
        self.ngf = ngf
        self.cngf = ngf
        self.last_block = self._last_block()
        self.mlist = nn.ModuleList()
        self.mlist.append(self._intermediate_block())
        self.mlist.append(self.last_block)
        self.image_to_block = self._rgb_to_block(self.cngf//2)
        self.prev_img2b = self.image_to_block

    def _last_block(self, c_out=1):
        block = nn.Sequential(
            nn.Conv2d(self.ngf, c_out, 4, 1, 0, bias=False)
        )
        return block

    def _intermediate_block(self):
        block = nn.Sequential(
            nn.Conv2d(self.cngf//2, self.cngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.cngf),
            nn.LeakyReLU(0.2)
        )
        return block

    def _rgb_to_block(self, c_out):
        block = nn.Sequential(
            nn.Conv2d(self.nc, c_out, 1,1,0,bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.2)
        )
        return block

    def add_layer(self):
        new_list = nn.ModuleList()
        new_list.append(self._intermediate_block())

        for module in self.mlist:
            new_list.append(module)

        self.mlist = new_list


        self.cngf = self.cngf // 2                             # maj de la taille de sortie
        self.prev_img2b = self.image_to_block                  # maj de l'ancienne img2b
        self.image_to_block = self._rgb_to_block(self.cngf)    # on créer un nouveau img2block avec les bonnes dim

    def forward(self, x, alpha=-1):
        x_copy = x
        for i, module in enumerate(self.mlist):
            is_first_block = i == 0


            if (is_first_block):
                #print("it's a first block")
                x_copy = self.image_to_block(x_copy)
                #print("ok")


            #print(x_copy.size())
            outputx = module(x_copy)
            #print(outputx.size())

            if (is_first_block and alpha>0):
                #print("first block with a>0")
                x_copy = alpha * self.prev_img2b(F.adaptive_avg_pool2d(x_copy, outputx.size(2))).detach() + (
                            1 - alpha) * outputx
            else:
                #print("it isn't a first block")
                x_copy = outputx
        #print("end")
        return x_copy.squeeze()




























