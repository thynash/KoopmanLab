from koopmanlab.models import kno
from koopmanlab import utils
from koopmanlab.models import koopmanViT
from koopmanlab.physics import NavierStokesLoss

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer

class koopman:
    def __init__(self, backbone = "KNO1d", autoencoder = "MLP", o = 16, m = 16, r = 8, t_in = 1, device = False, lambda_phy=0.0, nu=1e-3, dx=1/64, dy=1/64, dt=1.0):
        self.backbone = backbone
        self.autoencoder = autoencoder
        self.operator_size = o
        self.modes = m
        self.decompose = r
        self.device = device
        self.t_in = t_in
        self.lambda_phy = lambda_phy
        self.physics = NavierStokesLoss(nu=nu, dx=dx, dy=dy, dt=dt)
        # Core Model
        self.params = 0
        self.kernel = False
        # Opt Setting
        self.optimizer = False
        self.scheduler = False
        self.loss = torch.nn.MSELoss()
    def compile(self):
        if self.autoencoder == "MLP":
            encoder = kno.encoder_mlp(self.t_in, self.operator_size)
            decoder = kno.decoder_mlp(self.t_in, self.operator_size)
            print("The autoencoder type is MLP.")
        elif self.autoencoder == "Conv1d":
            encoder = kno.encoder_conv1d(self.t_in, self.operator_size)
            decoder = kno.decoder_conv1d(self.t_in, self.operator_size)
            print("The autoencoder type is Conv1d.")
        elif self.autoencoder == "Conv2d":
            encoder = kno.encoder_conv2d(self.t_in, self.operator_size)
            decoder = kno.decoder_conv2d(self.t_in, self.operator_size)
            print("The autoencoder type is Conv2d.")
        else:
            print("Wrong!")
        if self.backbone == "KNO1d":
            self.kernel = kno.KNO1d(encoder, decoder, self.operator_size, modes_x = self.modes, decompose = self.decompose).to(self.device)
            print("KNO1d model is completed.")
        elif self.backbone == "KNO2d":
            self.kernel = kno.KNO2d(encoder, decoder, self.operator_size, modes_x = self.modes, modes_y = self.modes,decompose = self.decompose).to(self.device)
            print("KNO2d model is completed.")
        if not self.kernel == False:
            self.params = utils.count_params(self.kernel)
            print("Koopman Model has been compiled!")
            print("The Model Parameters Number is ",self.params)
    def opt_init(self, opt, lr, step_size, gamma):
        if opt == "Adam":
            self.optimizer = utils.Adam(self.kernel.parameters(), lr= lr, weight_decay=1e-4)
        if not step_size == False:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def train_single(self, epochs, trainloader, evalloader = False):
        for ep in range(epochs):
            # Train
            self.kernel.train()
            t1 = default_timer()
            train_recons_full = 0
            train_pred_full = 0
            train_phy_full = 0
            for xx, yy in trainloader:
                bs = xx.shape[0]
                xx = xx.to(self.device)
                yy = yy.to(self.device)
                pred,im_re = self.kernel(xx)
                l_recons = self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))
                current = xx[..., -1:]
                predicted = pred[..., -1:]
                l_phy = self.physics(current, predicted)
                train_pred_full += l_pred.item()
                train_recons_full += l_recons.item()
                train_phy_full += l_phy.item()
                loss = 5*l_pred + 0.5*l_recons + self.lambda_phy*l_phy
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            train_pred_full /= len(trainloader)
            train_recons_full /= len(trainloader)
            train_phy_full /= len(trainloader)
            t2 = default_timer()
            test_pred_full = 0
            test_recons_full = 0
            test_phy_full = 0
            if evalloader:
                with torch.no_grad():
                    for xx, yy in evalloader:
                        bs = xx.shape[0]
                        xx = xx.to(self.device)
                        yy = yy.to(self.device)
                        pred,im_re = self.kernel(xx)
                        l_recons = self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                        l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))
                        current = xx[..., -1:]
                        predicted = pred[..., -1:]
                        l_phy = self.physics(current, predicted)
                        test_pred_full += l_pred.item()
                        test_recons_full += l_recons.item()
                        test_phy_full += l_phy.item()
                test_pred_full /= len(evalloader)
                test_recons_full /= len(evalloader)
                test_phy_full /= len(evalloader)
            if self.scheduler is not None:
                self.scheduler.step()
            if evalloader:
                if ep == 0:
                    print("Epoch", "Time", "[Train Recon]", "[Train Pred]", "[Train Phys]", "[Eval Recon]", "[Eval Pred]", "[Eval Phys]")
                print(ep, t2 - t1, train_recons_full, train_pred_full, train_phy_full, test_recons_full, test_pred_full, test_phy_full)
            else:
                if ep == 0:
                    print("Epoch", "Time", "Train Recon", "Train Pred", "Train Phys")
                print(ep, t2 - t1, train_recons_full, train_pred_full, train_phy_full)

    def test_single(self, testloader):
        test_pred_full = 0
        test_recons_full = 0
        test_phy_full = 0
        with torch.no_grad():
            for xx, yy in testloader:
                bs = xx.shape[0]
                xx = xx.to(self.device)
                yy = yy.to(self.device)
                pred,im_re = self.kernel(xx)
                l_recons = self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))
                current = xx[..., -1:]
                predicted = pred[..., -1:]
                l_phy = self.physics(current, predicted)
                test_pred_full += l_pred.item()
                test_recons_full += l_recons.item()
                test_phy_full += l_phy.item()
        test_pred_full /= len(testloader)
        test_recons_full /= len(testloader)
        test_phy_full /= len(testloader)
        print("Total prediction test mse error is ",test_pred_full)
        print("Total reconstruction test mse error is ",test_recons_full)
        print("Total physics test mse error is ",test_phy_full)
        return test_pred_full

    def train(self, epochs, trainloader, step = 1, T_out = 40, evalloader = False):
        T_eval = T_out
        for ep in range(epochs):
            self.kernel.train()
            t1 = default_timer()
            train_recons_full = 0
            train_pred_full = 0
            train_phy_full = 0
            for xx, yy in trainloader:
                l_recons = 0
                l_phy = 0
                xx = xx.to(self.device)
                yy = yy.to(self.device)
                bs = xx.shape[0]
                for t in range(0, T_out):
                    im,im_re = self.kernel(xx)
                    l_recons += self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                    current = xx[..., -1:]
                    predicted = im[..., -1:]
                    l_phy_step = self.physics(current, predicted)
                    if t == 0:
                        l_phy = l_phy_step
                        pred = im[...,-1:]
                    else:
                        l_phy += l_phy_step
                        pred = torch.cat((pred, im[...,-1:]), -1)
                    xx = torch.cat((xx[..., step:], im[...,-1:]), dim=-1)
                l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))
                l_recons /= T_out
                l_phy /= T_out
                loss = 5 * l_pred + 0.5 * l_recons + self.lambda_phy * l_phy
                train_pred_full += l_pred.item()
                train_recons_full += l_recons.item()
                train_phy_full += l_phy.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            train_pred_full /= len(trainloader)
            train_recons_full /= len(trainloader)
            train_phy_full /= len(trainloader)
            t2 = default_timer()
            test_pred_full = 0
            test_recons_full = 0
            test_phy_full = 0
            if evalloader:
                with torch.no_grad():
                    for xx, yy in evalloader:
                        bs = xx.shape[0]
                        l_recons = 0
                        l_phy = 0
                        xx = xx.to(self.device)
                        yy = yy.to(self.device)
                        for t in range(0, T_eval):
                            im, im_re = self.kernel(xx)
                            l_recons += self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                            current = xx[..., -1:]
                            predicted = im[..., -1:]
                            l_phy_step = self.physics(current, predicted)
                            if t == 0:
                                l_phy = l_phy_step
                                pred = im[...,-1:]
                            else:
                                l_phy += l_phy_step
                                pred = torch.cat((pred, im[...,-1:]), -1)
                            xx = torch.cat((xx[..., 1:], im[...,-1:]), dim=-1)
                        l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))
                        test_recons_full += l_recons.item() / T_eval
                        test_pred_full += l_pred.item()
                        test_phy_full += l_phy.item() / T_eval
                test_recons_full /= len(evalloader)
                test_pred_full /= len(evalloader)
                test_phy_full /= len(evalloader)
            if self.scheduler is not None:
                self.scheduler.step()
            if evalloader:
                if ep == 0:
                    print("Epoch", "Time", "[Train Recon]", "[Train Pred]", "[Train Phys]", "[Eval Recon]", "[Eval Pred]", "[Eval Phys]")
                print(ep, t2 - t1, train_recons_full, train_pred_full, train_phy_full, test_recons_full, test_pred_full, test_phy_full)
            else:
                if ep == 0:
                    print("Epoch", "Time", "Train Recon", "Train Pred", "Train Phys")
                print(ep, t2 - t1, train_recons_full, train_pred_full, train_phy_full)

    def train_semisupervised(self, epochs, labeled_loader, unlabeled_loader, evalloader=None, step=1, T_out=40):
        history = {
            "train_pred": [], "train_recon": [], "train_phy": [],
            "unlabel_recon": [], "unlabel_phy": [],
            "eval_pred": [], "eval_recon": [], "eval_phy": []
        }
        for ep in range(epochs):
            self.kernel.train()
            t1 = default_timer()
            tr_pred, tr_recon, tr_phy = 0, 0, 0
            if labeled_loader is not None:
                for xx, yy in labeled_loader:
                    xx, yy = xx.to(self.device), yy.to(self.device)
                    bs = xx.shape[0]
                    l_recons, l_phy = 0, 0
                    for t in range(T_out):
                        im, im_re = self.kernel(xx)
                        l_recons += self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                        cur, nxt = xx[..., -1:], im[..., -1:]
                        l_phy_step = self.physics(cur, nxt)
                        if t == 0:
                            l_phy = l_phy_step
                            pred = nxt
                        else:
                            l_phy += l_phy_step
                            pred = torch.cat((pred, nxt), -1)
                        xx = torch.cat((xx[..., step:], nxt), dim=-1)
                    l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))
                    l_recons /= T_out
                    l_phy /= T_out
                    loss = 5 * l_pred + 0.5 * l_recons + self.lambda_phy * l_phy
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    tr_pred += l_pred.item()
                    tr_recon += l_recons.item()
                    tr_phy += l_phy.item()

            un_recon, un_phy = 0, 0
            if unlabeled_loader is not None:
                for xx in unlabeled_loader:
                    if isinstance(xx, (list, tuple)): xx = xx[0]
                    xx = xx.to(self.device)
                    bs = xx.shape[0]
                    l_recons, l_phy = 0, 0
                    for t in range(T_out):
                        im, im_re = self.kernel(xx)
                        l_recons += self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                        cur, nxt = xx[..., -1:], im[..., -1:]
                        l_phy_step = self.physics(cur, nxt)
                        l_phy = l_phy_step if t == 0 else l_phy + l_phy_step
                        xx = torch.cat((xx[..., step:], nxt), dim=-1)
                    l_recons /= T_out
                    l_phy /= T_out
                    loss = 0.5 * l_recons + self.lambda_phy * l_phy
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    un_recon += l_recons.item()
                    un_phy += l_phy.item()

            ev_pred, ev_recon, ev_phy = 0, 0, 0
            if evalloader:
                self.kernel.eval()
                with torch.no_grad():
                    for xx, yy in evalloader:
                        xx, yy = xx.to(self.device), yy.to(self.device)
                        bs = xx.shape[0]
                        l_recons, l_phy = 0, 0
                        for t in range(T_out):
                            im, im_re = self.kernel(xx)
                            l_recons += self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                            cur, nxt = xx[..., -1:], im[..., -1:]
                            l_phy_step = self.physics(cur, nxt)
                            if t == 0:
                                l_phy, pred = l_phy_step, nxt
                            else:
                                l_phy += l_phy_step
                                pred = torch.cat((pred, nxt), -1)
                            xx = torch.cat((xx[..., 1:], nxt), dim=-1)
                        l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))
                        ev_pred += l_pred.item()
                        ev_recon += (l_recons.item() / T_out)
                        ev_phy += (l_phy.item() / T_out)
                ev_pred /= len(evalloader); ev_recon /= len(evalloader); ev_phy /= len(evalloader)
            
            if self.scheduler is not None: self.scheduler.step()
            
            if labeled_loader is not None:
                history["train_pred"].append(tr_pred/len(labeled_loader))
                history["train_recon"].append(tr_recon/len(labeled_loader))
                history["train_phy"].append(tr_phy/len(labeled_loader))
            else:
                history["train_pred"].append(np.nan)
                history["train_recon"].append(np.nan)
                history["train_phy"].append(np.nan)

            if unlabeled_loader is not None:
                history["unlabel_recon"].append(un_recon/len(unlabeled_loader))
                history["unlabel_phy"].append(un_phy/len(unlabeled_loader))
            else:
                history["unlabel_recon"].append(np.nan)
                history["unlabel_phy"].append(np.nan)

            history["eval_pred"].append(ev_pred); history["eval_recon"].append(ev_recon); history["eval_phy"].append(ev_phy)
            
            print(f"Epoch {ep} | Labeled(Pred:{history['train_pred'][-1]:.6f} Recon:{history['train_recon'][-1]:.6f}) | Unlabeled(Recon:{history['unlabel_recon'][-1]:.6f})")
        return history

    def test(self, testloader, step = 1, T_out = 40, path = False, is_save = False, is_plot = False):
        time_error = torch.zeros([T_out,1])
        test_pred_full = 0
        test_recons_full = 0
        test_phy_full = 0
        loc = 0
        with torch.no_grad():
            for xx, yy in testloader:
                bs = xx.shape[0]
                xx = xx.to(self.device)
                yy = yy.to(self.device)
                l_recons = 0
                l_phy = 0
                for t in range(0, T_out):
                    y = yy[..., t:t + 1]
                    im, im_re = self.kernel(xx)
                    l_recons += self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                    t_error = self.loss(im[...,-1:],y)
                    current = xx[..., -1:]
                    predicted = im[..., -1:]
                    l_phy_step = self.physics(current, predicted)
                    if t == 0:
                        l_phy = l_phy_step
                        pred = im[...,-1:]
                    else:
                        l_phy += l_phy_step
                        pred = torch.cat((pred, im[...,-1:]), -1)
                    time_error[t] = time_error[t] + t_error.item()
                    xx = torch.cat((xx[..., 1:], im[...,-1:]), dim=-1)
                l_recons /= T_out
                l_phy /= T_out
                l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))
                test_recons_full += l_recons.item()
                test_pred_full += l_pred.item()
                test_phy_full += l_phy.item()
                if(loc == 0 and is_save):
                    torch.save({"pred":pred, "yy":yy}, path+ "pred_yy.pt")
                if(loc == 0 ) and is_plot:
                    for i in range(T_out):
                        plt.subplot(1,3,1)
                        plt.title("Predict")
                        plt.imshow(pred[0,...,i].cpu().detach().numpy())
                        plt.subplot(1,3,2)
                        plt.imshow(yy[0,...,i].cpu().detach().numpy())
                        plt.title("Label")
                        plt.subplot(1,3,3)
                        plt.imshow(pred[0,...,i].cpu().detach().numpy()-yy[0,...,i].cpu().detach().numpy())
                        plt.title("Error")
                        plt.show()
                        plt.savefig(path + "time_"+str(i)+".png")
                        plt.close()
                loc = loc + 1
        test_pred_full /= loc
        test_recons_full /= loc
        test_phy_full /= loc
        time_error /= len(testloader)
        print("Total prediction test mse error is ",test_pred_full)
        print("Total reconstruction test mse error is ",test_recons_full)
        print("Total physics test mse error is ",test_phy_full)
        return time_error

    def save(self, path):
        (fpath,_) = os.path.split(path)
        if not os.path.isfile(fpath):
            os.makedirs(fpath)
        torch.save({"koopman":self,"model":self.kernel,"model_params":self.kernel.state_dict()}, path)

class koopman_vit:
    def __init__(self, decoder = "Conv2d", depth = 16, resolution=(256, 256), patch_size=(4, 4),
            in_chans=1, out_chans=1, embed_dim=768, parallel = False, device = False):
        # Model Hyper-parameters
        self.decoder = decoder
        self.resolution = resolution
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.depth = depth
        # Core Model
        self.params = 0
        self.kernel = False
        # Opt Setting
        self.optimizer = False
        self.scheduler = False
        self.device = device
        self.parallel = parallel
        self.loss = torch.nn.MSELoss()
    def compile(self):
        self.kernel = koopmanViT.ViT(img_size=self.resolution, patch_size=self.patch_size, in_chans=self.in_chans, out_chans=self.out_chans, num_blocks=self.num_blocks, embed_dim = self.embed_dim, depth=self.depth, settings = self.decoder).to(self.device)
        if self.parallel:
            self.kernel = torch.nn.DataParallel(self.kernel)
        self.params = utils.count_params(self.kernel)
        print("Koopman Fourier Vision Transformer has been compiled!")
        print("The Model Parameters Number is ",self.params)
    def opt_init(self, opt, lr, step_size, gamma):
        if opt == "Adam":
            self.optimizer = utils.Adam(self.kernel.parameters(), lr= lr, weight_decay=1e-4)
        if not step_size == False:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
    def train_multi(self, epochs, trainloader, T_out = 10, evalloader = False):
        T_eval = T_out
        for ep in range(epochs):
            self.kernel.train()
            t1 = default_timer()
            train_recons_full = 0
            train_pred_full = 0
            for xx, yy in trainloader:
                l_recons = 0
                xx = xx.to(self.device)
                yy = yy.to(self.device)
                bs = xx.shape[0]
                for t in range(0, T_out):
                    y = yy[:, t:t + 1]
                    im,im_re = self.kernel(xx)
                    l_recons += self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                    if t == 0:
                        pred = im[:, -1:]
                    else:
                        pred = torch.cat((pred, im[:, -1:]), -1)
                    xx = im
                l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))
                loss = 5 * l_pred + 0.5 * l_recons
                train_pred_full += l_pred.item()
                train_recons_full += l_recons.item()/T_out
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            train_pred_full = train_pred_full / len(trainloader)
            train_recons_full = train_recons_full / len(trainloader)
            t2 = default_timer()
            test_pred_full = 0
            test_recons_full = 0
            if evalloader:
                with torch.no_grad():
                    for xx, yy in evalloader:
                        xx = xx.to(self.device)
                        yy = yy.to(self.device)
                        for t in range(0, T_eval):
                            im, im_re = self.kernel(xx)
                            l_recons += self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                            if t == 0:
                                pred = im
                            else:
                                pred = torch.cat((pred, im), 1)
                            xx = im
                        l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))
                        test_recons_full += l_recons.item() / T_eval
                        test_pred_full += l_pred.item()
                test_recons_full = test_recons_full / len(evalloader)
                test_pred_full = test_pred_full / len(evalloader)
            self.scheduler.step()
            if evalloader:
                if ep == 0:
                    print("Epoch","Time","[Train Recons MSE]","[Train Pred MSE]","[Eval Recons MSE]","[Eval Pred MSE]")
                print(ep, t2 - t1, train_recons_full, train_pred_full, test_recons_full, test_pred_full)
            else:
                if ep == 0:
                    print("Epoch","Time","Train Recons MSE","Train Pred MSE")
                print(ep, t2 - t1, train_recons_full, train_pred_full)
    def test_multi(self, testloader, step = 1, T_out = 5, path = False, is_save = False, is_plot = False):
        time_error = torch.zeros([T_out,1])
        test_pred_full = 0
        test_recons_full = 0
        loc = 0
        with torch.no_grad():
            for xx, yy in testloader:
                bs = xx.shape[0]
                xx = xx.to(self.device)
                yy = yy.to(self.device)
                l_recons = 0
                for t in range(0, T_out):
                    y = yy[:, t:t + 1]
                    im, im_re = self.kernel(xx)
                    l_recons += self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                    t_error = self.loss(im, y)
                    xx = im
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), 1)
                    time_error[t] = time_error[t] + t_error.item()
                test_recons_full += l_recons.item() / T_out
                l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))
                test_pred_full += l_pred.item()
                if(loc == 0 & is_save):
                    torch.save({"pred":pred, "yy":yy}, path+ "pred_yy.pt")
                if(loc == 0 & is_plot):
                    for i in range(T_out):
                        plt.subplot(1,3,1)
                        plt.title("Predict")
                        plt.imshow(pred[0,i].cpu().detach().numpy())
                        plt.subplot(1,3,2)
                        plt.imshow(yy[0,i].cpu().detach().numpy())
                        plt.title("Label")
                        plt.subplot(1,3,3)
                        plt.imshow(pred[0,i].cpu().detach().numpy()-yy[0,i].cpu().detach().numpy())
                        plt.title("Error")
                        plt.show()
                        plt.savefig(path + "time_"+str(i)+".png")
                        plt.close()
                loc = loc + 1
        test_pred_full = test_pred_full / loc
        test_recons_full = test_recons_full / loc
        time_error = time_error / len(testloader)
        print("Total prediction test mse error is ",test_pred_full)
        print("Total reconstruction test mse error is ",test_recons_full)
        return time_error
    def train_single(self, epochs, trainloader, evalloader = False):
        for ep in range(epochs):
            self.kernel.train()
            t1 = default_timer()
            train_recons_full = 0
            train_pred_full = 0
            for x, y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                bs = x.shape[0]
                im,im_re = self.kernel(x)
                l_recons = self.loss(im_re.reshape(bs, -1), x.reshape(bs, -1))
                l_pred = self.loss(im.reshape(bs, -1), y.reshape(bs, -1))
                loss = 5 * l_pred + 0.5 * l_recons
                train_pred_full += l_pred.item()
                train_recons_full += l_recons.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            train_pred_full = train_pred_full / len(trainloader)
            train_recons_full = train_recons_full / len(trainloader)
            t2 = default_timer()
            test_pred_full = 0
            test_recons_full = 0
            if evalloader:
                with torch.no_grad():
                    for x, y in evalloader:
                        x = x.to(self.device)
                        y = y.to(self.device)
                        im, im_re = self.kernel(x)
                        l_recons = self.loss(im_re.reshape(bs, -1), x.reshape(bs, -1))
                        l_pred = self.loss(im.reshape(bs, -1), y.reshape(bs, -1))
                        test_recons_full += l_recons.item()
                        test_pred_full += l_pred.item()
                test_recons_full = test_recons_full / len(evalloader)
                test_pred_full = test_pred_full / len(evalloader)
            self.scheduler.step()
            if evalloader:
                if ep == 0:
                    print("Epoch","Time","[Train Recons MSE]","[Train Pred MSE]","[Eval Recons MSE]","[Eval Pred MSE]")
                print(ep, t2 - t1, train_recons_full, train_pred_full, test_recons_full, test_pred_full)
            else:
                if ep == 0:
                    print("Epoch","Time","Train Recons MSE","Train Pred MSE")
                print(ep, t2 - t1, train_recons_full, train_pred_full)
    def test_single(self, testloader, T_out = 1, path = False, is_save = False, is_plot = False):
        time_error = torch.zeros([T_out,1])
        test_pred_full = 0
        test_recons_full = 0
        loc = 0
        with torch.no_grad():
            for xx, yy in testloader:
                bs = xx.shape[0]
                xx = xx.to(self.device)
                yy = yy.to(self.device)
                l_recons = 0
                for t in range(0, T_out):
                    y = yy[:, t:t + 1]
                    im, im_re = self.kernel(xx)
                    l_recons += self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                    t_error = self.loss(im, y)
                    xx = im
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), 1)
                    time_error[t] = time_error[t] + t_error.item()
                test_recons_full += l_recons.item() / T_out
                l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))
                test_pred_full += l_pred.item()
                if(loc == 0 & is_save):
                    torch.save({"pred":pred, "yy":yy}, path+ "pred_yy.pt")
                if(loc == 0 & is_plot):
                    for i in range(T_out):
                        plt.subplot(1,3,1)
                        plt.title("Predict")
                        plt.imshow(pred[0,i].cpu().detach().numpy())
                        plt.subplot(1,3,2)
                        plt.imshow(yy[0,i].cpu().detach().numpy())
                        plt.title("Label")
                        plt.subplot(1,3,3)
                        plt.imshow(pred[0,i].cpu().detach().numpy()-yy[0,i].cpu().detach().numpy())
                        plt.title("Error")
                        plt.show()
                        plt.savefig(path + "time_"+str(i)+".png")
                        plt.close()
                loc = loc + 1
        test_pred_full = test_pred_full / len(testloader)
        test_recons_full = test_recons_full / len(testloader)
        time_error = time_error / len(testloader)
        print("Total prediction test mse error is ",test_pred_full)
        print("Total reconstruction test mse error is ",test_recons_full)
        return time_error
    def save(self, path):
        torch.save({"koopman":self,"model":self.kernel,"model_params":self.kernel.state_dict()}, path)
