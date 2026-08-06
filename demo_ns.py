import torch
import koopmanlab as kp
# Setting your computing device
device = torch.device("cpu")
print("Using device:", device)

# Path
data_path = "/content/NavierStokes_V1e-5_N1200_T20.mat"
fig_path = "./content"
save_path = "./content"

# Loading Data
train_loader, test_loader = kp.data.navier_stokes(data_path, batch_size = 10, T_in = 10, T_out = 10, type = "1e-5", sub = 1)

# Hyper parameters
ep = 1 # Training Epoch
o = 32 # Koopman Operator Size
m = 16 # Modes
r = 8 # Power of Koopman Matrix

# Model
koopman_model = kp.model.koopman(backbone = "KNO2d", autoencoder = "MLP", o = o, m = m, r = r, t_in = 10, device = device)
koopman_model.compile()
koopman_model.opt_init("Adam", lr = 0.005, step_size=100, gamma=0.5)
koopman_model.train(epochs=ep, trainloader = train_loader, evalloader = test_loader,T_out=10)

# Result and Saving
time_error = koopman_model.test(test_loader, path = fig_path, T_out=10,is_save = True, is_plot = True)
filename = "ns_time_error_op" + str(o) + "m" + str(m) + "r" +str(r) + ".pt"
torch.save({"time_error":time_error,"params":koopman_model.params}, save_path + filename)
