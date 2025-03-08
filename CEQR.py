import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import gamma_cal_loss
from quantilelosses import loss_evi
from matplotlib.lines import Line2D

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
set_seeds(0)

# function with added noise
def f_with_noise(x, y): 
    base_function = np.cos(x) * np.exp(-(x**2 + y**2) / 20) + x * np.sin(y) + np.sin(x**2) / 2 + (x**2 + y**2) / 50
    noise_scale = 2.0 * np.exp(-0.3 * np.sqrt(x**2 + y**2))
    noise = np.random.normal(loc=0, scale=noise_scale, size=noise_scale.shape)
    return base_function + noise

output_size = 1
num_quantiles = 50
evi_coeff = 5.0 # 0.5 for non QR loss

# model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.output_1 = nn.Sequential(
            nn.Linear(2, 64),  # 2 input features
            nn.PReLU(),
            nn.Linear(64, 64),
            nn.PReLU(),
            nn.Linear(64, output_size) # 1 output value
        )

        self.NIG = nn.Sequential(
            nn.Linear(2, 256),  # 2 input features
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, 4*output_size*2) # 1 output value
        )

        init_modules = [self.output_1, self.NIG]

        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def call(self, out):
        mu, logv, logalpha, logbeta = torch.split(out, output_size*2, dim=-1)

        v = F.softplus(logv)
        alpha = F.softplus(logalpha) + 1
        beta = F.softplus(logbeta)
        return torch.concat([mu, v, alpha, beta], axis=-1)

    def forward(self, xy):
        output1 = self.output_1(xy)
        nig = self.NIG(xy)
        output2 = self.call(nig)

        return output1, output2

model = SimpleNN()

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x_batch, y_batch, z_batch in loader:
        x_batch, y_batch, z_batch = x_batch.to(device), y_batch.to(device), z_batch.to(device)
        xy = torch.column_stack((x_batch,y_batch))
        optimizer.zero_grad()
        output1, output2 = model(xy)
        gamma, v, alpha, beta = torch.split(output2, output_size*2, dim=-1)
        gamma = gamma.view(x_batch.shape[0], 1, 2).squeeze()
        v = v.view(x_batch.shape[0], 1, 2).squeeze()
        alpha = alpha.view(x_batch.shape[0], 1, 2).squeeze()
        beta = beta.view(x_batch.shape[0], 1, 2).squeeze()
        mse_loss = criterion(output1.squeeze(), z_batch)
        loss_evidence = loss_evi(z_batch, gamma, v, alpha, beta, evi_coeff)
        loss_gamma_cal = gamma_cal_loss(gamma,z_batch,0.5,device)
        loss = mse_loss + loss_evidence + loss_gamma_cal
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss

def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output1, output2 = model(data)
            gamma, v, alpha, beta = torch.split(output2, output_size*2, dim=-1)
            gamma = gamma.view(data.shape[0], 1, 2).squeeze()
            v = v.view(data.shape[0], 1, 2).squeeze()
            alpha = alpha.view(data.shape[0], 1, 2).squeeze()
            beta = beta.view(data.shape[0], 1, 2).squeeze()
            mse_loss = criterion(output1, target)
            loss_evidence = loss_evi(target, gamma, v, alpha, beta, evi_coeff)
            loss_gamma_cal = gamma_cal_loss(gamma,target,0.5,device)
            loss = mse_loss + loss_evidence + loss_gamma_cal
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss

# example synthetic data generation for demonstration purposes
x = np.random.uniform(-5, 5, 500)
y = np.random.uniform(-5, 5, 500)
x = x[np.argsort(x)]
y = y[np.argsort(y)]
x, y = np.meshgrid(x, y)
z = f_with_noise(x, y)

# convert to tensors
x_tensor = torch.tensor(x.flatten(), dtype=torch.float32)
y_tensor = torch.tensor(y.flatten(), dtype=torch.float32)
z_tensor = torch.tensor(z.flatten(), dtype=torch.float32)

# create a dataset and dataloader for training
dataset = TensorDataset(x_tensor, y_tensor, z_tensor)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

model.to(device)

# regression loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    #test_loss = test(model, test_loader, criterion, device)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss}")

################# print plot of training samples

model.eval()
xy_tensor = torch.column_stack((x_tensor,y_tensor)).to(device)
with torch.no_grad():
    predicted_values, output2 = model(xy_tensor)
    predicted_values = predicted_values.cpu().numpy()

gamma, v, alpha, beta = torch.split(output2, output_size*2, dim=-1)
gamma = gamma.cpu().numpy()
v = v.cpu().numpy()
alpha = alpha.cpu().numpy()
beta = beta.cpu().numpy()

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Helvetica']
plt.rcParams.update({'font.size': 16})

# plot the surface (including upper and lower quantile) with noise
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

upper_q = gamma[:,1].reshape(500,500)
lower_q = gamma[:,0].reshape(500,500)

surface95 = ax.plot_surface(x, y, gamma[:,1].reshape(500,500), cmap='Blues', edgecolor='none')
surface = ax.plot_surface(x, y, z, cmap='Greys', edgecolor='none')
surface05 = ax.plot_surface(x, y, gamma[:,0].reshape(500,500), cmap='Reds', edgecolor='none')

# condition: Z1 < Z2 < Z3
condition_satisfied_strict = np.logical_and(lower_q <= z, z <= upper_q)

# count how many values between upper and low quantile
count_strict = np.sum(condition_satisfied_strict)

margcov = (count_strict / lower_q.size) * 100
print(f"Percentage of values where Z2 is between Z1 and Z3: {margcov:.2f}%")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('F(X,Y)')

surface95._facecolors2d = surface95._facecolor3d
surface95._edgecolors2d = surface95._edgecolor3d
surface._facecolors2d = surface._facecolor3d
surface._edgecolors2d = surface._edgecolor3d
surface05._facecolors2d = surface05._facecolor3d
surface05._edgecolors2d = surface05._edgecolor3d

legend_elements = [
    Line2D([0], [0], color='blue', lw=2, label='Upper Quantile (95%)'),
    Line2D([0], [0], color='black', lw=2, label='Ground Truth w/ Noise'),
    Line2D([0], [0], color='red', lw=2, label='Lower Quantile (5%)')
]

plt.title('Conformalized Evidential Quantile Regression',y=1.025)
plt.legend(handles=legend_elements,loc='upper center',bbox_to_anchor=(0.5, 0.025), ncol=3,prop={'size':13})
plt.tight_layout()
plt.savefig('CEQR_3D.png',dpi=300,bbox_inches='tight')
plt.show()

################## evaluate model on fresh test set
    
x = np.random.uniform(-15, 15, 350)
y = np.random.uniform(-15, 15, 350)
x = x[np.argsort(x)]
y = y[np.argsort(y)]
x, y = np.meshgrid(x, y)
z = f_with_noise(x, y)

x_tensor = torch.tensor(x.flatten(), dtype=torch.float32)
y_tensor = torch.tensor(y.flatten(), dtype=torch.float32)
z_tensor = torch.tensor(z.flatten(), dtype=torch.float32)

model.eval()
xy_tensor = torch.column_stack((x_tensor,y_tensor)).to(device)
with torch.no_grad():
    predicted_values, output2 = model(xy_tensor)
    predicted_values = predicted_values.cpu().numpy()

gamma, v, alpha, beta = torch.split(output2, output_size*2, dim=-1)
gamma = gamma.cpu().numpy()
v = v.cpu().numpy()
alpha = alpha.cpu().numpy()
beta = beta.cpu().numpy()

var = np.sqrt((beta /(v*(alpha - 1))))
upper = gamma[:,1].reshape(350,350)
lower = gamma[:,0].reshape(350,350)
upper_var = var[:,1].reshape(350,350)
lower_var = var[:,0].reshape(350,350)
aleatoric = abs(upper-lower)
epistemic_95 = abs((upper+upper_var) - (upper-upper_var))
epistemic_05 = abs((lower+lower_var) - (lower-lower_var))
epistemic = epistemic_95 + epistemic_05

fig = plt.figure(figsize=(10,10))
ax_aleatoric = fig.add_subplot(111, projection='3d')
surface = ax_aleatoric.plot_surface(x, y, aleatoric, cmap='viridis', edgecolor='k')

# specify the range to change the pane color
x_range = [-5, 5.25]
y_range = [-5.5, 5.25]
z_plane = -0.25  # appropriate z level depending on the plot
# create a rectangle in the specified range
X_fill = np.array([[x_range[0], x_range[0]], [x_range[1], x_range[1]]])
Y_fill = np.array([[y_range[0], y_range[1]], [y_range[0], y_range[1]]])
Z_fill = np.array([[z_plane, z_plane], [z_plane, z_plane]])
# plot the rectangle as a semi-transparent surface
ax_aleatoric.plot_surface(X_fill, Y_fill, Z_fill,color='white')
# set pane color to light purple using RGB
ax_aleatoric.get_zaxis().set_pane_color((0.9, 0.7, 0.9, 1.0))  # RGBA for a light purple

ax_aleatoric.set_xlabel('X')
ax_aleatoric.set_ylabel('Y')
ax_aleatoric.set_zlabel('Magnitude')

for i in [1,2,6,7]:
    plt.setp(ax_aleatoric.get_xticklabels()[i],color='purple')
    plt.setp(ax_aleatoric.get_yticklabels()[i],color='purple')


plt.title('Aleatoric Uncertainty',y=1.025)
plt.colorbar(surface,fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('aleatoric_3D.png',dpi=300,bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(10,10))
ax_epistemic = fig.add_subplot(111, projection='3d')
surface = ax_epistemic.plot_surface(x, y, epistemic, cmap='magma', edgecolor='k')

x_range = [-5, 5.25]
y_range = [-5.5, 5.25]
z_plane = -0.25 

X_fill = np.array([[x_range[0], x_range[0]], [x_range[1], x_range[1]]])
Y_fill = np.array([[y_range[0], y_range[1]], [y_range[0], y_range[1]]])
Z_fill = np.array([[z_plane, z_plane], [z_plane, z_plane]])

ax_epistemic.plot_surface(X_fill, Y_fill, Z_fill, color='white')

ax_epistemic.get_zaxis().set_pane_color((0.9, 0.7, 0.9, 1.0))

ax_epistemic.set_xlabel('X')
ax_epistemic.set_ylabel('Y')
ax_epistemic.set_zlabel('Magnitude')

for i in [1,2,6,7]:
    plt.setp(ax_epistemic.get_xticklabels()[i],color='purple')
    plt.setp(ax_epistemic.get_yticklabels()[i],color='purple')

plt.title('Epistemic Uncertainty',y=1.025)
plt.colorbar(surface,fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('epistemic_3D.png',dpi=300,bbox_inches='tight')
plt.show()