#%% import pytorch 
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt

# %%

# simulate data
gaussian_centers = torch.tensor([[-10., -10], [6, 6], [8, -10]])  # 3 centers of 2-d gaussian

gaussian_std = torch.tensor(0.1)
num_samples = 100


#%%
seed = 42
torch.manual_seed(seed)

Gaussian = MultivariateNormal(gaussian_centers.reshape(-1), 
                              covariance_matrix=torch.eye(gaussian_centers.shape[0] * gaussian_centers.shape[1]) * gaussian_std)

data = Gaussian.sample((num_samples,))
data = torch.cat(torch.split(data, gaussian_centers.shape[1], dim=1))  # z
# %%

plt.scatter(data[:, 0], data[:, 1])
plt.scatter(gaussian_centers[:, 0], gaussian_centers[:, 1], c='red', marker='x')
plt.title('Simulated Data')
# %%
# Linear time schedular Gaussian Flow matching 
T = 1000 

# X = alpha_t * z + beta_t * e, X_0 = e, X_T = z
z = data
alpha_t = torch.linspace(0, 1, T)
beta_t = torch.linspace(1, 0, T)


data_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

# learn the ODE flow gradient u(x, t) = dX_t/dt
theta_model = nn.Sequential(
    nn.Linear(3, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

optimizer = torch.optim.Adam(theta_model.parameters(), lr=0.001)
loss_fun = nn.MSELoss()

Loss_record= []

total_epochs = 100
for epoch in range(100):
    print(f'Epoch {epoch+1}/{total_epochs}')
    for z in data_loader:
        t = torch.randint(0, T, (z.shape[0],)) # random time steps
        e = torch.normal(mean= 0., std = 1., size=z.shape)  # noise
        x = alpha_t[t].unsqueeze(-1) * z + beta_t[t].unsqueeze(-1) * e  # X_t = alpha_t * z + beta_t * e
        
        input_data = torch.cat((x, t.unsqueeze(1)/T), dim=1)
        
        
        flow_gradient_pred = theta_model(input_data)
        optimizer.zero_grad()
        loss = loss_fun(flow_gradient_pred, z - e)
        loss.backward()
        optimizer.step()
        Loss_record.append(loss.item())
        
        
#%%
plt.plot(Loss_record)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss for Flow Matching Model')



# %%
# Sample from trained model
X_0 = torch.normal(mean=0., std=1., size=(100, 2))

with torch.no_grad():
    x = X_0.clone()
    for t in range(T):
        t_norm = torch.full((X_0.shape[0],), t / T)
        input_data = torch.cat((x, t_norm.unsqueeze(1)), dim=1)
        flow_gradient_pred = theta_model(input_data)
        x = x +  1/T * flow_gradient_pred  # Euler step
        
    
    
    
#%%
plt.scatter(x[:, 0], x[:, 1], c='blue', marker='o', label='Sampled Data from Flow Matching')
plt.scatter(gaussian_centers[:, 0], gaussian_centers[:, 1], c='red', marker='x', label='Gaussian Centers')
plt.scatter(X_0[:, 0], X_0[:, 1], c='green', marker='o', label='Initial Samples')
plt.xlabel('X-axis')
plt.title('Sampled Data from Flow Matching Model')
plt.legend()
# %%
