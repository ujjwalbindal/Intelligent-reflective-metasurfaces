import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ------------------- Device & Constants -------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

M = N = 32
input_res = 128
input_dim = input_res * input_res
output_dim = M * N
wavelength = 1.0
d = wavelength / 4

# ------------------- Physics -------------------
def target_intensity_pattern(theta_users, phi_users, theta_vals, phi_vals, array_beamfactor=14):
    T_grid, P_grid = np.meshgrid(theta_vals, phi_vals, indexing='ij')
    I = np.zeros_like(T_grid)
    for t_u, p_u in zip(theta_users, phi_users):
        Sx = np.sin(T_grid)*np.cos(P_grid) - np.sin(t_u)*np.cos(p_u)
        Sy = np.sin(T_grid)*np.sin(P_grid) - np.sin(t_u)*np.sin(p_u)
        sinc_x = np.sinc(array_beamfactor * Sx)
        sinc_y = np.sinc(array_beamfactor * Sy)
        beam = (np.cos(T_grid)**2) * (np.abs(sinc_x * sinc_y)**2)
        I += beam
    I /= len(theta_users)
    I /= I.max()
    return I

def phase_vector_to_intensity(phase_vector, M, N, wavelength, d, theta_vals, phi_vals):
    phase_profile = phase_vector.reshape(M, N)
    k = 2 * np.pi / wavelength
    x = np.arange(N) 
    x = d * x
    y = np.arange(M) 
    y = d * y
    TH, PH = np.meshgrid(theta_vals, phi_vals, indexing='ij')
    sintheta = np.sin(TH)
    costheta = np.cos(TH)
    AF = np.zeros_like(TH, dtype=complex)
    for m in range(M):
        for n in range(N):
            psi = (k * (
                x[n] * sintheta*np.cos(PH) +
                y[m] * sintheta * np.sin(PH)
            ) + phase_profile[m, n])
            AF += np.exp(1j * psi)
    intensity = (costheta ** 2) * (np.abs(AF) ** 2)
    intensity /= np.max(intensity)
    return intensity

# ------------------- Data Generation -------------------
def create_data_sample(resolution, num_users, theta_max_deg):
    theta_vals = np.linspace(0, np.deg2rad(theta_max_deg), resolution)
    phi_vals = np.linspace(0, 2 * np.pi, resolution)
    theta_users = np.random.uniform(0, np.deg2rad(theta_max_deg-10), num_users)
    phi_users = np.random.uniform(0, 2*np.pi, num_users)
    intensity = target_intensity_pattern(theta_users, phi_users, theta_vals, phi_vals)
    return intensity

def create_dataset(n_samples=80, resolution=128, num_users=10, theta_max_deg=60):
    images = []
    for _ in range(n_samples):
        img = create_data_sample(resolution, num_users, theta_max_deg)
        images.append(img)
    images = np.stack(images).reshape(n_samples, -1)
    return torch.tensor(images, dtype=torch.float32)

# ------------------- Neural Net with Xavier Init -------------------
class MetaSurfMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, output_dim)
        self.lrelu = nn.LeakyReLU(0.01)
        self.init_weights()

    def forward(self, x):
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = self.fc3(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

# ------------------- Weighted Physics Loss (per paper) -------------------
def compute_weighted_mse(target_intensity, pred_intensity, resolution=128):
    # Calculate weights (exp(8 ln(2) * (theta/(pi/2))^2))
    theta_vals = torch.linspace(0, np.pi / 2, resolution)
    phi_vals = torch.linspace(0, 2 * np.pi, resolution)
    theta_grid, phi_grid = torch.meshgrid(theta_vals, phi_vals, indexing="ij")
    weights = np.exp(8 * torch.log(torch.tensor(2)) * (theta_grid / (torch.pi/2))**2)
    # Target and prediction come in shape (flat,)
    diff2 = weights.ravel() * (pred_intensity.ravel() - target_intensity.ravel())**2
    return diff2.mean()

# ------------------- Training Config -------------------
num_epochs = 1000
batch_size = 64
theta_max_deg = 60
resolution = 128
num_users = 3
test_sample = 16
# ----- Generate initial DB set -----
initial_samples = 64
train_set = create_dataset(initial_samples, resolution, num_users, theta_max_deg)
test_set = create_dataset(test_sample, resolution, num_users, theta_max_deg)  # separate test set
print(test_set.shape)
X_train = train_set.to(device)
X_test = test_set.to(device)

model = MetaSurfMLP(input_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-4)

# ------- Train/test loop -------
for epoch in range(num_epochs):
    # --- Generate new batch (64) and select 64 from database ---
    new_samples = create_dataset(batch_size, resolution, num_users, theta_max_deg).to(device)
    # Choose 64 random old DB samples (handle small DB size gracefully for early epochs)
    if X_train.size(0) > batch_size:
        perm = torch.randperm(X_train.size(0))[:batch_size]
        db_samples = X_train[perm]
    else:
        db_samples = X_train

    # Build batch
    train_inputs = torch.cat([new_samples, db_samples], dim=0)

    model.train()
    optimizer.zero_grad()
    pred_phases = model(train_inputs) % (2 * np.pi)
    loss = 0
    for i in range(train_inputs.shape[0]):
        pp = pred_phases[i].detach().cpu().numpy()
        tt = train_inputs[i].detach().cpu().numpy()
        pred_intensity = phase_vector_to_intensity(
            pp, M, N, wavelength, d,
            np.linspace(0, np.deg2rad(theta_max_deg), resolution),
            np.linspace(0, 2*np.pi, resolution)
        )
        loss += compute_weighted_mse(tt, pred_intensity, resolution)
    loss = torch.tensor(loss / train_inputs.shape[0], requires_grad=True)
    print(train_inputs.shape)
    loss.backward()
    optimizer.step()

    # Merge both new/old data into train set to grow DB
    X_train = torch.cat([X_train.cpu(), new_samples.cpu()], dim=0).to(device)

    print(f"Epoch {epoch+1} / {num_epochs} | Train weighted loss: {loss.item():.5f} | DB size: {X_train.size(0)}")

# ----------- Evaluate on test set and visualize ---------------
model.eval()
with torch.no_grad():
    test_mse, test_corr = 0, 0
    pred_intensities = []
    pred_phases_list = []
    for i in range(X_test.shape[0]):
        input_flat = X_test[i].to(device)
        pred_phase_flat = model(input_flat.unsqueeze(0)).cpu().numpy() % (2 * np.pi)
        input_img = input_flat.cpu().numpy()
        pred_intensity = phase_vector_to_intensity(
            pred_phase_flat, M, N, wavelength, d,
            np.linspace(0, np.deg2rad(theta_max_deg), resolution),
            np.linspace(0, 2*np.pi, resolution)
        )
        pred_intensities.append(pred_intensity)
        pred_phases_list.append(pred_phase_flat.reshape(M, N))
        # MSE/correlation
        test_mse += ((pred_intensity.ravel() - input_img.ravel())**2).mean()
        test_corr += np.corrcoef(pred_intensity.ravel(), input_img.ravel())[0, 1]
    test_mse /= X_test.shape[0]
    test_corr /= X_test.shape[0]
    print(f"[Test] Avg. MSE: {test_mse:.6f} | Avg. Correlation: {test_corr:.3f}")

    # -- Display for a single random test example --
    i = np.random.randint(len(X_test))
    input_intensity = X_test[i].cpu().numpy().reshape((resolution, resolution))
    pred_phase = pred_phases_list[i]
    print(pred_phase)
    pred_intensity = pred_intensities[i].reshape((resolution, resolution))
    print(pred_intensity)
    # Use pcolormesh polar for all
    theta_vals = np.linspace(0, np.deg2rad(theta_max_deg), resolution)
    phi_vals = np.linspace(0, 2*np.pi, resolution)
    R, P = np.meshgrid(np.rad2deg(theta_vals), phi_vals, indexing='ij')
    Xp = R * np.cos(P)
    Yp = R * np.sin(P)
    theta_32 = np.linspace(0, np.deg2rad(theta_max_deg), M)
    phi_32 = np.linspace(0, 2*np.pi, N)
    R_32, P_32 = np.meshgrid(np.rad2deg(theta_32), phi_32, indexing='ij')
    X_32 = R_32 * np.cos(P_32)
    Y_32 = R_32 * np.sin(P_32)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    pcm1 = axs[0].pcolormesh(Xp, Yp, input_intensity, cmap='jet', shading='auto')
    axs[0].set_title("Target Intensity (Polar)")
    axs[0].axis('equal')
    plt.colorbar(pcm1, ax=axs, shrink=0.9)
    pcm2 = axs[1].pcolormesh(X_32, Y_32, pred_phase % (2*np.pi), cmap='twilight', shading='auto', vmin=0, vmax=2*np.pi)
    axs[1].set_title("Predicted Phase (32x32, Polar)")
    axs[1].axis('equal')
    plt.colorbar(pcm2, ax=axs[1], shrink=0.9, ticks=[0, np.pi, 2*np.pi], label='Phase [0, 2π]')
    pcm3 = axs[2].pcolormesh(Xp, Yp, pred_intensity, cmap='jet', shading='auto')
    axs[2].set_title("Predicted Intensity (Polar)")
    axs[2].axis('equal')
    plt.colorbar(pcm3, ax=axs[2], shrink=0.9)
    plt.tight_layout()
    plt.show()
