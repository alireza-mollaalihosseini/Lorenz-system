# Learning the Lorenz Attractor with a Convolutional VAE

University of Padova coursework (M.Sc. Physics of Data, 2020). A variational autoencoder (TensorFlow /
Keras) learns a compact latent representation of trajectory windows of the chaotic Lorenz system.

## Pipeline

1. **Simulation:** the Lorenz equations (σ = 10, ρ = 50, β = 4/3) integrated with `scipy.integrate.odeint`
   (Δt = 0.01). The transient is discarded and the three coordinates are min–max scaled.
2. **Windows:** the trajectory is split into windows of 100 time steps × 3 channels (x, y, z).
3. **Model:**
   * encoder: Conv1D (32, 64, 64 filters) + max-pooling + dropout → dense → 4-dimensional latent space
     with the reparameterisation trick;
   * decoder: dense → Conv1DTranspose layers back to 100 × 3;
   * L2 regularisation.
4. **Training:** custom `train_step` with reconstruction + KL loss, Adam, early stopping.
5. **Evaluation:** per-coordinate reconstruction error, comparison plots of the original and
   reconstructed signals and 3D attractor, and a check on a new trajectory started from a different
   initial condition.

## Run

Open `Optimum_solution.ipynb` (written for Google Colab). Requirements: `tensorflow`, `numpy`, `scipy`,
`pandas`, `scikit-learn`, `matplotlib`, `seaborn`.
