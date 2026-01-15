  Use Case: Directed Connectivity Analysis (Granger Causality)
  Goal: Verify the agent's ability to simulate autoregressive data and recover directed connectivity patterns using Frites.
  Prompt:
  "I want to analyze directed connectivity in a simulated brain network.
  1. First, use the frites_sim_ar tool to generate a simulated dataset with an autoregressive model (use ar_type='hga', n_epochs=50, n_times=300) and save it to sim_ar_data.nc. Save the python code you used to generate the code to sim_ar.py
  2. Next, compute the covariance-based Granger Causality on this dataset using the frites_conn_covgc tool. Set dt=10, lag=2, and t0=[100, 150, 200]. Save the result to gc_result.nc.
  3. Finally, write and execute a Python script to load gc_result.nc and plot the mean connectivity matrix as a heatmap, saving the figure to gc_heatmap.png.
  4. Store the files in the directory /braina/usecases/Granger_causality

