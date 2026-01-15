Goal: Verify the agent's ability to simulate autoregressive data and recover directed connectivity patterns using Frites.
Prompt:
1. Inspect and take inspiration from the Jupyter notebooks in @braina/examples
2. Generate code and Jupyter Notebooks of the following problems 3 and 4. Save the Jupyter Notebooks as braina_solutions_TD.ipynb in /braina/usecases/master_td/
3. Auto-regressive model
Simulate a simple autoregressive model composed of a source signal X and a target signal Y (pairwise AR model). Generate a Jupyter notebook with 
        a. The signals must oscillated at 40Hz 
        b. Generate 100 trials with a single stimulus and plot the single trial data and causal coupling over time. Plot the time series using the plot() function 
        c. Generate 100 trials with a three stimuli and plot the single trial data and causal coupling over time
        d. Write the equation of the AR model generating the data. Hint: look at the source code
4. Single-trial time-resolved (dynamic) Functional Connectivity
Take the data simulated in 1b and estimate dynamic functional connectivity between X and Y with the following parameters: window length = 0.75 seconds, win_step = 0.02  seconds
        a. Computed the DFC for each trial and plot the average time course across trials
        b. Test different window_length values and plot different examples, try to interpret the results

