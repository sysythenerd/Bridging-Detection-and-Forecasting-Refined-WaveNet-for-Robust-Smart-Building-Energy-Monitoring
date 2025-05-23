# Bridging-Detection-and-Forecasting-Refined-WaveNet-for-Robust-Smart-Building-Energy-Monitoring
A **Refined Gated WaveNet architecture** is used in this paper to detect anomalies in univariate energy consumption data without supervision. Time series from smart building sensors are used to test the method, which is trained solely on normal data and assessed against artificial anomaly injections.

**Goal**: Use a Refined WaveNet model to detect anomalies in time series of energy use.

**Structure**: WaveNet featuring a refinement head based on GRU or CNN, dilated causal convolutions, skip connections, and gated activations.

*Learning Setting**: Completely unsupervised; only normal data is used to train the model.

**Output**: Export of anomaly indices and detection of anomalies using evaluation metrics.

