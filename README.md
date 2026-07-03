# QoS-MAE

QoS-MAE: A Masked Autoencoder with Reputation Modeling for Web Service QoS Prediction. We proposed a QoS prediction method based on masked autoencoders (QoS-MAE), which first applies a region-aware MF method to complete the QoS matrix and assigns a reputation score to each QoS observation, and then adopts a reputation-guided masking strategy that preferentially masks low-reputation observations and reconstructs them from contextual information. 

## Environment

pip install -r requirements.txt


## Parameters

For QoS-MAE, the parameters are set to $N_e = 6$, $N_d = 3$, $\rho = 0.7$, $\beta = 0.5$, $\gamma = 0.3$, $p = 16$, $\lambda = 1$, $\eta_1 = 0.001$, and $\eta_2 = 0.001$. For the response time dataset, the parameters are set to \(D=256\), \(D'=128\), and \(\alpha=0.6\). For the throughput dataset, they are set to \(D=128\), \(D'=64\), and \(\alpha=0.4\).

## Config files:

```text
configs/rt.yaml
configs/tp.yaml
```
