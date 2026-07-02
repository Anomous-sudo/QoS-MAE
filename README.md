# QoS-MAE

QoS-MAE: A Masked Autoencoder with Reputation Modeling for Web Service QoS Prediction.

## Environment

pip install -r requirements.txt


## Parameters

| Item | Response Time | Throughput |
|---|---:|---:|
| encoder dim `D` | 256 | 128 |
| decoder dim `D'` | 128 | 64 |
| encoder layers `Ne` | 6 | 6 |
| decoder layers `Nd` | 3 | 3 |
| mask rate `rho` | 0.75 | 0.75 |
| `beta` | 0.5 | 0.5 |
| `gamma` | 0.3 | 0.3 |
| `alpha` | 0.6 | 0.6 |

## Config files:

```text
configs/rt.yaml
configs/tp.yaml
```
