# Independent continuous-behavior confirmation

The frozen study completed 99 cases and 693 dynamics lanes.
The frozen protocol decision was **pass**.
Calibration used 19 separate null seeds. Positive and null test sets used 20 and 60 seeds.

## Primary two-dimensional joint encoder

| Dynamics | Behavioral MAE | Positive-unit MAE | Semantic MAE | Origin MAE | Null false seeds | Exact 95% upper bound | Failed gates |
|---|---:|---:|---:|---:|---:|---:|---|
| oracle | 0.01011 | 0.02387 | 0.01011 | 0.05828 | 0/60 | 4.8703% | — |
| known_support | 0.01023 | 0.02440 | 0.01011 | 0.05828 | 0/60 | 4.8703% | — |
| empty_support | 0.10758 | 0.43033 | 0.01011 | 0.05828 | 0/60 | 4.8703% | behavioral_mae, positive_unit_mae, neuron_power, group_power |
| conventional_cgc | 0.01170 | 0.03023 | 0.01011 | 0.05828 | 0/60 | 4.8703% | — |
| conventional_cgc_star | 0.01310 | 0.03567 | 0.01011 | 0.05828 | 0/60 | 4.8703% | — |
| fast_cgc | 0.01095 | 0.02724 | 0.01011 | 0.05828 | 0/60 | 4.8703% | — |
| fast_cgc_star | 0.01346 | 0.03707 | 0.01011 | 0.05828 | 0/60 | 4.8703% | — |

## Learned readouts at controlled known support

| Readout | Overall MAE mean ± SD | Active MAE mean ± SD | Semantic MAE | Null MAE |
|---|---:|---:|---:|---:|
| PCA, 2D | 0.047896 ± 0.005793 | 0.099147 ± 0.016373 | 0.047899 | 0.008990 |
| PCA, 4D | 0.057818 ± 0.007448 | 0.108513 ± 0.020234 | 0.057883 | 0.007957 |
| PLS, 2D | 0.048440 ± 0.006002 | 0.099720 ± 0.017019 | 0.048458 | 0.008885 |
| Full neural linear | 0.045009 ± 0.006002 | 0.117130 ± 0.021214 | 0.045180 | 0.002257 |
| Behavior-only network, 2D | 0.009498 ± 0.002204 | 0.025298 ± 0.008328 | 0.009349 | 0.000876 |
| Joint network, 2D | 0.010231 ± 0.002161 | 0.024396 ± 0.007536 | 0.010105 | 0.001170 |
| Joint network, 4D | 0.008492 ± 0.002069 | 0.021641 ± 0.006977 | 0.008264 | 0.000721 |
| Full neural nonlinear | 0.007746 ± 0.001895 | 0.021285 ± 0.006607 | 0.007471 | 0.000668 |

## Paired comparisons on the 20 positive seeds

Positive differences favor the joint encoder. The values below summarize the secondary comparisons.

| Dynamics | Comparator | Mean MAE improvement | Exploratory 95% interval | Better seeds | Holm p |
|---|---|---:|---:|---:|---:|
| known_support | PCA, 2D | 0.037666 | [0.035545, 0.039754] | 20/20 | 2.861e-05 |
| known_support | PLS, 2D | 0.038210 | [0.035965, 0.040369] | 20/20 | 2.861e-05 |
| known_support | Behavior-only network, 2D | -0.000733 | [-0.001133, -0.000335] | 3/20 | 0.0077305 |
| conventional_cgc | PCA, 2D | 0.036596 | [0.034453, 0.038825] | 20/20 | 2.861e-05 |
| conventional_cgc | PLS, 2D | 0.037114 | [0.034895, 0.039410] | 20/20 | 2.861e-05 |
| conventional_cgc | Behavior-only network, 2D | -0.000691 | [-0.001120, -0.000258] | 4/20 | 0.011818 |
| conventional_cgc_star | PCA, 2D | 0.035720 | [0.033886, 0.037642] | 20/20 | 2.861e-05 |
| conventional_cgc_star | PLS, 2D | 0.036332 | [0.034372, 0.038357] | 20/20 | 2.861e-05 |
| conventional_cgc_star | Behavior-only network, 2D | -0.000934 | [-0.001322, -0.000539] | 2/20 | 0.0016098 |
| fast_cgc | PCA, 2D | 0.037103 | [0.034849, 0.039362] | 20/20 | 2.861e-05 |
| fast_cgc | PLS, 2D | 0.037656 | [0.035319, 0.039967] | 20/20 | 2.861e-05 |
| fast_cgc | Behavior-only network, 2D | -0.000775 | [-0.001182, -0.000368] | 3/20 | 0.0077305 |
| fast_cgc_star | PCA, 2D | 0.035599 | [0.033377, 0.037752] | 20/20 | 2.861e-05 |
| fast_cgc_star | PLS, 2D | 0.036197 | [0.033863, 0.038456] | 20/20 | 2.861e-05 |
| fast_cgc_star | Behavior-only network, 2D | -0.000890 | [-0.001232, -0.000543] | 1/20 | 0.00020027 |

The independent unit is the generator seed. Error units follow the new continuous behavior.
The confirmation covers the declared fully observed, model-matched synthetic system.
The frozen comparison between learned readouts addresses behavioral-effect accuracy. The additional sign tests show how consistently those differences favor a given encoder across seeds.
