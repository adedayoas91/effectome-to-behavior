| Method | Scenario | Eligible seeds/planned | Top-k recall [95% CI] | Group wins [95% CI] | Null upper bound | Decision |
|---|---|---:|---|---|---|---|
| conventional_cgc | single_driver | 20/20 | 0.275 [0.150, 0.425] | 0.475 [0.300, 0.650] | NA | fail |
| conventional_cgc | redundant_group | 20/20 | unavailable | 0.125 [0.050, 0.225] | NA | fail |
| conventional_cgc | correlated_bystander | 20/20 | 0.050 [0.000, 0.125] | 0.425 [0.300, 0.550] | NA | fail |
| conventional_cgc | null | 60/60 | unavailable | 0.408 [0.325, 0.508] | 0.880 | fail |
| conventional_cgc | hidden_confounder | 20/20 | unavailable | 0.675 [0.525, 0.825] | NA | diagnostic |
| conventional_cgc_star | single_driver | 20/20 | 0.075 [0.000, 0.175] | 0.450 [0.300, 0.600] | NA | fail |
| conventional_cgc_star | redundant_group | 20/20 | unavailable | 0.275 [0.125, 0.425] | NA | fail |
| conventional_cgc_star | correlated_bystander | 20/20 | 0.050 [0.000, 0.125] | 0.425 [0.275, 0.600] | NA | fail |
| conventional_cgc_star | null | 60/60 | unavailable | 0.492 [0.392, 0.592] | 0.498 | fail |
| conventional_cgc_star | hidden_confounder | 20/20 | unavailable | 0.525 [0.375, 0.700] | NA | diagnostic |
| fast_cgc | single_driver | 20/20 | 0.200 [0.075, 0.326] | 0.500 [0.350, 0.650] | NA | fail |
| fast_cgc | redundant_group | 20/20 | unavailable | 0.150 [0.025, 0.300] | NA | fail |
| fast_cgc | correlated_bystander | 20/20 | 0.200 [0.100, 0.300] | 0.500 [0.375, 0.625] | NA | fail |
| fast_cgc | null | 60/60 | unavailable | 0.358 [0.267, 0.450] | 0.796 | fail |
| fast_cgc | hidden_confounder | 20/20 | unavailable | 0.675 [0.575, 0.775] | NA | diagnostic |
| fast_cgc_star | single_driver | 20/20 | 0.025 [0.000, 0.075] | 0.375 [0.250, 0.501] | NA | fail |
| fast_cgc_star | redundant_group | 20/20 | unavailable | 0.225 [0.100, 0.375] | NA | fail |
| fast_cgc_star | correlated_bystander | 20/20 | 0.075 [0.000, 0.150] | 0.375 [0.225, 0.525] | NA | fail |
| fast_cgc_star | null | 60/60 | unavailable | 0.458 [0.367, 0.558] | 0.514 | fail |
| fast_cgc_star | hidden_confounder | 20/20 | unavailable | 0.325 [0.200, 0.450] | NA | diagnostic |
