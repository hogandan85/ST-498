# Stage 1 — Univariate model specifications (selection window <=2019Q4)

_Models fit on the stationary training series (<=2019Q4); orders, coefficients and IC are reported in stationary space. 'Full order' folds the transform's differencing back onto (log-)levels._

Significance: *** p<0.01, ** p<0.05, * p<0.10. σ² and IC are comparable **within** a variable (same stationary series), not across variables.

---

### `us_house_price_idx`  ·  group: us_house_price_idx
**Transform:** `dlog` → stationary series  ·  **Selected by test RMSE:** ARMA  

- Estimated order (stationary space): **(1,0,0)**  
- Full model order: **(1,1,0) on log-levels**  
- logLik 436.61 · AIC -869.21 · BIC -863.65 · HQIC -866.96 · σ² 3.76e-05 · Ljung–Box(8) p 0.181

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `ar.L1` | 0.8744 | 0.0367 | 23.82 | 0.000 | *** |
| `sigma2` | 3.76e-05 | 4.12e-06 | 9.13 | 0.000 | *** |


**Candidate comparison:**

| model | est. order | AIC | BIC | RMSE (stat.) | beats Mean |
|---|---|---:|---:|---:|:--:|
| Mean | (0,0,0) | -710.35 | -704.79 | 0.0089 |  |
| AR | (1,0,0) | -862.31 | -854.00 | 0.0090 |  |
| ARMA ← | (1,0,0) | -869.21 | -863.65 | 0.0088 | ✓ |
| SARMA | (1,0,0) | -869.21 | -863.65 | 0.0088 | ✓ |


### `us_consumer_confidence`  ·  group: us_consumer_confidence
**Transform:** `diff` → stationary series  ·  **Selected by test RMSE:** Mean  

- Estimated order (stationary space): **(0,0,0)**  
- Full model order: **(0,1,0)**  
- logLik -104.97 · AIC 213.93 · BIC 219.49 · HQIC — · σ² 0.3417 · Ljung–Box(8) p —

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `const` | -0.0054 | 0.0538 | -0.10 | 0.920 |  |


**Candidate comparison:**

| model | est. order | AIC | BIC | RMSE (stat.) | beats Mean |
|---|---|---:|---:|---:|:--:|
| Mean ← | (0,0,0) | 213.93 | 219.49 | 0.3698 |  |
| AR | (4,0,0) | 49.00 | 65.47 | 0.3895 |  |
| ARMA | (0,0,2) | 54.44 | 62.77 | 0.3706 |  |
| SARMA | (0,0,2)(0,0,1)[4] | 50.64 | 61.75 | 0.3706 |  |


### `us_real_gdp`  ·  group: gdp
**Transform:** `dlog` → stationary series  ·  **Selected by test RMSE:** ARMA  

- Estimated order (stationary space): **(2,0,0)**  
- Full model order: **(2,1,0) on log-levels**  
- logLik 456.09 · AIC -904.17 · BIC -893.06 · HQIC -899.66 · σ² 2.74e-05 · Ljung–Box(8) p 0.934

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `intercept` | 0.0031 | 0.0008 | 3.86 | 0.000 | *** |
| `ar.L1` | 0.3212 | 0.0812 | 3.96 | 0.000 | *** |
| `ar.L2` | 0.1823 | 0.0946 | 1.93 | 0.054 | * |
| `sigma2` | 2.74e-05 | 2.8e-06 | 9.77 | 0.000 | *** |


**Candidate comparison:**

| model | est. order | AIC | BIC | RMSE (stat.) | beats Mean |
|---|---|---:|---:|---:|:--:|
| Mean | (0,0,0) | -884.02 | -878.46 | 0.0042 |  |
| AR | (1,0,0) | -893.96 | -885.65 | 0.0042 |  |
| ARMA ← | (2,0,0) | -904.17 | -893.06 | 0.0042 | ✓ |
| SARMA | (2,0,0) | -904.17 | -893.06 | 0.0042 | ✓ |


### `us_gdp_yoy_growth`  ·  group: gdp
**Transform:** `none+diff` → stationary series  ·  **Selected by test RMSE:** ARMA  

- Estimated order (stationary space): **(1,0,0)**  
- Full model order: **(1,1,0)**  
- logLik -133.96 · AIC 271.92 · BIC 277.48 · HQIC 274.18 · σ² 0.5558 · Ljung–Box(8) p 0.001

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `ar.L1` | 0.3256 | 0.0859 | 3.79 | 0.000 | *** |
| `sigma2` | 0.5558 | 0.0562 | 9.88 | 0.000 | *** |


**Candidate comparison:**

| model | est. order | AIC | BIC | RMSE (stat.) | beats Mean |
|---|---|---:|---:|---:|:--:|
| Mean | (0,0,0) | 285.29 | 290.85 | 0.7482 |  |
| AR | (4,0,0) | 239.98 | 256.45 | 0.7907 |  |
| ARMA ← | (1,0,0) | 271.92 | 277.48 | 0.7468 | ✓ |
| SARMA | (1,0,0)(2,0,0)[4] | 236.55 | 247.66 | 0.7617 |  |


### `us_bond_yield_10y`  ·  group: us_bond_yield_10y
**Transform:** `diff` → stationary series  ·  **Selected by test RMSE:** ARMA  

- Estimated order (stationary space): **(2,0,2)**  
- Full model order: **(2,1,2)**  
- logLik -70.35 · AIC 150.70 · BIC 164.59 · HQIC 156.34 · σ² 0.1879 · Ljung–Box(8) p 0.572

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `ar.L1` | -1.2805 | 0.0620 | -20.65 | 0.000 | *** |
| `ar.L2` | -0.9213 | 0.0490 | -18.82 | 0.000 | *** |
| `ma.L1` | 1.3955 | 0.0686 | 20.35 | 0.000 | *** |
| `ma.L2` | 0.9721 | 0.0737 | 13.20 | 0.000 | *** |
| `sigma2` | 0.1879 | 0.0299 | 6.28 | 0.000 | *** |


**Candidate comparison:**

| model | est. order | AIC | BIC | RMSE (stat.) | beats Mean |
|---|---|---:|---:|---:|:--:|
| Mean | (0,0,0) | 153.60 | 159.16 | 0.4632 |  |
| AR | (4,0,0) | 149.38 | 165.85 | 0.4622 | ✓ |
| ARMA ← | (2,0,2) | 150.70 | 164.59 | 0.4334 | ✓ |
| SARMA | (0,0,0) | 153.44 | 156.21 | 0.4387 | ✓ |


### `us_bond_yield_1y`  ·  group: us_bond_yield_1y
**Transform:** `diff` → stationary series  ·  **Selected by test RMSE:** ARMA  

- Estimated order (stationary space): **(0,0,3)**  
- Full model order: **(0,1,3)**  
- logLik -73.39 · AIC 154.79 · BIC 165.90 · HQIC 159.30 · σ² 0.2003 · Ljung–Box(8) p 0.967

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `ma.L1` | 0.1681 | 0.0629 | 2.67 | 0.007 | *** |
| `ma.L2` | 0.1339 | 0.0658 | 2.04 | 0.042 | ** |
| `ma.L3` | 0.3530 | 0.0810 | 4.36 | 0.000 | *** |
| `sigma2` | 0.2003 | 0.0189 | 10.62 | 0.000 | *** |


**Candidate comparison:**

| model | est. order | AIC | BIC | RMSE (stat.) | beats Mean |
|---|---|---:|---:|---:|:--:|
| Mean | (0,0,0) | 171.80 | 177.36 | 0.6999 |  |
| AR | (4,0,0) | 152.98 | 169.45 | 0.6933 | ✓ |
| ARMA ← | (0,0,3) | 154.79 | 165.90 | 0.6817 | ✓ |
| SARMA | (3,0,0) | 156.17 | 167.29 | 0.6873 | ✓ |


### `us_term_spread`  ·  group: us_term_spread
**Transform:** `none` → stationary series  ·  **Selected by test RMSE:** Mean  

- Estimated order (stationary space): **(0,0,0)**  
- Full model order: **(0,0,0)**  
- logLik -176.43 · AIC 356.85 · BIC 362.43 · HQIC — · σ² 1.1080 · Ljung–Box(8) p —

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `const` | 1.4672 | 0.0965 | 15.21 | 0.000 | *** |


**Candidate comparison:**

| model | est. order | AIC | BIC | RMSE (stat.) | beats Mean |
|---|---|---:|---:|---:|:--:|
| Mean ← | (0,0,0) | 356.85 | 362.43 | 1.9222 |  |
| AR | (4,0,0) | 95.09 | 111.62 | 2.0085 |  |
| ARMA | (3,0,2) | 98.24 | 117.75 | 2.3585 |  |
| SARMA | (2,0,2)(1,0,0)[4] | 99.79 | 119.30 | 2.3528 |  |


### `us_reer`  ·  group: reer
**Transform:** `dlog` → stationary series  ·  **Selected by test RMSE:** AR  

- Estimated order (stationary space): **(1,0,0)**  
- Full model order: **(1,1,0) on log-levels**  
- logLik 233.26 · AIC -460.52 · BIC -452.64 · HQIC -457.33 · σ² 0.0006 · Ljung–Box(8) p 0.882

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `const` | 0.0010 | 0.0024 | 0.43 | 0.670 |  |
| `L1` | 0.1023 | 0.0987 | 1.04 | 0.300 |  |


**Candidate comparison:**

| model | est. order | AIC | BIC | RMSE (stat.) | beats Mean |
|---|---|---:|---:|---:|:--:|
| Mean | (0,0,0) | -466.68 | -461.41 | 0.0256 |  |
| AR ← | (1,0,0) | -460.52 | -452.64 | 0.0256 | ✓ |
| ARMA | (0,0,0) | -468.50 | -465.87 | 0.0258 |  |
| SARMA | (0,0,0) | -468.50 | -465.87 | 0.0258 |  |


### `us_reer_log_ret`  ·  group: reer
**Transform:** `none` → stationary series  ·  **Selected by test RMSE:** AR  

- Estimated order (stationary space): **(1,0,0)**  
- Full model order: **(1,0,0)**  
- logLik 233.26 · AIC -460.52 · BIC -452.64 · HQIC -457.33 · σ² 0.0006 · Ljung–Box(8) p 0.882

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `const` | 0.0010 | 0.0024 | 0.43 | 0.670 |  |
| `L1` | 0.1023 | 0.0987 | 1.04 | 0.300 |  |


**Candidate comparison:**

| model | est. order | AIC | BIC | RMSE (stat.) | beats Mean |
|---|---|---:|---:|---:|:--:|
| Mean | (0,0,0) | -466.68 | -461.41 | 0.0256 |  |
| AR ← | (1,0,0) | -460.52 | -452.64 | 0.0256 | ✓ |
| ARMA | (0,0,0) | -468.50 | -465.87 | 0.0258 |  |
| SARMA | (0,0,0) | -468.50 | -465.87 | 0.0258 |  |


### `us_oil_price`  ·  group: oil
**Transform:** `dlog` → stationary series  ·  **Selected by test RMSE:** ARMA  

- Estimated order (stationary space): **(0,0,2)**  
- Full model order: **(0,1,2) on log-levels**  
- logLik 31.99 · AIC -57.99 · BIC -49.65 · HQIC -54.60 · σ² 0.0341 · Ljung–Box(8) p 0.923

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `ma.L1` | -0.0751 | 0.0823 | -0.91 | 0.362 |  |
| `ma.L2` | -0.2777 | 0.0763 | -3.64 | 0.000 | *** |
| `sigma2` | 0.0341 | 0.0029 | 11.61 | 0.000 | *** |


**Candidate comparison:**

| model | est. order | AIC | BIC | RMSE (stat.) | beats Mean |
|---|---|---:|---:|---:|:--:|
| Mean | (0,0,0) | -51.15 | -45.59 | 0.1560 |  |
| AR | (3,0,0) | -76.16 | -62.39 | 0.1557 | ✓ |
| ARMA ← | (0,0,2) | -57.99 | -49.65 | 0.1547 | ✓ |
| SARMA | (0,0,0) | -52.87 | -50.09 | 0.1547 | ✓ |


### `us_oil_log_ret`  ·  group: oil
**Transform:** `none` → stationary series  ·  **Selected by test RMSE:** ARMA  

- Estimated order (stationary space): **(0,0,2)**  
- Full model order: **(0,0,2)**  
- logLik 32.60 · AIC -59.21 · BIC -50.84 · HQIC -55.81 · σ² 0.0340 · Ljung–Box(8) p 0.919

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `ma.L1` | -0.0835 | 0.0805 | -1.04 | 0.300 |  |
| `ma.L2` | -0.2614 | 0.0749 | -3.49 | 0.000 | *** |
| `sigma2` | 0.0340 | 0.0027 | 12.38 | 0.000 | *** |


**Candidate comparison:**

| model | est. order | AIC | BIC | RMSE (stat.) | beats Mean |
|---|---|---:|---:|---:|:--:|
| Mean | (0,0,0) | -52.44 | -46.87 | 0.1559 |  |
| AR | (4,0,0) | -74.20 | -57.68 | 0.1557 | ✓ |
| ARMA ← | (0,0,2) | -59.21 | -50.84 | 0.1547 | ✓ |
| SARMA | (2,0,0) | -59.47 | -51.11 | 0.1547 | ✓ |


### `us_credit`  ·  group: credit
**Transform:** `dlog+diff` → stationary series  ·  **Selected by test RMSE:** Mean  

- Estimated order (stationary space): **(0,0,0)**  
- Full model order: **(0,2,0) on log-levels**  
- logLik 449.94 · AIC -895.87 · BIC -890.33 · HQIC — · σ² 2.85e-05 · Ljung–Box(8) p —

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `const` | -5.88e-05 | 0.0005 | -0.12 | 0.905 |  |


**Candidate comparison:**

| model | est. order | AIC | BIC | RMSE (stat.) | beats Mean |
|---|---|---:|---:|---:|:--:|
| Mean ← | (0,0,0) | -895.87 | -890.33 | 0.0037 |  |
| AR | (1,0,0) | -919.62 | -911.33 | 0.0037 |  |
| ARMA | (1,0,0) | -929.56 | -924.02 | 0.0037 |  |
| SARMA | (1,0,0)(2,0,2)[4] | -946.23 | -929.60 | 0.0043 |  |


### `us_credit_qoq_growth`  ·  group: credit
**Transform:** `none+diff` → stationary series  ·  **Selected by test RMSE:** Mean  

- Estimated order (stationary space): **(0,0,0)**  
- Full model order: **(0,1,0)**  
- logLik -95.43 · AIC 194.87 · BIC 200.43 · HQIC — · σ² 0.2911 · Ljung–Box(8) p —

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `const` | -0.0079 | 0.0497 | -0.16 | 0.874 |  |


**Candidate comparison:**

| model | est. order | AIC | BIC | RMSE (stat.) | beats Mean |
|---|---|---:|---:|---:|:--:|
| Mean ← | (0,0,0) | 194.87 | 200.43 | 0.3729 |  |
| AR | (4,0,0) | 157.35 | 173.82 | 0.3753 |  |
| ARMA | (1,0,0) | 161.51 | 167.07 | 0.3742 |  |
| SARMA | (1,0,2)(2,0,1)[4] | 146.15 | 168.39 | 0.4084 |  |


### `us_credit_yoy_growth`  ·  group: credit
**Transform:** `none+diff` → stationary series  ·  **Selected by test RMSE:** Mean  

- Estimated order (stationary space): **(0,0,0)**  
- Full model order: **(0,1,0)**  
- logLik -127.72 · AIC 259.45 · BIC 265.01 · HQIC — · σ² 0.5009 · Ljung–Box(8) p —

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `const` | -0.0337 | 0.0652 | -0.52 | 0.605 |  |


**Candidate comparison:**

| model | est. order | AIC | BIC | RMSE (stat.) | beats Mean |
|---|---|---:|---:|---:|:--:|
| Mean ← | (0,0,0) | 259.45 | 265.01 | 0.7577 |  |
| AR | (4,0,0) | 171.39 | 187.86 | 0.7835 |  |
| ARMA | (5,0,0) | 171.59 | 188.26 | 0.7960 |  |
| SARMA | (1,0,0)(0,0,1)[4] | 178.97 | 187.31 | 0.7834 |  |


### `us_industrial_production`  ·  group: us_industrial_production
**Transform:** `dlog` → stationary series  ·  **Selected by test RMSE:** ARMA  

- Estimated order (stationary space): **(2,0,0)**  
- Full model order: **(2,1,0) on log-levels**  
- logLik 358.07 · AIC -708.15 · BIC -697.03 · HQIC -703.63 · σ² 0.0001 · Ljung–Box(8) p 0.718

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `intercept` | 0.0018 | 0.0012 | 1.42 | 0.157 |  |
| `ar.L1` | 0.4147 | 0.1167 | 3.55 | 0.000 | *** |
| `ar.L2` | 0.1414 | 0.0752 | 1.88 | 0.060 | * |
| `sigma2` | 0.0001 | 9.81e-06 | 14.48 | 0.000 | *** |


**Candidate comparison:**

| model | est. order | AIC | BIC | RMSE (stat.) | beats Mean |
|---|---|---:|---:|---:|:--:|
| Mean | (0,0,0) | -678.67 | -673.11 | 0.0082 |  |
| AR | (1,0,0) | -701.07 | -692.75 | 0.0082 | ✓ |
| ARMA ← | (2,0,0) | -708.15 | -697.03 | 0.0082 | ✓ |
| SARMA | (2,0,0)(0,0,1)[4] | -708.49 | -694.59 | 0.0082 |  |


### `us_vix`  ·  group: vix
**Transform:** `none` → stationary series  ·  **Selected by test RMSE:** Mean  

- Estimated order (stationary space): **(0,0,0)**  
- Full model order: **(0,0,0)**  
- logLik -407.93 · AIC 819.86 · BIC 825.43 · HQIC — · σ² 52.5075 · Ljung–Box(8) p —

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `const` | 19.2080 | 0.6643 | 28.92 | 0.000 | *** |


**Candidate comparison:**

| model | est. order | AIC | BIC | RMSE (stat.) | beats Mean |
|---|---|---:|---:|---:|:--:|
| Mean ← | (0,0,0) | 819.86 | 825.43 | 5.4122 |  |
| AR | (4,0,0) | 744.07 | 760.59 | 5.4897 |  |
| ARMA | (1,0,1) | 767.11 | 778.26 | 5.5014 |  |
| SARMA | (1,0,1) | 767.11 | 778.26 | 5.5014 |  |


### `us_vix_log_ret`  ·  group: vix
**Transform:** `none` → stationary series  ·  **Selected by test RMSE:** AR  

- Estimated order (stationary space): **(2,0,0)**  
- Full model order: **(2,0,0)**  
- logLik -3.35 · AIC 14.71 · BIC 25.76 · HQIC 19.19 · σ² 0.0620 · Ljung–Box(8) p 0.940

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `const` | -0.0068 | 0.0230 | -0.30 | 0.768 |  |
| `L1` | -0.4412 | 0.0877 | -5.03 | 0.000 | *** |
| `L2` | -0.2554 | 0.0874 | -2.92 | 0.003 | *** |


**Candidate comparison:**

| model | est. order | AIC | BIC | RMSE (stat.) | beats Mean |
|---|---|---:|---:|---:|:--:|
| Mean | (0,0,0) | 39.25 | 44.81 | 0.2298 |  |
| AR ← | (2,0,0) | 14.71 | 25.76 | 0.2298 | ✓ |
| ARMA | (0,0,1) | 14.93 | 20.49 | 0.2299 |  |
| SARMA | (0,0,1) | 14.93 | 20.49 | 0.2299 |  |


### `us_delinquency_rate`  ·  group: us_delinquency_rate
**Transform:** `diff` → stationary series  ·  **Selected by test RMSE:** ARMA  

- Estimated order (stationary space): **(1,0,0)**  
- Full model order: **(1,1,0)**  
- logLik 31.12 · AIC -58.23 · BIC -52.74 · HQIC -56.00 · σ² 0.0340 · Ljung–Box(8) p 0.166

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `ar.L1` | 0.5198 | 0.0574 | 9.06 | 0.000 | *** |
| `sigma2` | 0.0340 | 0.0025 | 13.77 | 0.000 | *** |


**Candidate comparison:**

| model | est. order | AIC | BIC | RMSE (stat.) | beats Mean |
|---|---|---:|---:|---:|:--:|
| Mean | (0,0,0) | -23.25 | -17.76 | 0.1457 |  |
| AR | (1,0,0) | -56.62 | -48.41 | 0.1483 |  |
| ARMA ← | (1,0,0) | -58.23 | -52.74 | 0.1325 | ✓ |
| SARMA | (1,0,0)(2,0,1)[4] | -68.00 | -51.53 | 0.1470 |  |


### `us_cpi`  ·  group: us_cpi
**Transform:** `none+diff` → stationary series  ·  **Selected by test RMSE:** AR  

- Estimated order (stationary space): **(4,0,0)**  
- Full model order: **(4,1,0)**  
- logLik -123.99 · AIC 259.97 · BIC 276.44 · HQIC 266.66 · σ² 0.5058 · Ljung–Box(8) p 0.003

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `const` | -0.0419 | 0.0666 | -0.63 | 0.529 |  |
| `L1` | -0.0262 | 0.0777 | -0.34 | 0.736 |  |
| `L2` | -0.1198 | 0.0777 | -1.54 | 0.123 |  |
| `L3` | 0.0227 | 0.0767 | 0.30 | 0.767 |  |
| `L4` | -0.5334 | 0.0766 | -6.97 | 0.000 | *** |


**Candidate comparison:**

| model | est. order | AIC | BIC | RMSE (stat.) | beats Mean |
|---|---|---:|---:|---:|:--:|
| Mean | (0,0,0) | 306.43 | 311.99 | 0.9095 |  |
| AR ← | (4,0,0) | 259.97 | 276.44 | 0.9060 | ✓ |
| ARMA | (4,0,0) | 270.09 | 283.98 | 0.9139 |  |
| SARMA | (0,0,0)(0,0,2)[4] | 231.52 | 239.86 | 0.9167 |  |


### `us_unemployment`  ·  group: us_unemployment
**Transform:** `diff` → stationary series  ·  **Selected by test RMSE:** ARMA  

- Estimated order (stationary space): **(1,0,3)**  
- Full model order: **(1,1,3)**  
- logLik 1.51 · AIC 6.98 · BIC 20.87 · HQIC 12.62 · σ² 0.0560 · Ljung–Box(8) p 0.166

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `ar.L1` | -0.4297 | 0.1551 | -2.77 | 0.006 | *** |
| `ma.L1` | 0.8697 | 0.1245 | 6.99 | 0.000 | *** |
| `ma.L2` | 0.8081 | 0.0826 | 9.78 | 0.000 | *** |
| `ma.L3` | 0.7091 | 0.0860 | 8.25 | 0.000 | *** |
| `sigma2` | 0.0560 | 0.0066 | 8.54 | 0.000 | *** |


**Candidate comparison:**

| model | est. order | AIC | BIC | RMSE (stat.) | beats Mean |
|---|---|---:|---:|---:|:--:|
| Mean | (0,0,0) | 62.58 | 68.13 | 0.1290 |  |
| AR | (2,0,0) | 11.33 | 22.38 | 0.1361 |  |
| ARMA ← | (1,0,3) | 6.98 | 20.87 | 0.1251 | ✓ |
| SARMA | (2,0,0)(2,0,1)[4] | -2.86 | 13.81 | 0.1271 | ✓ |


### `us_sp500_close`  ·  group: sp500
**Transform:** `dlog` → stationary series  ·  **Selected by test RMSE:** AR  

- Estimated order (stationary space): **(2,0,0)**  
- Full model order: **(2,1,0) on log-levels**  
- logLik 136.45 · AIC -264.90 · BIC -253.85 · HQIC -260.42 · σ² 0.0057 · Ljung–Box(8) p 0.894

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `const` | 0.0190 | 0.0073 | 2.59 | 0.010 | *** |
| `L1` | 0.0385 | 0.0906 | 0.43 | 0.671 |  |
| `L2` | 0.0223 | 0.0905 | 0.25 | 0.805 |  |


**Candidate comparison:**

| model | est. order | AIC | BIC | RMSE (stat.) | beats Mean |
|---|---|---:|---:|---:|:--:|
| Mean | (0,0,0) | -269.72 | -264.17 | 0.0753 |  |
| AR ← | (2,0,0) | -264.90 | -253.85 | 0.0753 | ✓ |
| ARMA | (0,0,0) | -269.72 | -264.17 | 0.0753 |  |
| SARMA | (0,0,0) | -269.72 | -264.17 | 0.0753 |  |


### `us_sp500_log_ret`  ·  group: sp500
**Transform:** `none` → stationary series  ·  **Selected by test RMSE:** AR  

- Estimated order (stationary space): **(1,0,0)**  
- Full model order: **(1,0,0)**  
- logLik 136.91 · AIC -267.82 · BIC -259.48 · HQIC -264.43 · σ² 0.0059 · Ljung–Box(8) p 0.875

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `const` | 0.0184 | 0.0072 | 2.56 | 0.011 | ** |
| `L1` | 0.0278 | 0.0917 | 0.30 | 0.762 |  |


**Candidate comparison:**

| model | est. order | AIC | BIC | RMSE (stat.) | beats Mean |
|---|---|---:|---:|---:|:--:|
| Mean | (0,0,0) | -272.46 | -266.89 | 0.0754 |  |
| AR ← | (1,0,0) | -267.82 | -259.48 | 0.0753 | ✓ |
| ARMA | (0,0,0) | -272.46 | -266.89 | 0.0754 |  |
| SARMA | (0,0,0) | -272.46 | -266.89 | 0.0754 |  |


