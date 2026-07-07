# Stage 1 — Forward (production) model specifications (full-sample refit)

_Production forecasts: the CV-selected model per series refit on the full sample and projected 20 quarters. Parameters differ from the hold-out fit because the estimation window now includes 2020+._

Significance: *** p<0.01, ** p<0.05, * p<0.10. σ² and IC are comparable **within** a variable (same stationary series), not across variables.

---

### `us_real_gdp`  ·  group: gdp
**Transform:** `dlog` → stationary series  ·  **Selected by CV (full-sample refit):** Mean  

- Estimated order (stationary space): **(0,0,0)**  
- Full model order: **(0,1,0) on log-levels**  
- logLik 441.33 · AIC -878.67 · BIC -872.74 · HQIC — · σ² 0.0001 · Ljung–Box(8) p —

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `const` | 0.0061 | 0.0009 | 6.58 | 0.000 | *** |



### `us_unemployment`  ·  group: us_unemployment
**Transform:** `diff` → stationary series  ·  **Selected by CV (full-sample refit):** ARMA  

- Estimated order (stationary space): **(0,0,0)**  
- Full model order: **(0,1,0)**  
- logLik -151.63 · AIC 305.26 · BIC 308.22 · HQIC 306.46 · σ² 0.4881 · Ljung–Box(8) p 0.987

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `sigma2` | 0.4881 | 0.0107 | 45.59 | 0.000 | *** |



### `us_cpi`  ·  group: us_cpi
**Transform:** `none+diff` → stationary series  ·  **Selected by CV (full-sample refit):** Mean  

- Estimated order (stationary space): **(0,0,0)**  
- Full model order: **(0,1,0)**  
- logLik -187.78 · AIC 379.56 · BIC 385.49 · HQIC — · σ² 0.8093 · Ljung–Box(8) p —

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `const` | -0.0181 | 0.0755 | -0.24 | 0.811 |  |



### `us_consumer_confidence`  ·  group: us_consumer_confidence
**Transform:** `diff` → stationary series  ·  **Selected by CV (full-sample refit):** SARMA  

- Estimated order (stationary space): **(3,0,2)(0,0,1)[4]**  
- Full model order: **(3,1,2)(0,0,1)[4]**  
- logLik -87.00 · AIC 190.01 · BIC 213.71 · HQIC 199.64 · σ² 0.1928 · Ljung–Box(8) p 0.959

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `intercept` | 0.0002 | 0.0027 | 0.08 | 0.933 |  |
| `ar.L1` | 0.7870 | 0.2725 | 2.89 | 0.004 | *** |
| `ar.L2` | 0.2112 | 0.3993 | 0.53 | 0.597 |  |
| `ar.L3` | -0.3672 | 0.2678 | -1.37 | 0.170 |  |
| `ma.L1` | -0.1725 | 0.2842 | -0.61 | 0.544 |  |
| `ma.L2` | -0.8005 | 0.2334 | -3.43 | 0.001 | *** |
| `ma.S.L4` | 0.0754 | 0.1582 | 0.48 | 0.634 |  |
| `sigma2` | 0.1928 | 0.0222 | 8.67 | 0.000 | *** |



### `us_bond_yield_10y`  ·  group: us_bond_yield_10y
**Transform:** `diff` → stationary series  ·  **Selected by CV (full-sample refit):** Mean  

- Estimated order (stationary space): **(0,0,0)**  
- Full model order: **(0,1,0)**  
- logLik -89.76 · AIC 183.53 · BIC 189.45 · HQIC — · σ² 0.2055 · Ljung–Box(8) p —

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `const` | -0.0311 | 0.0380 | -0.82 | 0.413 |  |



### `us_credit`  ·  group: credit
**Transform:** `dlog+diff` → stationary series  ·  **Selected by CV (full-sample refit):** ARMA  

- Estimated order (stationary space): **(0,0,2)**  
- Full model order: **(0,2,2) on log-levels**  
- logLik 549.54 · AIC -1093.08 · BIC -1084.21 · HQIC -1089.48 · σ² 2.54e-05 · Ljung–Box(8) p 0.168

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `ma.L1` | -0.5250 | 0.0843 | -6.23 | 0.000 | *** |
| `ma.L2` | 0.0846 | 0.0717 | 1.18 | 0.238 |  |
| `sigma2` | 2.54e-05 | 2.12e-06 | 11.97 | 0.000 | *** |



### `us_sp500_close`  ·  group: sp500
**Transform:** `dlog` → stationary series  ·  **Selected by CV (full-sample refit):** Mean  

- Estimated order (stationary space): **(0,0,0)**  
- Full model order: **(0,1,0) on log-levels**  
- logLik 159.62 · AIC -315.25 · BIC -309.32 · HQIC — · σ² 0.0063 · Ljung–Box(8) p —

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `const` | 0.0210 | 0.0067 | 3.16 | 0.002 | *** |



### `us_vix`  ·  group: vix
**Transform:** `none` → stationary series  ·  **Selected by CV (full-sample refit):** AR  

- Estimated order (stationary space): **(4,0,0)**  
- Full model order: **(4,0,0)**  
- logLik -456.72 · AIC 925.45 · BIC 943.10 · HQIC 932.62 · σ² 39.9144 · Ljung–Box(8) p 0.866

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `const` | 6.8729 | 1.8831 | 3.65 | 0.000 | *** |
| `L1` | 0.4209 | 0.0845 | 4.98 | 0.000 | *** |
| `L2` | 0.1365 | 0.0909 | 1.50 | 0.133 |  |
| `L3` | 0.0574 | 0.0903 | 0.64 | 0.525 |  |
| `L4` | 0.0275 | 0.0836 | 0.33 | 0.743 |  |



### `us_house_price_idx`  ·  group: us_house_price_idx
**Transform:** `dlog` → stationary series  ·  **Selected by CV (full-sample refit):** AR  

- Estimated order (stationary space): **(1,0,0)**  
- Full model order: **(1,1,0) on log-levels**  
- logLik 511.10 · AIC -1016.20 · BIC -1007.34 · HQIC -1012.60 · σ² 4.38e-05 · Ljung–Box(8) p 0.050

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `const` | 0.0008 | 0.0006 | 1.39 | 0.165 |  |
| `L1` | 0.8448 | 0.0442 | 19.11 | 0.000 | *** |



### `us_industrial_production`  ·  group: us_industrial_production
**Transform:** `dlog` → stationary series  ·  **Selected by CV (full-sample refit):** SARMA  

- Estimated order (stationary space): **(1,0,0)**  
- Full model order: **(1,1,0) on log-levels**  
- logLik 406.08 · AIC -806.17 · BIC -797.28 · HQIC -802.56 · σ² 0.0002 · Ljung–Box(8) p 0.634

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `intercept` | 0.0022 | 0.0015 | 1.40 | 0.161 |  |
| `ar.L1` | 0.3568 | 0.0522 | 6.83 | 0.000 | *** |
| `sigma2` | 0.0002 | 1.3e-05 | 15.37 | 0.000 | *** |



### `us_oil_price`  ·  group: oil
**Transform:** `dlog` → stationary series  ·  **Selected by CV (full-sample refit):** Mean  

- Estimated order (stationary space): **(0,0,0)**  
- Full model order: **(0,1,0) on log-levels**  
- logLik 18.01 · AIC -32.02 · BIC -26.10 · HQIC — · σ² 0.0455 · Ljung–Box(8) p —

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `const` | 0.0072 | 0.0179 | 0.40 | 0.686 |  |



### `us_reer`  ·  group: reer
**Transform:** `dlog` → stationary series  ·  **Selected by CV (full-sample refit):** ARMA  

- Estimated order (stationary space): **(0,0,0)**  
- Full model order: **(0,1,0) on log-levels**  
- logLik 288.67 · AIC -575.33 · BIC -572.49 · HQIC -574.18 · σ² 0.0006 · Ljung–Box(8) p 0.904

**Coefficients (stationary space):**

| term | coef | std. err. | stat | p | |
|---|---:|---:|---:|---:|:--|
| `sigma2` | 0.0006 | 9.01e-05 | 6.89 | 0.000 | *** |



