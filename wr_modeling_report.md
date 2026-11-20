# NFL Big Data Bowl 2027 — WR Advanced Route Kinetics & LODCO Modeling Report

- **Thời gian tạo**: `2026-11-20 11:12:50`
- **Đề tài**: *Wide Receiver — From Track Speed to Route Kinetics*
- **Cỡ mẫu đánh giá**: Nhóm chuẩn `n_all >= 50` snaps (**N = 69 WR**) và Toàn bộ WR có dữ liệu NFL REG (**N = 83 WR**).
- **Ký hiệu ý nghĩa thống kê**: `**` (*p < 0.01*), `*` (*p < 0.05*), `.` (*p < 0.10*).

## 1. Chuẩn hóa theo nhóm bài tập thuần nhất & Đặc trưng vật lý ghép (`05_wr_advanced_kinetics.py`)

- Tổng hợp từ **107 WR** trên **1,105 reps** thuộc 10 bài `SKILL_DRILLS_WR` chạy ổn định cả 3 khóa Combine (`2023`–`2025`), sau khi chuẩn hóa Z-score trong nội bộ từng `drill_name`.

| Nhóm vật lý | drill_name | WRs | Med max_s | Med Decel | Med a_lat (yd/s²) | Med Braking Power | Med Retention |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hard-Break | `COMEBACK_ROUTE_RIGHT` | 98 | 8.14 | -6.67 | 7.35 | 54.54 | 0.351 |
| Hard-Break | `CURL_ROUTE_RIGHT` | 103 | 7.10 | -7.23 | 6.73 | 49.96 | 0.287 |
| Speed-Cut | `DAGGER_ROUTE_LEFT` | 107 | 8.32 | -2.03 | 6.98 | 16.77 | 0.853 |
| Speed-Cut | `SLANT_ROUTE_LEFT` | 107 | 8.27 | 2.05 | 4.16 | 0.00 | 1.000 |
| Speed-Cut | `SLOT_STRIKE_ROUTE_LEFT` | 101 | 8.25 | -7.10 | 7.76 | 34.65 | 0.381 |
| Speed-Cut | `SPEED_OUT_ROUTE_LEFT` | 107 | 7.00 | -1.40 | 7.08 | 9.58 | 0.907 |
| Vertical Speed | `GO_ROUTE_RIGHT` | 98 | 10.14 | -0.24 | 1.48 | 3.31 | 0.994 |
| Vertical Speed | `POST_CORNER_ROUTE_RIGHT` | 96 | 9.52 | -0.06 | 5.12 | 0.89 | 1.001 |
| Vertical Speed | `SLOT_SAIL_ROUTE_RIGHT` | 98 | 8.46 | -1.24 | 7.30 | 9.02 | 0.928 |
| Vertical Speed | `RED_ZONE_FADE_RIGHT` | 98 | 6.44 | 2.10 | 1.72 | 0.00 | 1.003 |

### 1.1. Tương quan đơn biến của bộ đặc trưng vật lý ghép trên `N = 69` WR

| Đặc trưng | soe_break ρ (p) | soe_break r (p) | soe_all ρ (p) | soe_all r (p) | soe_man ρ (p) | soe_man r (p) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `route_max_s` (yd/s) | +0.270* (0.025) | +0.287* (0.017) | +0.170 (0.163) | +0.162 (0.182) | +0.079 (0.518) | +0.092 (0.454) |
| `vertical_route_max_s_z` (Weighted Z) | +0.223. (0.066) | +0.266* (0.027) | +0.195 (0.109) | +0.150 (0.220) | +0.025 (0.838) | -0.014 (0.907) |
| `speed_cut_retention_z` (Z) | -0.097 (0.429) | -0.015 (0.903) | -0.090 (0.461) | -0.149 (0.222) | -0.051 (0.675) | +0.011 (0.926) |
| `peak_centripetal_accel` (yd/s²) | +0.034 (0.781) | +0.076 (0.535) | -0.049 (0.691) | -0.060 (0.623) | -0.212. (0.080) | -0.246* (0.042) |
| `peak_centripetal_accel_z` (Z) | +0.024 (0.843) | +0.061 (0.616) | -0.045 (0.711) | -0.087 (0.477) | -0.194 (0.111) | -0.213. (0.079) |
| `braking_power_index` (Hard-Break) | -0.065 (0.598) | -0.081 (0.509) | -0.051 (0.676) | -0.073 (0.550) | -0.086 (0.482) | -0.122 (0.319) |
| `braking_power_index_z` (Hard-Break Z) | -0.069 (0.575) | -0.076 (0.533) | -0.068 (0.578) | -0.085 (0.488) | -0.085 (0.487) | -0.121 (0.321) |
| `pure_kinetics_z` (Sensor-Only Composite) | +0.293* (0.015) | +0.317** (0.008) | +0.184 (0.131) | +0.168 (0.168) | +0.034 (0.780) | +0.049 (0.689) |
| `route_kinetics_index` (Weight-Controlled) | +0.358** (0.002) | +0.452** (0.000) | +0.319** (0.008) | +0.369** (0.002) | +0.159 (0.193) | +0.162 (0.183) |
| Baseline: `forty` (s) | -0.140 (0.270) | -0.081 (0.527) | -0.144 (0.256) | -0.132 (0.300) | -0.032 (0.802) | -0.044 (0.728) |
| Baseline: `ten_yd_split` (s) | -0.223. (0.076) | -0.215. (0.088) | -0.154 (0.223) | -0.235. (0.062) | -0.165 (0.193) | -0.211. (0.095) |
| Baseline: `ngs_athleticism_score` | -0.094 (0.442) | -0.085 (0.487) | -0.121 (0.322) | -0.084 (0.494) | -0.132 (0.280) | -0.115 (0.347) |
| Baseline: `combine_weight` (lbs) | -0.374** (0.002) | -0.455** (0.000) | -0.459** (0.000) | -0.509** (0.000) | -0.266* (0.027) | -0.272* (0.024) |

---

## 2. Cầu nối Combine-to-NFL Kinetic Translation (`06_wr_game_tracking_bridge.py`)

- Quét lazy streaming `game_tracking_2023..2025.csv` trên **35,427 route plays** hợp lệ (`REG`, `play_nullified_by_penalty == "N"`), trích xuất **927,243 active frames** (từ `ball_snap` đến `pass_forward`) cho **83 WR**.
- Bảng kiểm định Combine-to-NFL Kinetic Translation trên nhóm chuẩn **N = 69 WR** (`n_all >= 50`):

| Đặc trưng Combine (ρ / r, p-val Spearman) | `nfl_route_p95_s` | `nfl_route_max_s` | `nfl_break_centripetal_p95` | `nfl_break_decel_p05` |
| :--- | ---: | ---: | ---: | ---: |
| Combine 40-Yd Tracking `forty_track_max_s` | +0.567** / +0.518 (p=0.000) | +0.522** / +0.339 (p=0.000) | -0.196 / -0.233 (p=0.124) | -0.109 / -0.102 (p=0.396) |
| Combine Route `route_max_s` | +0.369** / +0.236 (p=0.002) | +0.313** / +0.056 (p=0.009) | -0.007 / +0.096 (p=0.953) | -0.155 / -0.059 (p=0.204) |
| Combine Vertical `vertical_route_max_s_z` | +0.455** / +0.435 (p=0.000) | +0.398** / +0.274 (p=0.001) | -0.007 / +0.060 (p=0.957) | -0.156 / -0.059 (p=0.200) |
| Combine Break `apex_speed_retention` | -0.066 / -0.083 (p=0.588) | -0.029 / -0.051 (p=0.812) | -0.030 / -0.015 (p=0.805) | -0.263* / -0.337 (p=0.029) |
| Combine Break `pre_break_decel` | -0.008 / -0.050 (p=0.948) | +0.060 / +0.033 (p=0.626) | -0.104 / -0.097 (p=0.397) | -0.201. / -0.248 (p=0.097) |
| Combine Break `peak_centripetal_accel` | -0.250* / -0.244 (p=0.039) | -0.012 / +0.063 (p=0.921) | +0.179 / +0.196 (p=0.141) | -0.167 / -0.182 (p=0.169) |
| Combine Hard-Break `braking_power_index` | +0.129 / +0.109 (p=0.289) | -0.036 / -0.047 (p=0.771) | +0.147 / +0.137 (p=0.228) | +0.077 / +0.101 (p=0.528) |
| Baseline `forty` (s) | -0.474** / -0.403 (p=0.000) | -0.419** / -0.263 (p=0.001) | +0.172 / +0.191 (p=0.174) | +0.158 / +0.163 (p=0.213) |
| Baseline `ten_yd_split` (s) | -0.190 / -0.160 (p=0.132) | -0.091 / +0.057 (p=0.473) | +0.143 / +0.198 (p=0.260) | +0.183 / +0.131 (p=0.148) |
| Baseline `ngs_athleticism_score` | +0.127 / +0.120 (p=0.299) | +0.176 / +0.001 (p=0.148) | -0.332** / -0.357 (p=0.005) | -0.078 / -0.145 (p=0.524) |

---

## 3. Kết quả mô hình hóa Leave-One-Draft-Class-Out CV (`2023`, `2024`, `2025`)

- Quy trình chống rò rỉ tuyệt đối: trong từng fold (train trên 2 khóa Draft, kiểm thử mù trên khóa thứ 3), toàn bộ bước `SimpleImputer(median)`, `StandardScaler`, xếp hạng biến và chọn `top_k` qua inner 3-fold CV đều chạy **chỉ trên tập train**.

### 3.1. So sánh Model A vs Model B vs Model C trên nhóm chuẩn (`N = 69` WR)

| Target | Estimator | Model | OOF $R^2$ | OOF RMSE | Spearman ρ (p) | Pearson r | Δρ vs A [95% Boot CI] | ΔRMSE vs A [95% Boot CI] |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `soe_break` | RidgeCV | Model A (Baseline) | +0.1003 | 0.1472 | +0.261* (0.030) | +0.324** | Ref | Ref |
| `soe_break` | RidgeCV | Model B (Route Kinetics) | +0.0246 | 0.1532 | +0.176 (0.148) | +0.180 | -0.085 [-0.366, +0.217] | +0.0061 [-0.0086, +0.0209] |
| `soe_break` | RidgeCV | Model C (Combined) | +0.1316 | 0.1446 | +0.293* (0.015) | +0.373** | +0.032 [-0.121, +0.188] | -0.0026 [-0.0110, +0.0058] |
| `soe_break` | BayesianRidge | Model A (Baseline) | +0.1100 | 0.1464 | +0.274* (0.023) | +0.338** | Ref | Ref |
| `soe_break` | BayesianRidge | Model B (Route Kinetics) | +0.0292 | 0.1529 | +0.213. (0.079) | +0.202. | -0.061 [-0.326, +0.226] | +0.0065 [-0.0080, +0.0203] |
| `soe_break` | BayesianRidge | Model C (Combined) | +0.1526 | 0.1428 | +0.324** (0.007) | +0.396** | +0.050 [-0.104, +0.210] | -0.0035 [-0.0127, +0.0053] |
| `soe_break` | ElasticNetCV | Model A (Baseline) | +0.1164 | 0.1458 | +0.248* (0.040) | +0.347** | Ref | Ref |
| `soe_break` | ElasticNetCV | Model B (Route Kinetics) | +0.0330 | 0.1526 | +0.201. (0.098) | +0.202. | -0.047 [-0.350, +0.277] | +0.0067 [-0.0112, +0.0240] |
| `soe_break` | ElasticNetCV | Model C (Combined) | +0.1334 | 0.1444 | +0.293* (0.015) | +0.377** | +0.045 [-0.119, +0.210] | -0.0014 [-0.0110, +0.0077] |
| `soe_all` | RidgeCV | Model A (Baseline) | +0.1795 | 0.1376 | +0.378** (0.001) | +0.428** | Ref | Ref |
| `soe_all` | RidgeCV | Model B (Route Kinetics) | -0.0435 | 0.1552 | -0.092 (0.451) | -0.099 | -0.471 [-0.765, -0.143] | +0.0176 [-0.0018, +0.0357] |
| `soe_all` | RidgeCV | Model C (Combined) | +0.1795 | 0.1376 | +0.378** (0.001) | +0.428** | +0.000 [+0.000, +0.000] | +0.0000 [+0.0000, +0.0000] |
| `soe_all` | BayesianRidge | Model A (Baseline) | +0.1790 | 0.1377 | +0.386** (0.001) | +0.429** | Ref | Ref |
| `soe_all` | BayesianRidge | Model B (Route Kinetics) | -0.0356 | 0.1546 | -0.055 (0.653) | -0.069 | -0.441 [-0.742, -0.109] | +0.0169 [-0.0033, +0.0356] |
| `soe_all` | BayesianRidge | Model C (Combined) | +0.1790 | 0.1377 | +0.386** (0.001) | +0.429** | +0.000 [+0.000, +0.000] | +0.0000 [+0.0000, +0.0000] |
| `soe_all` | ElasticNetCV | Model A (Baseline) | +0.1954 | 0.1363 | +0.394** (0.001) | +0.445** | Ref | Ref |
| `soe_all` | ElasticNetCV | Model B (Route Kinetics) | -0.0391 | 0.1549 | -0.109 (0.374) | -0.059 | -0.503 [-0.825, -0.164] | +0.0186 [-0.0021, +0.0381] |
| `soe_all` | ElasticNetCV | Model C (Combined) | +0.1954 | 0.1363 | +0.394** (0.001) | +0.445** | +0.000 [+0.000, +0.000] | +0.0000 [+0.0000, +0.0000] |
| `nfl_route_p95_s` | RidgeCV | Model A (Baseline) | +0.2354 | 0.3122 | +0.521** (0.000) | +0.486** | Ref | Ref |
| `nfl_route_p95_s` | RidgeCV | Model B (Route Kinetics) | +0.0504 | 0.3479 | +0.285* (0.018) | +0.263* | -0.236 [-0.495, +0.044] | +0.0357 [-0.0084, +0.0826] |
| `nfl_route_p95_s` | RidgeCV | Model C (Combined) | +0.1771 | 0.3239 | +0.456** (0.000) | +0.427** | -0.065 [-0.182, +0.050] | +0.0117 [-0.0098, +0.0351] |
| `nfl_route_p95_s` | BayesianRidge | Model A (Baseline) | +0.2346 | 0.3123 | +0.526** (0.000) | +0.486** | Ref | Ref |
| `nfl_route_p95_s` | BayesianRidge | Model B (Route Kinetics) | +0.0391 | 0.3500 | +0.265* (0.028) | +0.242* | -0.261 [-0.533, +0.016] | +0.0376 [-0.0045, +0.0828] |
| `nfl_route_p95_s` | BayesianRidge | Model C (Combined) | +0.1902 | 0.3213 | +0.467** (0.000) | +0.443** | -0.058 [-0.159, +0.033] | +0.0089 [-0.0081, +0.0291] |
| `nfl_route_p95_s` | ElasticNetCV | Model A (Baseline) | +0.2072 | 0.3179 | +0.503** (0.000) | +0.461** | Ref | Ref |
| `nfl_route_p95_s` | ElasticNetCV | Model B (Route Kinetics) | +0.0569 | 0.3467 | +0.377** (0.001) | +0.292* | -0.126 [-0.371, +0.137] | +0.0288 [-0.0082, +0.0652] |
| `nfl_route_p95_s` | ElasticNetCV | Model C (Combined) | +0.1274 | 0.3335 | +0.414** (0.000) | +0.399** | -0.089 [-0.220, +0.034] | +0.0156 [-0.0056, +0.0382] |

### 3.2. Kiểm tra trên toàn bộ mẫu (`N = 83` WR — EB-shrunk)

| Target | Estimator | Model | OOF $R^2$ | OOF RMSE | Spearman ρ (p) | Δρ (C−A) [95% Boot CI] |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| `soe_break` | RidgeCV | Model A (Baseline) | +0.0337 | 0.1431 | +0.160 (0.148) | — |
| `soe_break` | RidgeCV | Model B (Route Kinetics) | +0.0574 | 0.1413 | +0.237* (0.031) | — |
| `soe_break` | RidgeCV | Model C (Combined) | +0.0899 | 0.1388 | +0.229* (0.037) | +0.069 [-0.088, +0.224] |
| `soe_break` | BayesianRidge | Model A (Baseline) | +0.0608 | 0.1410 | +0.208. (0.059) | — |
| `soe_break` | BayesianRidge | Model B (Route Kinetics) | +0.0509 | 0.1418 | +0.252* (0.021) | — |
| `soe_break` | BayesianRidge | Model C (Combined) | +0.0895 | 0.1389 | +0.231* (0.036) | +0.023 [-0.127, +0.163] |
| `soe_break` | ElasticNetCV | Model A (Baseline) | +0.0197 | 0.1441 | +0.151 (0.172) | — |
| `soe_break` | ElasticNetCV | Model B (Route Kinetics) | +0.0503 | 0.1418 | +0.261* (0.017) | — |
| `soe_break` | ElasticNetCV | Model C (Combined) | +0.0762 | 0.1399 | +0.234* (0.033) | +0.083 [-0.078, +0.238] |
| `soe_all` | RidgeCV | Model A (Baseline) | +0.1297 | 0.1335 | +0.344** (0.001) | — |
| `soe_all` | RidgeCV | Model B (Route Kinetics) | -0.0136 | 0.1441 | -0.034 (0.763) | — |
| `soe_all` | RidgeCV | Model C (Combined) | +0.1096 | 0.1351 | +0.312** (0.004) | -0.032 [-0.079, +0.013] |
| `soe_all` | BayesianRidge | Model A (Baseline) | +0.1252 | 0.1339 | +0.348** (0.001) | — |
| `soe_all` | BayesianRidge | Model B (Route Kinetics) | -0.0053 | 0.1435 | +0.018 (0.874) | — |
| `soe_all` | BayesianRidge | Model C (Combined) | +0.1073 | 0.1353 | +0.317** (0.004) | -0.031 [-0.078, +0.012] |
| `soe_all` | ElasticNetCV | Model A (Baseline) | +0.1258 | 0.1338 | +0.350** (0.001) | — |
| `soe_all` | ElasticNetCV | Model B (Route Kinetics) | -0.0162 | 0.1443 | -0.033 (0.770) | — |
| `soe_all` | ElasticNetCV | Model C (Combined) | +0.1258 | 0.1338 | +0.350** (0.001) | +0.000 [+0.000, +0.000] |
| `nfl_route_p95_s` | RidgeCV | Model A (Baseline) | +0.2441 | 0.3253 | +0.508** (0.000) | — |
| `nfl_route_p95_s` | RidgeCV | Model B (Route Kinetics) | +0.0221 | 0.3700 | +0.209. (0.058) | — |
| `nfl_route_p95_s` | RidgeCV | Model C (Combined) | +0.1945 | 0.3358 | +0.477** (0.000) | -0.031 [-0.143, +0.065] |
| `nfl_route_p95_s` | BayesianRidge | Model A (Baseline) | +0.2419 | 0.3258 | +0.507** (0.000) | — |
| `nfl_route_p95_s` | BayesianRidge | Model B (Route Kinetics) | +0.0475 | 0.3651 | +0.231* (0.035) | — |
| `nfl_route_p95_s` | BayesianRidge | Model C (Combined) | +0.1919 | 0.3363 | +0.473** (0.000) | -0.033 [-0.141, +0.059] |
| `nfl_route_p95_s` | ElasticNetCV | Model A (Baseline) | +0.2228 | 0.3298 | +0.486** (0.000) | — |
| `nfl_route_p95_s` | ElasticNetCV | Model B (Route Kinetics) | +0.0164 | 0.3711 | +0.213. (0.053) | — |
| `nfl_route_p95_s` | ElasticNetCV | Model C (Combined) | +0.2084 | 0.3329 | +0.454** (0.000) | -0.031 [-0.154, +0.087] |

### 3.3. Chi tiết theo từng khóa Draft & Vai trò từng biến (`soe_break`, `ElasticNetCV`, `N = 69`)

| Model | Holdout Year | Train / Val | Top-K | OOF $R^2$ | RMSE | Spearman ρ | Selected Features |
| :--- | :---: | :---: | ---: | ---: | ---: | ---: | :--- |
| Model A (Baseline) | 2023 | 40 / 29 | 4 | +0.1362 | 0.1291 | +0.399 | `combine_weight`, `combine_height`, `ngs_athleticism_score`, `forty` |
| Model A (Baseline) | 2024 | 46 / 23 | 1 | +0.0231 | 0.1357 | +0.073 | `combine_weight` |
| Model A (Baseline) | 2025 | 52 / 17 | 3 | +0.1206 | 0.1816 | +0.306 | `combine_weight`, `ten_yd_split`, `combine_height` |
| Model B (Route Kinetics) | 2023 | 40 / 29 | 2 | -0.0611 | 0.1431 | +0.294 | `pure_kinetics_z`, `route_max_s` |
| Model B (Route Kinetics) | 2024 | 46 / 23 | 1 | +0.0002 | 0.1373 | -0.038 | `vertical_route_max_s_z` |
| Model B (Route Kinetics) | 2025 | 52 / 17 | 1 | +0.0922 | 0.1845 | +0.471 | `pure_kinetics_z` |
| Model C (Combined) | 2023 | 40 / 29 | 1 | +0.0697 | 0.1340 | +0.433 | `route_kinetics_index` |
| Model C (Combined) | 2024 | 46 / 23 | 2 | +0.0635 | 0.1329 | +0.078 | `combine_weight`, `route_kinetics_index` |
| Model C (Combined) | 2025 | 52 / 17 | 3 | +0.1959 | 0.1737 | +0.473 | `route_kinetics_index`, `combine_weight`, `ten_yd_split` |

#### Trọng số chuẩn hóa trung bình qua 3 fold (`Model C (Combined)`, `soe_break`, `ElasticNetCV`)

| Feature | Selected Folds (/3) | Mean Std Coef | Mean |Std Coef| |
| :--- | :---: | ---: | ---: |
| `route_kinetics_index` | 3/3 | +0.0427 | 0.0427 |
| `combine_weight` | 2/3 | -0.0293 | 0.0293 |
| `ten_yd_split` | 1/3 | -0.0115 | 0.0115 |
| `forty` | 0/3 | +0.0000 | 0.0000 |
| `ngs_athleticism_score` | 0/3 | +0.0000 | 0.0000 |
| `combine_height` | 0/3 | +0.0000 | 0.0000 |
| `route_max_s` | 0/3 | +0.0000 | 0.0000 |
| `vertical_route_max_s_z` | 0/3 | +0.0000 | 0.0000 |
| `pure_kinetics_z` | 0/3 | +0.0000 | 0.0000 |
| `peak_centripetal_accel_z` | 0/3 | +0.0000 | 0.0000 |
| `braking_power_index_z` | 0/3 | +0.0000 | 0.0000 |
| `speed_cut_retention_z` | 0/3 | +0.0000 | 0.0000 |
| `pre_break_jerk` | 0/3 | +0.0000 | 0.0000 |
| `functional_speed_ratio` | 0/3 | +0.0000 | 0.0000 |

---

## 4. Ứng dụng tuyển trạch: Top 10 Hidden Gems vs Top 5 Overvalued Track Stars

### 4.1. Top 10 Hidden Gems (Vòng muộn >= Rd 3 / UDFA nhưng Route Kinetics & SOE cao)

| Cầu thủ (`display_name`) | Draft Year | Round | Overall Pick | 40-Yd | Wt (lbs) | Combine `route_max_s` | `route_kinetics_index` | NFL `route_p95_s` | `soe_break` | `soe_all` |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tez Johnson | 2025 | Rd 7 | #235 | 4.51s | 154 | 10.84 | +2.84 | 8.51 | +0.443 | +0.258 |
| Jalin Hyatt | 2023 | Rd 3 | #73 | 4.40s | 176 | 10.35 | +1.04 | 8.69 | +0.305 | +0.135 |
| Trey Palmer | 2023 | Rd 6 | #191 | 4.33s | 192 | 10.98 | +1.66 | 8.55 | +0.211 | +0.088 |
| Derius Davis | 2023 | Rd 4 | #125 | 4.36s | 165 | 10.35 | +1.33 | 8.53 | +0.197 | +0.551 |
| Jimmy Horn | 2025 | Rd 6 | #208 | 4.46s | 174 | 10.69 | +1.92 | 9.14 | +0.127 | +0.028 |
| Malik Washington | 2024 | Rd 6 | #184 | 4.47s | 191 | 10.09 | +0.30 | 7.84 | +0.256 | +0.213 |
| Tre Tucker | 2023 | Rd 3 | #100 | 4.40s | 182 | 10.63 | +1.48 | 8.65 | +0.103 | +0.123 |
| Arian Smith | 2025 | Rd 4 | #110 | 4.36s | 179 | 10.31 | +1.20 | 8.52 | +0.128 | +0.265 |
| Troy Franklin | 2024 | Rd 4 | #102 | 4.41s | 176 | 10.34 | +1.19 | 8.48 | +0.105 | +0.083 |
| Isaiah Bond | 2025 | UDFA | UDFA | 4.39s | 180 | 10.46 | +1.22 | 8.70 | +0.096 | +0.147 |

### 4.2. Top 5 Overvalued Track Stars (40-yard <= 4.48s hoặc Draft <= Rd 2 nhưng Route Kinetics & SOE thấp)

| Cầu thủ (`display_name`) | Draft Year | Round | Overall Pick | 40-Yd | Wt (lbs) | Combine `route_max_s` | `route_kinetics_index` | NFL `route_p95_s` | `soe_break` | `soe_all` |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dont'e Thornton | 2025 | Rd 4 | #108 | 4.30s | 205 | 8.51 | -2.02 | 8.13 | -0.265 | -0.257 |
| Jack Bech | 2025 | Rd 2 | #58 | — | 214 | 9.87 | -0.77 | 7.50 | -0.302 | -0.155 |
| KeAndre Lambert-Smith | 2025 | Rd 5 | #158 | 4.37s | 190 | 9.08 | -0.51 | 8.48 | -0.195 | +0.084 |
| Jonathan Mingo | 2023 | Rd 2 | #39 | 4.46s | 220 | 10.50 | -0.17 | 7.97 | -0.210 | -0.226 |
| Chimere Dike | 2025 | Rd 4 | #103 | 4.34s | 196 | 10.13 | +0.22 | 8.19 | -0.256 | -0.112 |

