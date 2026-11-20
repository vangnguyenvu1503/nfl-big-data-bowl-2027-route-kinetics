# NFL Big Data Bowl 2027 — WR Robustness Audit (Weight Confounding & Drill Opt-Out)

- **Thời gian tạo**: `2026-11-20 11:31:49`
- **Mục tiêu kiểm toán**: Giải trình 2 điểm phản biện trọng yếu của ban giám khảo NFL — (1) nhiễu cân nặng `combine_weight` và (2) sai lệch do cầu thủ bỏ bài tập (`drill opt-out`), đồng thời chạy lại LODCO CV minh bạch 100% không pha trộn cân nặng.
- **Ký hiệu**: `**` (*p < 0.01*), `*` (*p < 0.05*), `.` (*p < 0.10*).

## 1. Kiểm toán công thức & Nested OLS (Chứng minh giá trị gia tăng của Sensor)

### 1.1. Công thức chính xác trong `src/05_wr_advanced_kinetics.py`

```python
# 100% Sensor-Only Composite (HOÀN TOÀN KHÔNG chứa combine_weight):
pure_kinetics_z = (
    0.55 * route_max_s_z
    + 0.30 * vertical_route_max_s_z
    + 0.15 * peak_centripetal_accel_z
)

# Weight-Controlled Composite (có trừ weight_z nhằm bù nhiễu thể hình):
route_kinetics_index = (
    0.50 * route_max_s_z
    + 0.25 * vertical_route_max_s_z
    + 0.15 * peak_centripetal_accel_z
    + 0.10 * braking_power_index_z
    - 0.50 * weight_z
)
```

- **Kết luận kiểm toán công thức**: Trong `wr_modeling_report.md`, biến `route_kinetics_index` có chứa `-0.50 * weight_z` nên tương quan cao một phần đến từ `combine_weight` ($\rho = -0.374^{**}$ với `soe_break`). Ngược lại, `pure_kinetics_z` hoàn toàn sạch (`0%` `combine_weight`). Để trả lời dứt khoát liệu dữ liệu sensor Combine có mang lại **thông tin mới vượt trên cân nặng và 40-yard dash** hay không, ta chạy **Nested OLS F-test & Likelihood Ratio Test (LRT)** dưới đây.

### 1.2. Nested OLS F-Test & Likelihood Ratio Test dự báo `soe_break`

- **Model 0 (Size + Track Baseline)**: `combine_weight + combine_height + forty + ten_yd_split + ngs_athleticism_score`.
- **Model 1 (+ Sensor)**: `Model 0 + sensor_feature`.

| Cohort | Added Sensor | R² (M0 → M1) | Adj R² (M0 → M1, Δ) | β Sensor (p) | Partial r | F-test (p) | LRT χ² (p) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| N=69 (`n_all>=50`, median-imputed 40) | `pure_kinetics_z` | 0.2448 → 0.2797 | 0.1849 → 0.2100 (+0.0251) | +0.0393 (p=0.0882.) | +0.215 | 3.000 (p=0.0882.) | 3.261 (p=0.0710.) |
| N=69 (`n_all>=50`, median-imputed 40) | `route_max_s` | 0.2448 → 0.2765 | 0.1849 → 0.2064 (+0.0215) | +0.0546 (p=0.1048) | +0.205 | 2.709 (p=0.1048) | 2.951 (p=0.0858.) |
| N=69 (`n_all>=50`, median-imputed 40) | `vertical_route_max_s_z` | 0.2448 → 0.2587 | 0.1849 → 0.1870 (+0.0021) | +0.0257 (p=0.2858) | +0.135 | 1.159 (p=0.2858) | 1.278 (p=0.2582) |
| N=69 (`n_all>=50`, median-imputed 40) | `route_speed_z` | 0.2448 → 0.2589 | 0.1849 → 0.1872 (+0.0023) | -0.0498 (p=0.2816) | -0.137 | 1.180 (p=0.2816) | 1.301 (p=0.2541) |
| N=64 (`n_all>=50`, complete 40) | `pure_kinetics_z` | 0.2359 → 0.2995 | 0.1701 → 0.2257 (+0.0557) | +0.0549 (p=0.0267*) | +0.288 | 5.171 (p=0.0267*) | 5.558 (p=0.0184*) |
| N=64 (`n_all>=50`, complete 40) | `route_max_s` | 0.2359 → 0.2909 | 0.1701 → 0.2162 (+0.0462) | +0.0738 (p=0.0400*) | +0.268 | 4.416 (p=0.0400*) | 4.775 (p=0.0289*) |
| N=64 (`n_all>=50`, complete 40) | `vertical_route_max_s_z` | 0.2359 → 0.2644 | 0.1701 → 0.1870 (+0.0169) | +0.0396 (p=0.1427) | +0.193 | 2.209 (p=0.1427) | 2.434 (p=0.1188) |
| N=64 (`n_all>=50`, complete 40) | `route_speed_z` | 0.2359 → 0.2511 | 0.1701 → 0.1722 (+0.0021) | -0.0536 (p=0.2880) | -0.141 | 1.150 (p=0.2880) | 1.279 (p=0.2581) |
| N=83 (`All REG WRs`, median-imputed 40) | `pure_kinetics_z` | 0.1409 → 0.1830 | 0.0851 → 0.1184 (+0.0333) | +0.0420 (p=0.0516.) | +0.221 | 3.909 (p=0.0516.) | 4.163 (p=0.0413*) |
| N=83 (`All REG WRs`, median-imputed 40) | `route_max_s` | 0.1409 → 0.1797 | 0.0851 → 0.1150 (+0.0299) | +0.0600 (p=0.0617.) | +0.213 | 3.597 (p=0.0617.) | 3.839 (p=0.0501.) |
| N=83 (`All REG WRs`, median-imputed 40) | `vertical_route_max_s_z` | 0.1409 → 0.1623 | 0.0851 → 0.0962 (+0.0111) | +0.0297 (p=0.1673) | +0.158 | 1.944 (p=0.1673) | 2.096 (p=0.1477) |
| N=83 (`All REG WRs`, median-imputed 40) | `route_speed_z` | 0.1409 → 0.1526 | 0.0851 → 0.0857 (+0.0006) | -0.0431 (p=0.3095) | -0.117 | 1.047 (p=0.3095) | 1.135 (p=0.2866) |
| N=75 (`All REG WRs`, complete 40) | `pure_kinetics_z` | 0.1575 → 0.2364 | 0.0965 → 0.1690 (+0.0725) | +0.0593 (p=0.0100**) | +0.306 | 7.023 (p=0.0100**) | 7.372 (p=0.0066**) |
| N=75 (`All REG WRs`, complete 40) | `route_max_s` | 0.1575 → 0.2279 | 0.0965 → 0.1598 (+0.0633) | +0.0825 (p=0.0152*) | +0.289 | 6.198 (p=0.0152*) | 6.542 (p=0.0105*) |
| N=75 (`All REG WRs`, complete 40) | `vertical_route_max_s_z` | 0.1575 → 0.1960 | 0.0965 → 0.1251 (+0.0286) | +0.0422 (p=0.0755.) | +0.214 | 3.257 (p=0.0755.) | 3.509 (p=0.0610.) |
| N=75 (`All REG WRs`, complete 40) | `route_speed_z` | 0.1575 → 0.1692 | 0.0965 → 0.0959 (-0.0006) | -0.0440 (p=0.3315) | -0.118 | 0.957 (p=0.3315) | 1.048 (p=0.3060) |
| N=61 (`n_all>=50` & `has_go_route`) | `pure_kinetics_z` | 0.2733 → 0.3002 | 0.2072 → 0.2225 (+0.0153) | +0.0467 (p=0.1550) | +0.193 | 2.080 (p=0.1550) | 2.306 (p=0.1289) |
| N=61 (`n_all>=50` & `has_go_route`) | `route_max_s` | 0.2733 → 0.3012 | 0.2072 → 0.2236 (+0.0164) | +0.0947 (p=0.1474) | +0.196 | 2.161 (p=0.1474) | 2.394 (p=0.1218) |
| N=61 (`n_all>=50` & `has_go_route`) | `vertical_route_max_s_z` | 0.2733 → 0.2905 | 0.2072 → 0.2117 (+0.0045) | +0.0275 (p=0.2571) | +0.154 | 1.312 (p=0.2571) | 1.464 (p=0.2262) |
| N=61 (`n_all>=50` & `has_go_route`) | `route_speed_z` | 0.2733 → 0.2829 | 0.2072 → 0.2032 (-0.0040) | -0.0438 (p=0.3982) | -0.115 | 0.725 (p=0.3982) | 0.814 (p=0.3670) |

### 1.3. Chi tiết từng hệ số trong Model 1 (`Model 0 + pure_kinetics_z`)

#### Complete 40-yd Cohort (`N = 64`, `n_all >= 50`) — $R^2 = 0.2995$, $R^2_{\text{adj}} = 0.2257$

| Biến | Coef (β) | Std Err | t-stat | p-val | 95% CI |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Intercept` | +1.01893 | 2.15729 | +0.472 | 0.6385 | [-3.30096, +5.33882] |
| `combine_weight` | -0.00447 | 0.00203 | -2.199 | 0.0319* | [-0.00853, -0.00040] |
| `combine_height` | +0.00936 | 0.01243 | +0.754 | 0.4542 | [-0.01552, +0.03425] |
| `forty` | +0.14688 | 0.35509 | +0.414 | 0.6807 | [-0.56417, +0.85794] |
| `ten_yd_split` | -0.91580 | 0.76258 | -1.201 | 0.2348 | [-2.44284, +0.61125] |
| `ngs_athleticism_score` | -0.00099 | 0.00372 | -0.265 | 0.7918 | [-0.00844, +0.00646] |
| `pure_kinetics_z` | +0.05494 | 0.02416 | +2.274 | 0.0267* | [+0.00656, +0.10332] |

#### All REG WRs (`N = 83`, median-imputed 40-yd) — $R^2 = 0.1830$, $R^2_{\text{adj}} = 0.1184$

| Biến | Coef (β) | Std Err | t-stat | p-val | 95% CI |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Intercept` | +0.69093 | 1.50710 | +0.458 | 0.6479 | [-2.31071, +3.69257] |
| `combine_weight` | -0.00286 | 0.00160 | -1.791 | 0.0772. | [-0.00604, +0.00032] |
| `combine_height` | +0.00120 | 0.01026 | +0.117 | 0.9070 | [-0.01923, +0.02163] |
| `forty` | +0.10840 | 0.25545 | +0.424 | 0.6725 | [-0.40038, +0.61718] |
| `ten_yd_split` | -0.44813 | 0.65168 | -0.688 | 0.4938 | [-1.74607, +0.84980] |
| `ngs_athleticism_score` | -0.00024 | 0.00209 | -0.116 | 0.9081 | [-0.00440, +0.00392] |
| `pure_kinetics_z` | +0.04199 | 0.02123 | +1.977 | 0.0516. | [-0.00031, +0.08428] |

### 1.4. Kiểm định giá trị thực chiến trên sân NFL (`player_play.csv` & `player_career_successes.csv`, `N = 69`)

| Feature (ρ / r, p-val S) | Expected YAC | Actual YAC | YPRR | Target Rate | Career Snaps (Draft Z) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `pure_kinetics_z` (100% Sensor) | +0.301* / +0.267 (p=0.012) | +0.157 / +0.086 (p=0.198) | +0.165 / +0.171 (p=0.176) | +0.148 / +0.125 (p=0.224) | +0.092 / +0.077 (p=0.453) |
| `size_adjusted_route_kinetics` (Residual) | +0.201. / +0.193 (p=0.097) | +0.086 / +0.045 (p=0.482) | +0.179 / +0.178 (p=0.141) | +0.153 / +0.138 (p=0.209) | +0.161 / +0.113 (p=0.186) |
| `route_max_s` | +0.297* / +0.245 (p=0.013) | +0.134 / +0.047 (p=0.272) | +0.175 / +0.155 (p=0.151) | +0.139 / +0.100 (p=0.254) | +0.049 / +0.012 (p=0.689) |
| `vertical_route_max_s_z` | +0.320** / +0.273 (p=0.007) | +0.192 / +0.179 (p=0.114) | +0.097 / +0.149 (p=0.429) | +0.098 / +0.114 (p=0.422) | +0.160 / +0.203 (p=0.190) |
| `route_speed_z` (Opt-out-proof) | +0.247* / +0.230 (p=0.041) | +0.224. / +0.134 (p=0.065) | +0.196 / +0.110 (p=0.106) | +0.235. / +0.144 (p=0.052) | +0.089 / +0.083 (p=0.465) |
| `nfl_route_p95_s` (On-field) | +0.326** / +0.322 (p=0.006) | +0.046 / +0.077 (p=0.708) | -0.071 / -0.109 (p=0.561) | -0.146 / -0.175 (p=0.232) | -0.183 / -0.201 (p=0.132) |
| Baseline: `forty` (s) | -0.291* / -0.296 (p=0.020) | -0.164 / -0.185 (p=0.196) | -0.100 / -0.143 (p=0.432) | -0.092 / -0.109 (p=0.469) | -0.115 / -0.156 (p=0.366) |
| Baseline: `ngs_athleticism_score` | +0.013 / +0.032 (p=0.914) | -0.054 / -0.004 (p=0.658) | -0.015 / -0.066 (p=0.904) | +0.027 / -0.044 (p=0.825) | +0.209. / +0.213 (p=0.085) |

---

## 2. Kiểm toán Drill Opt-Out & Chỉ số sạch chống bỏ bài (`route_speed_z` & `size_adjusted_route_kinetics`)

- **Phát hiện quan trọng về Drill Opt-Out**:
  - Trong `combine_tracking.csv`, các bài route dài như `GO_ROUTE_RIGHT` (trung vị `10.14 yd/s`) và `POST_CORNER_ROUTE_RIGHT` (`9.52 yd/s`) có tốc độ đỉnh cao hơn hẳn các bài ngắn như `CURL_ROUTE_RIGHT` (`7.10 yd/s`) hay `SPEED_OUT_ROUTE_LEFT` (`7.00 yd/s`).
  - Nếu dùng `route_max_s = max(s)` trực tiếp, cầu thủ **bỏ bài `GO_ROUTE_RIGHT` và `POST_CORNER_ROUTE_RIGHT`** sẽ bị gán `route_max_s` rất thấp một cách giả tạo dù họ chạy nhanh ở các bài ngắn hơn.
  - Kiểm tra xác nhận: **Dont'e Thornton** (`4.30s` 40-yd) chỉ chạy **5/12** bài route (bỏ cả `GO_ROUTE_RIGHT` lẫn `POST_CORNER_ROUTE_RIGHT`), nên `route_max_s = 8.51` chủ yếu phản ánh opt-out; trên 5 bài đã chạy, `route_speed_z = +0.10` (mức trung bình). Tương tự **KeAndre Lambert-Smith** (`4.37s`) chỉ chạy **8/12** bài (bỏ cả `GO` và `POST_CORNER`), nên dù `route_max_s = 9.08`, trên 8 bài đã chạy anh đạt `route_speed_z = +1.02`!
  - Để chống opt-out 100%, ta bổ sung **`route_speed_z`** (trung bình Z-score của `drill_max_s` chỉ trên các bài cầu thủ **thực sự chạy**) và **`size_adjusted_route_kinetics`** (phần dư OLS của `pure_kinetics_z ~ combine_weight`, trực giao hoàn toàn với cân nặng: $r = 0.0000$).

### 2.1. Kiểm toán đầy đủ 15 cầu thủ Case Study (10 Hidden Gems + 5 Overvalued Track Stars cũ)

| Cầu thủ | Draft | 40-Yd | Wt | Drills | Ran GO? | Ran PC? | `route_max_s` | `route_speed_z` | `pure_kinetics_z` | `size_adj_kin` | `soe_break` | Bài cốt lõi (10 bài) bị bỏ |
| :--- | :---: | ---: | ---: | :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Tez Johnson | 2025 Rd 7 | 4.51s | 154 | 11/12 | Yes | **NO** | 10.84 | +0.07 | +1.55 | +1.10 | +0.443 | POST_CORNER_ROUTE_RIGHT |
| Jalin Hyatt | 2023 Rd 3 | 4.40s | 176 | 9/12 | Yes | **NO** | 10.35 | -0.32 | +0.41 | +0.19 | +0.305 | COMEBACK_ROUTE_RIGHT, POST_CORNER_ROUTE_RIGHT, RED_ZONE_FADE_RIGHT |
| Malik Washington | 2024 Rd 6 | 4.47s | 191 | 11/12 | Yes | Yes | 10.09 | -0.46 | +0.13 | +0.07 | +0.256 | Đủ 10/10 bài cốt lõi |
| Trey Palmer | 2023 Rd 6 | 4.33s | 192 | 12/12 | Yes | Yes | 10.98 | +0.03 | +1.69 | +1.63 | +0.211 | Đủ 10/10 bài cốt lõi |
| Derius Davis | 2023 Rd 4 | 4.36s | 165 | 12/12 | Yes | Yes | 10.35 | -0.02 | +0.39 | +0.06 | +0.197 | Đủ 10/10 bài cốt lõi |
| Arian Smith | 2025 Rd 4 | 4.36s | 179 | 12/12 | Yes | Yes | 10.31 | +0.95 | +0.47 | +0.28 | +0.128 | Đủ 10/10 bài cốt lõi |
| Jimmy Horn | 2025 Rd 6 | 4.46s | 174 | 12/12 | Yes | Yes | 10.69 | +0.46 | +1.27 | +1.03 | +0.127 | Đủ 10/10 bài cốt lõi |
| Troy Franklin | 2024 Rd 4 | 4.41s | 176 | 12/12 | Yes | Yes | 10.34 | +0.13 | +0.46 | +0.24 | +0.105 | Đủ 10/10 bài cốt lõi |
| Tre Tucker | 2023 Rd 3 | 4.40s | 182 | 12/12 | Yes | Yes | 10.63 | -0.13 | +1.09 | +0.93 | +0.103 | Đủ 10/10 bài cốt lõi |
| Isaiah Bond | 2025 UDFA | 4.39s | 180 | 12/12 | Yes | Yes | 10.46 | +0.03 | +0.83 | +0.66 | +0.096 | Đủ 10/10 bài cốt lõi |
| KeAndre Lambert-Smith | 2025 Rd 5 | 4.37s | 190 | 8/12 | **NO (Opt-out)** | **NO** | 9.08 | +1.02 | -0.92 | -0.99 | -0.195 | COMEBACK_ROUTE_RIGHT, GO_ROUTE_RIGHT, POST_CORNER_ROUTE_RIGHT, RED_ZONE_FADE_RIGHT |
| Jonathan Mingo | 2023 Rd 2 | 4.46s | 220 | 12/12 | Yes | Yes | 10.50 | +0.10 | +0.69 | +0.93 | -0.210 | Đủ 10/10 bài cốt lõi |
| Chimere Dike | 2025 Rd 4 | 4.34s | 196 | 12/12 | Yes | Yes | 10.13 | +0.20 | +0.12 | +0.10 | -0.256 | Đủ 10/10 bài cốt lõi |
| Dont'e Thornton | 2025 Rd 4 | 4.30s | 205 | 5/12 | **NO (Opt-out)** | **NO** | 8.51 | +0.10 | -1.94 | -1.85 | -0.265 | COMEBACK_ROUTE_RIGHT, CURL_ROUTE_RIGHT, GO_ROUTE_RIGHT, POST_CORNER_ROUTE_RIGHT, RED_ZONE_FADE_RIGHT, SLOT_SAIL_ROUTE_RIGHT, SLOT_STRIKE_ROUTE_LEFT |
| Jack Bech | 2025 Rd 2 | — | 214 | 12/12 | Yes | Yes | 9.87 | -0.10 | -0.32 | -0.14 | -0.302 | Đủ 10/10 bài cốt lõi |

### 2.2. Danh sách Overvalued Track Stars ĐÃ LOẠI BỎ Opt-Out (`has_go_route == True` & `forty <= 4.48s` hoặc `Rd <= 2`)

| Cầu thủ | Draft | 40-Yd | Wt | Drills | `route_max_s` | `route_speed_z` | `pure_kinetics_z` | `size_adj_kin` | NFL `p95_s` | `soe_break` |
| :--- | :---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jack Bech | 2025 Rd 2 | — | 214 | 12/12 (Yes GO) | 9.87 | -0.10 | -0.32 | -0.14 | 7.50 | -0.302 |
| Jamari Thrash | 2024 Rd 5 | 4.46s | 188 | 12/12 (Yes GO) | 9.87 | +0.17 | -0.38 | -0.48 | 7.64 | -0.139 |
| Ladd McConkey | 2024 Rd 2 | 4.39s | 186 | 12/12 (Yes GO) | 9.85 | -0.18 | -0.53 | -0.64 | 7.97 | -0.089 |
| Chimere Dike | 2025 Rd 4 | 4.34s | 196 | 12/12 (Yes GO) | 10.13 | +0.20 | +0.12 | +0.10 | 8.19 | -0.256 |
| A.T. Perry | 2023 Rd 6 | 4.47s | 198 | 12/12 (Yes GO) | 10.14 | -0.36 | -0.07 | -0.07 | 8.30 | -0.100 |
| Jayden Reed | 2023 Rd 2 | 4.45s | 187 | 12/12 (Yes GO) | 10.33 | +0.27 | +0.32 | +0.21 | 8.44 | -0.157 |

---

## 3. Hai cặp đối chứng "Apples-to-Apples" hoàn hảo cho Hero Visual

- Điều kiện lọc tự động: `|Δ weight| <= 10 lbs`, `|Δ forty| <= 0.08s`, chạy chung **>= 4 bài drill** (gồm cả Vertical & Break và **cả hai đều chạy `GO_ROUTE_RIGHT`**), không dính nhiễu opt-out.

| Cặp đối chứng (Gem vs Track Clone) | Cân nặng | 40-Yd | Drill chung | `pure_kinetics_z` | `size_adj_kin` | `soe_break` |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: |
| Trey Palmer vs Ladd McConkey | 192 vs 186 lbs (Δ=6) | 4.33s vs 4.39s (Δ=0.06s) | 12 bài (cả 2 có GO) | +1.69 vs -0.53 | +1.63 vs -0.64 | +0.211 vs -0.089 |
| Trey Palmer vs Chimere Dike | 192 vs 196 lbs (Δ=4) | 4.33s vs 4.34s (Δ=0.01s) | 11 bài (cả 2 có GO) | +1.69 vs +0.12 | +1.63 vs +0.10 | +0.211 vs -0.256 |
| Jason Brownlee vs Ronnie Bell | 198 vs 191 lbs (Δ=7) | 4.59s vs 4.54s (Δ=0.05s) | 12 bài (cả 2 có GO) | +0.67 vs -0.74 | +0.67 vs -0.80 | +0.156 vs -0.168 |
| Malik Heath vs Jalen Coker | 213 vs 208 lbs (Δ=5) | 4.64s vs 4.57s (Δ=0.07s) | 12 bài (cả 2 có GO) | +0.58 vs -0.54 | +0.75 vs -0.43 | +0.208 vs -0.077 |
| Arian Smith vs Ladd McConkey | 179 vs 186 lbs (Δ=7) | 4.36s vs 4.39s (Δ=0.03s) | 11 bài (cả 2 có GO) | +0.47 vs -0.53 | +0.28 vs -0.64 | +0.128 vs -0.089 |
| Tre Tucker vs Ladd McConkey | 182 vs 186 lbs (Δ=4) | 4.40s vs 4.39s (Δ=0.01s) | 12 bài (cả 2 có GO) | +1.09 vs -0.53 | +0.93 vs -0.64 | +0.103 vs -0.089 |

### 3.1. Hero Pair #1: **Trey Palmer** (2023, Rd 6 (#191)) vs **Ladd McConkey** (2024, Rd 2 (#34))

- **Thể hình & Track**: Trey Palmer (**192 lbs**, `4.33s` 40-yd) vs Ladd McConkey (**186 lbs**, `4.39s` 40-yd) — chạy chung **12 bài drill**.
- **Combine Route Kinetics**: `route_max_s` = **10.98** vs **9.85 yd/s** | `pure_kinetics_z` = **+1.69** vs **-0.53** | `size_adjusted_route_kinetics` = **+1.63** vs **-0.64**.
- **Thực chiến NFL**: `nfl_route_p95_s` = **8.55** vs **7.97 yd/s** | `soe_break` = **+0.211** vs **-0.089 yd** | `soe_all` = **+0.088** vs **-0.050 yd**.

| Bài drill chạy chung | Trey Palmer max_s | Ladd McConkey max_s | Chênh lệch (Gem − Clone) |
| :--- | ---: | ---: | ---: |
| `COMEBACK_ROUTE_RIGHT` | 8.43 | 7.57 | +0.86 |
| `CURL_ROUTE_RIGHT` | 7.05 | 7.43 | -0.38 |
| `DAGGER_ROUTE_LEFT` | 8.16 | 8.52 | -0.36 |
| `GO_ROUTE_RIGHT` | 10.98 | 9.85 | +1.13 |
| `OVER_SHOULDER_ADJUST` | 7.49 | 7.96 | -0.47 |
| `POST_CORNER_ROUTE_LEFT` | 7.94 | 8.28 | -0.34 |
| `POST_CORNER_ROUTE_RIGHT` | 9.84 | 9.07 | +0.77 |
| `RED_ZONE_FADE_RIGHT` | 6.77 | 5.62 | +1.15 |
| `SLANT_ROUTE_LEFT` | 8.11 | 8.94 | -0.83 |
| `SLOT_SAIL_ROUTE_RIGHT` | 7.97 | 8.25 | -0.28 |
| `SLOT_STRIKE_ROUTE_LEFT` | 6.84 | 8.27 | -1.43 |
| `SPEED_OUT_ROUTE_LEFT` | 7.03 | 7.12 | -0.09 |

### 3.2. Hero Pair #2: **Jason Brownlee** (2023, UDFA) vs **Ronnie Bell** (2023, Rd 7 (#253))

- **Thể hình & Track**: Jason Brownlee (**198 lbs**, `4.59s` 40-yd) vs Ronnie Bell (**191 lbs**, `4.54s` 40-yd) — chạy chung **12 bài drill**.
- **Combine Route Kinetics**: `route_max_s` = **10.39** vs **9.78 yd/s** | `pure_kinetics_z` = **+0.67** vs **-0.74** | `size_adjusted_route_kinetics` = **+0.67** vs **-0.80**.
- **Thực chiến NFL**: `nfl_route_p95_s` = **7.77** vs **8.53 yd/s** | `soe_break` = **+0.156** vs **-0.168 yd** | `soe_all` = **+0.094** vs **-0.109 yd**.

| Bài drill chạy chung | Jason Brownlee max_s | Ronnie Bell max_s | Chênh lệch (Gem − Clone) |
| :--- | ---: | ---: | ---: |
| `COMEBACK_ROUTE_RIGHT` | 8.06 | 8.28 | -0.21 |
| `CURL_ROUTE_RIGHT` | 7.53 | 7.00 | +0.53 |
| `DAGGER_ROUTE_LEFT` | 7.88 | 8.09 | -0.21 |
| `GO_ROUTE_RIGHT` | 10.39 | 9.78 | +0.61 |
| `OVER_SHOULDER_ADJUST` | 7.15 | 7.97 | -0.82 |
| `POST_CORNER_ROUTE_LEFT` | 8.36 | 8.39 | -0.03 |
| `POST_CORNER_ROUTE_RIGHT` | 9.84 | 8.75 | +1.09 |
| `RED_ZONE_FADE_RIGHT` | 6.42 | 6.77 | -0.35 |
| `SLANT_ROUTE_LEFT` | 8.43 | 8.46 | -0.04 |
| `SLOT_SAIL_ROUTE_RIGHT` | 8.88 | 8.31 | +0.58 |
| `SLOT_STRIKE_ROUTE_LEFT` | 7.78 | 8.80 | -1.03 |
| `SPEED_OUT_ROUTE_LEFT` | 6.41 | 7.85 | -1.44 |

---

## 4. Kết quả LODCO CV sạch 100% (No Weight-Mixing) dự báo `soe_break`

- **Model A (Baseline)**: `forty, ten_yd_split, ngs_athleticism_score, combine_height, combine_weight`.
- **Model B (100% Pure Sensor — Không pha trộn cân nặng)**: `pure_kinetics_z, route_max_s, vertical_route_max_s_z, route_speed_z, peak_centripetal_accel_z, braking_power_index_z, speed_cut_retention_z, pre_break_jerk`.
- **Model C (Combined — Fold-Internal Residualization Only)**: `Model A + Model B` cộng thêm 2 biến quan hệ sensor-cân nặng được ước lượng **hoàn toàn bên trong tập train của từng fold** (không dùng bất kỳ tham số toàn cục nào từ ngoài fold).

| Cohort | Estimator | Model | OOF $R^2$ | OOF RMSE | Spearman ρ (p) | Pearson r (p) | Δρ vs A [95% Boot CI] | ΔRMSE vs A [95% Boot CI] |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| N=69 (`n_all >= 50`) | RidgeCV | Model A (Baseline) | +0.1003 | 0.1472 | +0.261* (0.030) | +0.324** (0.007) | Ref | Ref |
| N=69 (`n_all >= 50`) | RidgeCV | Model B (Pure Sensor) | +0.0246 | 0.1532 | +0.176 (0.148) | +0.180 (0.138) | -0.085 [-0.366, +0.217] | +0.0061 [-0.0086, +0.0209] |
| N=69 (`n_all >= 50`) | RidgeCV | Model C (Combined Clean) | +0.1696 | 0.1414 | +0.342** (0.004) | +0.417** (0.000) | +0.081 [-0.075, +0.241] | -0.0058 [-0.0149, +0.0031] |
| N=69 (`n_all >= 50`) | BayesianRidge | Model A (Baseline) | +0.1100 | 0.1464 | +0.274* (0.023) | +0.338** (0.005) | Ref | Ref |
| N=69 (`n_all >= 50`) | BayesianRidge | Model B (Pure Sensor) | +0.0163 | 0.1539 | +0.180 (0.138) | +0.176 (0.149) | -0.094 [-0.378, +0.204] | +0.0075 [-0.0078, +0.0225] |
| N=69 (`n_all >= 50`) | BayesianRidge | Model C (Combined Clean) | +0.1737 | 0.1410 | +0.343** (0.004) | +0.421** (0.000) | +0.069 [-0.082, +0.226] | -0.0053 [-0.0139, +0.0029] |
| N=69 (`n_all >= 50`) | ElasticNetCV | Model A (Baseline) | +0.1164 | 0.1458 | +0.248* (0.040) | +0.347** (0.004) | Ref | Ref |
| N=69 (`n_all >= 50`) | ElasticNetCV | Model B (Pure Sensor) | +0.0029 | 0.1549 | +0.191 (0.116) | +0.138 (0.259) | -0.057 [-0.354, +0.264] | +0.0091 [-0.0085, +0.0269] |
| N=69 (`n_all >= 50`) | ElasticNetCV | Model C (Combined Clean) | +0.1431 | 0.1436 | +0.298* (0.013) | +0.394** (0.001) | +0.050 [-0.096, +0.196] | -0.0022 [-0.0105, +0.0060] |
| N=83 (`All REG WRs`) | RidgeCV | Model A (Baseline) | +0.0337 | 0.1431 | +0.160 (0.148) | +0.199. (0.071) | Ref | Ref |
| N=83 (`All REG WRs`) | RidgeCV | Model B (Pure Sensor) | +0.0552 | 0.1415 | +0.256* (0.020) | +0.236* (0.032) | +0.096 [-0.171, +0.359] | -0.0016 [-0.0116, +0.0087] |
| N=83 (`All REG WRs`) | RidgeCV | Model C (Combined Clean) | +0.1308 | 0.1357 | +0.260* (0.018) | +0.363** (0.001) | +0.100 [-0.058, +0.255] | -0.0074 [-0.0151, -0.0000] |
| N=83 (`All REG WRs`) | BayesianRidge | Model A (Baseline) | +0.0608 | 0.1410 | +0.208. (0.059) | +0.259* (0.018) | Ref | Ref |
| N=83 (`All REG WRs`) | BayesianRidge | Model B (Pure Sensor) | +0.0525 | 0.1417 | +0.256* (0.019) | +0.231* (0.036) | +0.049 [-0.217, +0.313] | +0.0006 [-0.0108, +0.0119] |
| N=83 (`All REG WRs`) | BayesianRidge | Model C (Combined Clean) | +0.1236 | 0.1362 | +0.247* (0.025) | +0.354** (0.001) | +0.039 [-0.107, +0.176] | -0.0048 [-0.0119, +0.0023] |
| N=83 (`All REG WRs`) | ElasticNetCV | Model A (Baseline) | +0.0197 | 0.1441 | +0.151 (0.172) | +0.198. (0.072) | Ref | Ref |
| N=83 (`All REG WRs`) | ElasticNetCV | Model B (Pure Sensor) | +0.0503 | 0.1418 | +0.261* (0.017) | +0.229* (0.037) | +0.109 [-0.186, +0.378] | -0.0023 [-0.0146, +0.0123] |
| N=83 (`All REG WRs`) | ElasticNetCV | Model C (Combined Clean) | +0.1281 | 0.1359 | +0.296** (0.007) | +0.358** (0.001) | +0.145 [-0.019, +0.298] | -0.0082 [-0.0164, +0.0004] |

### 4.1. Chi tiết từng fold & Trọng số chuẩn hóa (`ElasticNetCV`, `N = 69`, `No Weight-Mixing`)

| Model | Holdout Year | Train / Val | Top-K | OOF $R^2$ | RMSE | Spearman ρ | Selected Features |
| :--- | :---: | :---: | ---: | ---: | ---: | ---: | :--- |
| Model A | 2023 | 40 / 29 | 4 | +0.1362 | 0.1291 | +0.399 | `combine_weight`, `combine_height`, `ngs_athleticism_score`, `forty` |
| Model A | 2024 | 46 / 23 | 1 | +0.0231 | 0.1357 | +0.073 | `combine_weight` |
| Model A | 2025 | 52 / 17 | 3 | +0.1206 | 0.1816 | +0.306 | `combine_weight`, `ten_yd_split`, `combine_height` |
| Model B (Pure Sensor) | 2023 | 40 / 29 | 2 | -0.0611 | 0.1431 | +0.294 | `pure_kinetics_z`, `route_max_s` |
| Model B (Pure Sensor) | 2024 | 46 / 23 | 1 | +0.0002 | 0.1373 | -0.038 | `vertical_route_max_s_z` |
| Model B (Pure Sensor) | 2025 | 52 / 17 | 4 | +0.0138 | 0.1923 | +0.471 | `pure_kinetics_z`, `route_max_s`, `vertical_route_max_s_z`, `functional_speed_ratio` |
| Model C (Combined) | 2023 | 40 / 29 | 1 | +0.1003 | 0.1317 | +0.408 | `fold_pk_minus_wt_z` |
| Model C (Combined) | 2024 | 46 / 23 | 2 | +0.0840 | 0.1314 | +0.173 | `combine_weight`, `fold_pk_minus_wt_z` |
| Model C (Combined) | 2025 | 52 / 17 | 5 | +0.1804 | 0.1753 | +0.407 | `fold_pk_minus_wt_z`, `combine_weight`, `ten_yd_split`, `pure_kinetics_z`, `combine_height` |

#### Trọng số chuẩn hóa trung bình qua 3 fold (`Model C Clean`, `ElasticNetCV`, `N = 69`)

| Feature | Selected Folds (/3) | Mean Std Coef | Mean |Std Coef| |
| :--- | :---: | ---: | ---: |
| `fold_pk_minus_wt_z` | 3/3 | +0.0343 | 0.0343 |
| `combine_weight` | 2/3 | -0.0280 | 0.0280 |
| `ten_yd_split` | 1/3 | -0.0126 | 0.0126 |
| `combine_height` | 1/3 | +0.0112 | 0.0112 |
| `forty` | 0/3 | +0.0000 | 0.0000 |
| `ngs_athleticism_score` | 0/3 | +0.0000 | 0.0000 |
| `pure_kinetics_z` | 1/3 | +0.0000 | 0.0000 |
| `route_max_s` | 0/3 | +0.0000 | 0.0000 |
| `vertical_route_max_s_z` | 0/3 | +0.0000 | 0.0000 |
| `route_speed_z` | 0/3 | +0.0000 | 0.0000 |
| `peak_centripetal_accel_z` | 0/3 | +0.0000 | 0.0000 |
| `braking_power_index_z` | 0/3 | +0.0000 | 0.0000 |
| `speed_cut_retention_z` | 0/3 | +0.0000 | 0.0000 |
| `pre_break_jerk` | 0/3 | +0.0000 | 0.0000 |
| `functional_speed_ratio` | 0/3 | +0.0000 | 0.0000 |
| `fold_size_adj_kinetics` | 0/3 | +0.0000 | 0.0000 |

