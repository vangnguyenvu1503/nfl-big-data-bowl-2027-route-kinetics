# NFL Big Data Bowl 2027 — WR Feature & Target Validation

- **Thời gian tạo**: `2026-11-20 10:55:07`
- **Đề tài**: *Wide Receiver — From Track Speed to Route Kinetics*
- **Ký hiệu mức ý nghĩa**: `**` (*p < 0.01*), `*` (*p < 0.05*), `.` (*p < 0.10*).

## 1. Chất lượng mô hình `expected_separation` Out-of-Fold & Mục tiêu SOE

- **Tập dữ liệu NFL REG (`play_nullified_by_penalty == "N"`, `combine_position == "WR"`)**: **31,035 plays** từ **83 WR** across 3 mùa giải (`2023`–`2025`).
- **Kiến trúc khử nhiễu chống rò rỉ**: Leave-One-Season-Out `LGBMRegressor` (vocabulary phân loại chỉ fit trên các mùa huấn luyện, dự đoán hoàn toàn Out-of-Fold cho mùa giữ lại) sử dụng 7 biến ngữ cảnh: `route_ran`, `team_coverage_man_zone`, `team_coverage_type`, `cushion`, `lined_up_position`, `in_motion_at_ball_snap`, `pass_length`.
- **Hiệu năng Out-of-Fold tổng thể**: **$R^2 = 0.3100$**, **RMSE = 1.7573 yd**, **MAE = 1.3293 yd** (độ lệch chuẩn mục tiêu gốc: 2.1156 yd → độ lệch chuẩn phần dư SOE: 1.7573 yd).

### 1.1. Kết quả Out-of-Fold theo từng mùa giải

| Mùa giữ lại (Holdout Season) | N Train | N Val | OOF $R^2$ | OOF RMSE (yd) | OOF MAE (yd) |
| :---: | ---: | ---: | ---: | ---: | ---: |
| 2023 | 24,422 | 6,613 | 0.3246 | 1.7902 | 1.3420 |
| 2024 | 20,113 | 10,922 | 0.3313 | 1.7279 | 1.3049 |
| 2025 | 17,535 | 13,500 | 0.2844 | 1.7645 | 1.3428 |
| **Overall OOF** | — | **31,035** | **0.3100** | **1.7573** | **1.3293** |

### 1.2. Mức đóng góp biến ngữ cảnh (Average Gain Share qua 3 folds)

| Biến ngữ cảnh | Gain Share | Avg Gain | Avg Splits |
| :--- | ---: | ---: | ---: |
| `route_ran` | 36.07% | 115,167.1 | 350.3 |
| `cushion` | 24.28% | 77,515.9 | 491.0 |
| `team_coverage_type` | 15.98% | 51,009.2 | 259.7 |
| `team_coverage_man_zone` | 12.66% | 40,404.6 | 41.7 |
| `pass_length` | 3.99% | 12,749.6 | 305.3 |
| `in_motion_at_ball_snap` | 3.76% | 12,004.7 | 87.7 |
| `lined_up_position` | 3.26% | 10,421.6 | 144.3 |

### 1.3. Phân phối mục tiêu SOE sau Empirical Bayes Shrinkage (`N = 83` WR)

- Tham số co cụm Empirical Bayes: $k_{all} = 50$ ($\mu_0 = -0.0022$), $k_{man} = 30$ ($\mu_0 = -0.0208$), $k_{break} = 40$ ($\mu_0 = -0.0052$).

| Target | WRs (n >= 50) | Mean (yd) | Std (yd) | Min (yd) | Median (yd) | Max (yd) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `soe_all` | 69 / 83 | +0.0063 | 0.1440 | -0.2896 | +0.0005 | +0.5508 |
| `soe_man` | 49 / 83 | -0.0107 | 0.1048 | -0.3009 | -0.0061 | +0.2232 |
| `soe_break` | 62 / 83 | -0.0026 | 0.1464 | -0.3074 | +0.0086 | +0.4433 |

---

## 2. Phân đoạn quỹ đạo & Bộ đặc trưng sinh cơ học WR (`N = 107` WR)

- Tổng số lượt chạy `SKILL_DRILLS_WR` đã phân đoạn: **1,529 reps** từ **107 WR**.
- Tham chiếu tốc độ track `ref_track_max_s` cho `functional_speed_ratio`: **94 WR** lấy trực tiếp từ `FORTY_YARD_DASH` tracking, **3 WR** quy đổi tuyến tính từ `forty` time, **10 WR** dùng trung vị nhóm (không chạy 40-yard dash).
- Kiểm tra ranh giới Break Apex toàn cục: `max(apex_idx / n_frames)` trên toàn bộ 1,529 reps là **0.7400** (< 0.8000 — không có rep nào rơi vào 20% khung hình đi bộ cuối bài).

### 2.1. Hồ sơ phân đoạn 8 bài tập `SKILL_DRILLS_WR` trọng tâm

| drill_name | Reps | WRs | Med Frames | Med Start Rel | Med Apex Rel | Max Apex Rel | Med End Rel | Med max_s | Med Decel | Med Curv |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `SLANT_ROUTE_LEFT` | 110 | 107 | 43.0 | 0.116 | 0.405 | 0.692 | 0.765 | 8.26 | 2.04 | 0.140 |
| `GAUNTLET_DRILL` | 200 | 107 | 91.0 | 0.091 | 0.730 | 0.740 | 0.785 | 8.78 | -0.72 | 0.032 |
| `SPEED_OUT_ROUTE_LEFT` | 124 | 107 | 43.5 | 0.093 | 0.600 | 0.729 | 0.767 | 7.01 | -1.40 | 0.185 |
| `DAGGER_ROUTE_LEFT` | 115 | 107 | 64.0 | 0.052 | 0.500 | 0.737 | 0.778 | 8.30 | -2.03 | 0.146 |
| `CURL_ROUTE_RIGHT` | 110 | 103 | 56.0 | 0.054 | 0.500 | 0.719 | 0.776 | 7.11 | -7.18 | 1.780 |
| `COMEBACK_ROUTE_RIGHT` | 110 | 98 | 54.0 | 0.060 | 0.654 | 0.729 | 0.778 | 8.13 | -6.67 | 0.747 |
| `GO_ROUTE_RIGHT` | 105 | 98 | 72.0 | 0.045 | 0.653 | 0.740 | 0.781 | 10.14 | -0.23 | 0.019 |
| `POST_CORNER_ROUTE_RIGHT` | 111 | 96 | 71.0 | 0.050 | 0.439 | 0.731 | 0.780 | 9.56 | -0.04 | 0.086 |

### 2.2. Thống kê phân phối 9 đặc trưng sinh cơ học (`N = 107` WR)

| Đặc trưng | Nulls | Mean | Std | P25 | Median | P75 | Min | Max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Biomech: `route_max_s` (yd/s) | 0 | 10.063 | 0.456 | 9.937 | 10.101 | 10.360 | 8.514 | 10.977 |
| Biomech: `functional_speed_ratio` (route/40 max_s) | 0 | 0.906 | 0.042 | 0.894 | 0.909 | 0.931 | 0.736 | 0.982 |
| Biomech: `pre_break_decel` (yd/s²) | 0 | -2.607 | 0.457 | -2.928 | -2.606 | -2.262 | -3.842 | -1.205 |
| Biomech: `pre_break_jerk` (yd/s³) | 0 | -7.612 | 1.685 | -8.398 | -7.424 | -6.405 | -13.000 | -4.056 |
| Biomech: `apex_speed_retention` (apex_s / stem_max_s) | 0 | 0.729 | 0.036 | 0.715 | 0.731 | 0.754 | 0.586 | 0.824 |
| Biomech: `break_curvature` (rad/yd) | 0 | 0.543 | 0.122 | 0.450 | 0.536 | 0.608 | 0.265 | 0.956 |
| Biomech: `post_break_exit_burst` (yd/s²) | 0 | 2.400 | 0.387 | 2.164 | 2.394 | 2.619 | 1.184 | 3.528 |
| Biomech: `gauntlet_speed_retention` (catch/peak s) | 0 | 0.859 | 0.046 | 0.848 | 0.869 | 0.883 | 0.648 | 0.926 |
| Biomech: `gauntlet_path_straightness` (net/path) | 0 | 0.999 | 0.001 | 0.999 | 0.999 | 0.999 | 0.995 | 1.000 |

---

## 3. Kiểm chứng tín hiệu: Đặc trưng sinh cơ học mới vs Baseline Combine

### 3.1. Nhóm chuẩn `n_all >= 50` snaps (`N = 69` WR có dữ liệu ổn định)

| Đặc trưng | N | soe_all ρ (p) | soe_all r (p) | soe_man ρ (p) | soe_man r (p) | soe_break ρ (p) | soe_break r (p) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Biomech: `route_max_s` (yd/s) | 69 | +0.170 (0.163) | +0.162 (0.182) | +0.079 (0.518) | +0.092 (0.454) | +0.270* (0.025) | +0.287* (0.017) |
| Biomech: `functional_speed_ratio` (route/40 max_s) | 69 | +0.000 (1.000) | +0.057 (0.644) | +0.017 (0.892) | +0.039 (0.748) | +0.103 (0.400) | +0.200 (0.100) |
| Biomech: `pre_break_decel` (yd/s²) | 69 | +0.135 (0.269) | +0.090 (0.463) | +0.127 (0.300) | +0.108 (0.379) | +0.091 (0.457) | +0.069 (0.572) |
| Biomech: `pre_break_jerk` (yd/s³) | 69 | +0.160 (0.189) | +0.112 (0.358) | -0.071 (0.562) | -0.063 (0.608) | +0.060 (0.625) | +0.035 (0.775) |
| Biomech: `apex_speed_retention` (apex_s / stem_max_s) | 69 | +0.117 (0.337) | +0.022 (0.856) | +0.211. (0.082) | +0.150 (0.219) | +0.166 (0.173) | +0.140 (0.251) |
| Biomech: `break_curvature` (rad/yd) | 69 | -0.083 (0.496) | -0.134 (0.271) | -0.151 (0.216) | -0.193 (0.113) | +0.009 (0.942) | -0.024 (0.845) |
| Biomech: `post_break_exit_burst` (yd/s²) | 69 | -0.025 (0.837) | -0.004 (0.977) | -0.168 (0.166) | -0.173 (0.156) | -0.131 (0.283) | -0.100 (0.413) |
| Biomech: `gauntlet_speed_retention` (catch/peak s) | 69 | -0.044 (0.722) | -0.012 (0.922) | -0.012 (0.921) | +0.020 (0.868) | +0.018 (0.885) | -0.066 (0.593) |
| Biomech: `gauntlet_path_straightness` (net/path) | 69 | -0.002 (0.985) | -0.011 (0.927) | +0.116 (0.341) | +0.151 (0.214) | -0.038 (0.754) | -0.068 (0.580) |
| Baseline: `forty` (s) | 64 | -0.144 (0.256) | -0.132 (0.300) | -0.032 (0.802) | -0.044 (0.728) | -0.140 (0.270) | -0.081 (0.527) |
| Baseline: `ten_yd_split` (s) | 64 | -0.154 (0.223) | -0.235. (0.062) | -0.165 (0.193) | -0.211. (0.095) | -0.223. (0.076) | -0.215. (0.088) |
| Baseline: `ngs_athleticism_score` | 69 | -0.121 (0.322) | -0.084 (0.494) | -0.132 (0.280) | -0.115 (0.347) | -0.094 (0.442) | -0.085 (0.487) |
| Baseline: `draft_overall_pick` | 61 | +0.170 (0.191) | +0.125 (0.337) | +0.268* (0.036) | +0.261* (0.043) | -0.013 (0.918) | +0.008 (0.952) |

### 3.2. Toàn bộ mẫu WR có dữ liệu NFL REG (`N = 83` WR — EB-shrunk)

| Đặc trưng | N | soe_all ρ (p) | soe_all r (p) | soe_man ρ (p) | soe_man r (p) | soe_break ρ (p) | soe_break r (p) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Biomech: `route_max_s` (yd/s) | 83 | +0.182. (0.099) | +0.166 (0.135) | +0.123 (0.268) | +0.101 (0.364) | +0.265* (0.015) | +0.283** (0.010) |
| Biomech: `functional_speed_ratio` (route/40 max_s) | 83 | +0.021 (0.853) | +0.065 (0.558) | +0.073 (0.509) | +0.055 (0.619) | +0.092 (0.410) | +0.197. (0.074) |
| Biomech: `pre_break_decel` (yd/s²) | 83 | +0.078 (0.482) | +0.059 (0.599) | +0.089 (0.422) | +0.088 (0.430) | +0.089 (0.424) | +0.072 (0.519) |
| Biomech: `pre_break_jerk` (yd/s³) | 83 | +0.090 (0.419) | +0.076 (0.494) | -0.047 (0.676) | -0.043 (0.697) | +0.056 (0.614) | +0.049 (0.661) |
| Biomech: `apex_speed_retention` (apex_s / stem_max_s) | 83 | +0.062 (0.575) | -0.005 (0.962) | +0.164 (0.139) | +0.119 (0.283) | +0.138 (0.215) | +0.122 (0.272) |
| Biomech: `break_curvature` (rad/yd) | 83 | -0.062 (0.580) | -0.114 (0.306) | -0.098 (0.379) | -0.149 (0.180) | -0.007 (0.950) | -0.025 (0.820) |
| Biomech: `post_break_exit_burst` (yd/s²) | 83 | -0.008 (0.942) | +0.018 (0.868) | -0.155 (0.161) | -0.140 (0.207) | -0.143 (0.198) | -0.104 (0.352) |
| Biomech: `gauntlet_speed_retention` (catch/peak s) | 83 | -0.031 (0.782) | +0.012 (0.915) | -0.022 (0.845) | +0.012 (0.914) | +0.020 (0.857) | -0.036 (0.750) |
| Biomech: `gauntlet_path_straightness` (net/path) | 83 | +0.039 (0.728) | +0.012 (0.917) | +0.135 (0.223) | +0.141 (0.205) | +0.077 (0.488) | +0.010 (0.930) |
| Baseline: `forty` (s) | 75 | -0.158 (0.177) | -0.133 (0.254) | -0.015 (0.900) | -0.022 (0.850) | -0.159 (0.173) | -0.095 (0.416) |
| Baseline: `ten_yd_split` (s) | 75 | -0.134 (0.252) | -0.188 (0.106) | -0.121 (0.302) | -0.135 (0.248) | -0.171 (0.143) | -0.163 (0.163) |
| Baseline: `ngs_athleticism_score` | 83 | -0.069 (0.534) | -0.050 (0.655) | -0.111 (0.318) | -0.103 (0.356) | -0.036 (0.747) | -0.049 (0.660) |
| Baseline: `draft_overall_pick` | 73 | +0.090 (0.450) | +0.069 (0.564) | +0.158 (0.183) | +0.168 (0.155) | -0.083 (0.487) | -0.052 (0.660) |

---

## 4. Nhận xét & Đánh giá bước tiếp theo

- **Mô hình khử nhiễu `expected_separation`**: Đạt OOF $R^2 = 0.3100$ (RMSE 1.7573 yd) và ổn định qua cả 3 mùa (`2023`–`2025`), loại bỏ phần lớn phương sai đến từ `route_ran`, `cushion`, `team_coverage_type` và `pass_length`.
- **So sánh tín hiệu mạnh nhất trên nhóm chuẩn `N = 69`**: Đặc trưng sinh cơ học nổi bật nhất là `route_max_s` trên `soe_break` ($\rho = +0.270$, $p = 0.0246$), so với baseline mạnh nhất `draft_overall_pick` trên `soe_man` ($\rho = +0.268$, $p = 0.0365$).
- **Ý nghĩa thực tế**: `route_max_s` trong các bài route Combine vượt trội `forty`, `ten_yd_split` và `ngs_athleticism_score` khi dự báo khả năng tạo khoảng trống trên các route bẻ góc (`soe_break`). Ở chiều ngược lại, các biến đơn lẻ như `pre_break_decel` hay `break_curvature` có tương quan đơn biến thấp hơn vì bị chi phối bởi phong cách/cân nặng/loại route mà WR chạy; điều này gợi mở bước tiếp theo là chuẩn hóa theo từng nhóm route (hoặc đưa vào mô hình phi tuyến / tương tác giữa `route_max_s`, `pre_break_decel`, `break_curvature`, và kích thước cơ thể).

