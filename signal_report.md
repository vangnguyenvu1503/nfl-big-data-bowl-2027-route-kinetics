# NFL Big Data Bowl 2027 — Preliminary Signal Discovery (OL vs WR vs DL)

- **Thời gian tạo**: `2026-11-20 10:36:13`
- **Ngưỡng mẫu kiểm định**: Tối thiểu **50 snaps hợp lệ** ở NFL Regular Season (2023–2025).
- **Ký hiệu mức ý nghĩa thống kê**: `**` (*p < 0.01*), `*` (*p < 0.05*), `.` (*p < 0.10*).

## 1. Chuẩn hóa dữ liệu & Định nghĩa đặc trưng vật lý

### 1.1. Các bước làm sạch và chuẩn hóa

| Bước chuẩn hóa | Số lượng trước → sau | Ghi chú |
| :--- | :--- | :--- |
| `games.csv` (`season_type == "REG"`) | 1,002 → **816** trận | Chỉ giữ Regular Season (272 trận × 3 mùa) |
| Chuẩn hóa mã đội `LA` → `LAR` | 43 trận (`games`); 9,897 dòng (`player_play`) | Còn đúng **32** đội ở `games` và **32** đội ở `player_play` |
| `player_play.csv` (`play_nullified_by_penalty == "N"` + `REG`) | 314,197 → 301,707 (N) → **240,412** (REG) | Loại bỏ các play bị penalty hủy và các trận PRE/POST |
| `combine_tracking.csv` (`entity_type == "PLAYER"`) | 463,189 → **431,094** dòng | Loại bỏ 32,095 dòng theo dõi bóng (`BALL`) |
| Gộp tên drill OL bị lệch gạch nối | `PASS_PRO-MIRROR_DRILL` (7,020) + `PASS_PRO_MIRROR_DRILL` (3,401) | Gộp thành `PASS_PRO_MIRROR_DRILL` |
| Gộp tên drill WR bị thiếu chữ `ROUTE` | `SPEED_OUT_LEFT` (1,675) + `SPEED_OUT_ROUTE_LEFT` (3,829) | Gộp thành `SPEED_OUT_ROUTE_LEFT` (tổng số `drill_name`: 56 → 54) |

### 1.2. Định nghĩa 4 đặc trưng vật lý từ Skill Drills (`combine_tracking.csv`)

- **Lưu ý kỹ thuật về gia tốc `a`**: Cột `a` gốc trong `combine_tracking.csv` chỉ là độ lớn vô hướng (`a >= 0.00`, `max = 10.72 yd/s²`). Do đó, trong mỗi `(nfl_id, event_id)` sắp xếp theo `time` (bước lấy mẫu cố định `dt = 0.1s` ở 100% khung hình liên tiếp), ta tính gia tốc tiếp tuyến có dấu `signed_a = (s_t - s_{t-1}) / dt` để đo lực hãm phanh/giảm tốc thực tế.
- **`max_s`** (`yd/s`): `max(s)` trên toàn bộ khung hình Skill Drills của cầu thủ.
- **`max_a`** (`yd/s²`): `max(a)` từ đại lượng gia tốc tracking.
- **`min_a`** (`yd/s²`): `min(signed_a)` — gia tốc âm cực đại phản ánh khả năng giảm tốc/hãm phanh đột ngột (giá trị càng âm tức phanh càng gắt).
- **`mean_abs_d_dir`** (`deg/s`): Trung bình tốc độ đổi hướng góc `min(|dir_t - dir_{t-1}|, 360 - |dir_t - dir_{t-1}|) / dt` có xử lý xoay vòng `0°/360°`.

---

## 2. Độ phủ mẫu thực tế ở NFL Regular Season & Phân phối biến mục tiêu

### 2.1. Nhóm OL (`combine_position == "OL"`, lọc `pass_rushers_encountered > 0`)

- **Độ phủ mẫu**: **46,492** snaps hợp lệ từ **89** cầu thủ OL có dữ liệu NFL REG; trong đó **70 / 121** cầu thủ Combine OL (57.9% toàn khóa Combine, 78.7% nhóm có ra sân REG) đạt ngưỡng **>= 50 snaps**.

| Vị trí (`nfl_position`) | Có >= 1 snap | Đạt >= 50 snaps | Tổng snaps |
| :--- | ---: | ---: | ---: |
| G | 39 | 31 | 20,843 |
| T | 35 | 30 | 19,714 |
| C | 15 | 9 | 5,935 |

| Chỉ tiêu | N | Mean | Std | P25 | Median | P75 | Min | Max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Play-level `peak_pressure_probability_allowed` | 46,492 | 0.284 | 0.240 | 0.135 | 0.188 | 0.314 | 0.019 | 1.000 |
| Play-level `pressure_allowed` (tỷ lệ) | 46,492 | 8.83% | 28.37% | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% |
| Player-level (>=50 snaps) `valid_snaps` | 70 | 658.757 | 467.781 | 248.000 | 557.500 | 983.000 | 57.000 | 1743.000 |
| Player-level (>=50 snaps) `peak_pressure_prob_allowed` | 70 | 0.289 | 0.037 | 0.264 | 0.282 | 0.313 | 0.210 | 0.389 |
| Player-level (>=50 snaps) `pressure_allowed_rate` | 70 | 9.25% | 2.78% | 7.28% | 9.08% | 10.98% | 3.03% | 18.56% |

### 2.2. Nhóm WR (`combine_position == "WR"`, lọc `route_ran` không null)

- **Độ phủ mẫu (tất cả route)**: **35,427** route snaps (**31,035** snaps có `separation_at_pass_forward`) từ **83** WR; trong đó **69 / 107** WR Combine đạt **>= 50 snaps** có đo separation.
- **Độ phủ mẫu (nhóm Break Routes: `OUT, IN, HITCH, SLANT, CORNER`)**: **18,382** route snaps (**16,062** snaps có separation); **62 / 107** WR đạt **>= 50 snaps** có separation.

| route_ran | Break? | Route Snaps | Sep Snaps | WRs | WRs (>=50) | Sep Mean | Sep Std | Sep Median | Target Rate |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GO | No | 6,834 | 6,076 | 81 | 48 | 2.604 | 1.863 | 1.996 | 11.97% |
| HITCH | Yes | 5,725 | 5,009 | 82 | 36 | 3.068 | 1.956 | 2.721 | 19.63% |
| CROSS | No | 4,480 | 3,816 | 79 | 33 | 3.318 | 2.217 | 2.824 | 20.38% |
| IN | Yes | 4,027 | 3,417 | 79 | 30 | 2.939 | 1.785 | 2.630 | 12.69% |
| POST | No | 3,647 | 3,122 | 81 | 28 | 2.977 | 1.850 | 2.634 | 11.98% |
| OUT | Yes | 3,389 | 2,994 | 81 | 26 | 3.022 | 1.940 | 2.609 | 27.94% |
| CORNER | Yes | 2,956 | 2,528 | 81 | 21 | 2.944 | 1.914 | 2.565 | 10.93% |
| SLANT | Yes | 2,285 | 2,114 | 80 | 14 | 2.794 | 1.998 | 2.200 | 24.64% |
| FLAT | No | 1,032 | 942 | 70 | 5 | 4.610 | 2.668 | 4.216 | 20.45% |
| SCREEN | No | 981 | 951 | 69 | 7 | 6.657 | 2.531 | 6.528 | 65.75% |
| WHEEL | No | 52 | 48 | 23 | 0 | 3.498 | 2.053 | 3.004 | 23.08% |
| ANGLE | No | 19 | 18 | 11 | 0 | 4.516 | 2.376 | 4.483 | 47.37% |

| Chỉ tiêu | N | Mean | Std | P25 | Median | P75 | Min | Max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Play-level `separation_at_pass_forward` (All routes) | 31,035 | 3.110 | 2.116 | 1.404 | 2.626 | 4.317 | 0.150 | 17.716 |
| Play-level `separation_at_pass_forward` (Break routes) | 16,062 | 2.976 | 1.918 | 1.379 | 2.596 | 4.162 | 0.228 | 17.568 |
| Player-level (>=50 sep snaps) `mean_sep_all` | 69 | 3.113 | 0.415 | 2.861 | 3.014 | 3.327 | 2.172 | 5.039 |
| Player-level (>=50 sep snaps) `mean_sep_break` | 62 | 3.005 | 0.244 | 2.831 | 2.976 | 3.145 | 2.578 | 3.629 |
| Player-level (>=50 route snaps) `target_rate_all` | 69 | 16.82% | 4.97% | 13.91% | 16.63% | 19.47% | 4.81% | 33.57% |
| Player-level (>=50 break snaps) `target_rate_break` | 62 | 17.41% | 5.13% | 14.06% | 17.15% | 20.78% | 6.00% | 32.21% |

### 2.3. Nhóm DL (`combine_position == "DL"`: EDGE `DE/OLB` và DT/IDL `DT/NT`)

- **Lưu ý cấu trúc dữ liệu DL**:
  - Tổng snaps DL ở REG: **50,407** snaps (98 cầu thủ); trong đó có **28,970** pass-rush snaps (`unblocked_pressure` non-null) và **33,417** snaps có `player_get_off` (25,294 trên pass-rush snaps; 82 cầu thủ có >= 50 get-off snaps tổng hợp).
  - Với ngưỡng **>= 50 pass-rush snaps**: có **79 / 118** cầu thủ DL (gồm **45 EDGE** và **34 DT/IDL**).
  - Biến `time_to_pressure` và `quick_pressure` chỉ ghi nhận khi cầu thủ thực sự tạo áp lực (2,704 lượt pressure / 28,970 pass-rush snaps = 9.33%). Để tránh sai lệch mẫu số, báo cáo tính đồng thời **`quick_pressure_rate`** = `số quick_pressure / tổng pass_rush_snaps` (tỷ lệ tạo áp lực nhanh thực tế) và **`quick_pressure_share`** = `P(quick_pressure | tạo được pressure)` ở nhóm có `>= 10 pressures` (**51** cầu thủ).

| Chỉ tiêu | N | Mean | Std | P25 | Median | P75 | Min | Max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Play-level `player_get_off` (Pass-rush snaps) | 25,294 | 0.962 | 0.268 | 0.779 | 0.917 | 1.092 | 0.004 | 2.000 |
| Play-level `time_to_pressure` (When pressure > 0) | 2,704 | 3.036 | 1.026 | 2.400 | 2.800 | 3.400 | 0.800 | 9.600 |
| Play-level `quick_pressure_rate` (Per pass-rush snap) | 28,970 | 2.78% | 16.44% | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% |
| Player-level (>=50 PR snaps) `mean_player_get_off` | 79 | 0.995 | 0.107 | 0.909 | 0.995 | 1.088 | 0.775 | 1.203 |
| Player-level (>=50 PR snaps) `mean_time_to_pressure` (N=77 có >=1 PR) | 78 | 3.189 | 0.459 | 2.914 | 3.110 | 3.391 | 2.611 | 5.800 |
| Player-level (>=50 PR snaps, >=10 press) `mean_time_to_pressure` | 51 | 3.154 | 0.330 | 2.933 | 3.131 | 3.341 | 2.611 | 4.264 |
| Player-level (>=50 PR snaps) `quick_pressure_rate` | 79 | 1.99% | 1.57% | 0.87% | 1.90% | 2.82% | 0.00% | 7.65% |
| Player-level (>=50 PR snaps, >=10 press) `quick_pressure_share` | 51 | 25.23% | 10.92% | 19.61% | 25.00% | 33.57% | 0.00% | 42.95% |
| EDGE (>=50 PR snaps) `mean_player_get_off` | 45 | 0.933 | 0.081 | 0.884 | 0.920 | 0.999 | 0.775 | 1.097 |
| DT/IDL (>=50 PR snaps) `mean_player_get_off` | 34 | 1.077 | 0.080 | 1.014 | 1.091 | 1.127 | 0.889 | 1.203 |
| EDGE (>=50 PR snaps) `quick_pressure_rate` | 45 | 2.63% | 1.67% | 1.48% | 2.43% | 3.65% | 0.00% | 7.65% |
| DT/IDL (>=50 PR snaps) `quick_pressure_rate` | 34 | 1.16% | 0.94% | 0.00% | 1.00% | 2.00% | 0.00% | 3.08% |

---

## 3. Kiểm tra tín hiệu sơ bộ (Skill Drills vs `ngs_athleticism_score` & `forty`)

### 3.1. Nhóm OL (`valid_snaps >= 50`)

#### Target: `peak_pressure_prob_allowed` (càng thấp càng tốt cho OL)

| Đặc trưng | N | Pearson r | p-val (P) | Spearman ρ | p-val (S) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Skill Drill `max_s` (yd/s) | 70 | +0.116 | 0.3378 | +0.101 | 0.4065 |
| Skill Drill `max_a` (yd/s²) | 70 | -0.077 | 0.5281 | +0.051 | 0.6758 |
| Skill Drill `min_a` (signed ds/dt, yd/s²) | 70 | +0.112 | 0.3564 | +0.180 | 0.1365 |
| Skill Drill `mean_abs_d_dir` (deg/s) | 70 | -0.023 | 0.8498 | -0.041 | 0.7353 |
| Baseline `ngs_athleticism_score` | 70 | -0.054 | 0.6567 | -0.095 | 0.4358 |
| Baseline `forty` (s) | 56 | -0.095 | 0.4854 | -0.074 | 0.5874 |

#### Target: `pressure_allowed_rate` (càng thấp càng tốt cho OL)

| Đặc trưng | N | Pearson r | p-val (P) | Spearman ρ | p-val (S) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Skill Drill `max_s` (yd/s) | 70 | +0.080 | 0.5126 | +0.085 | 0.4820 |
| Skill Drill `max_a` (yd/s²) | 70 | -0.052 | 0.6716 | +0.092 | 0.4477 |
| Skill Drill `min_a` (signed ds/dt, yd/s²) | 70 | +0.086 | 0.4783 | +0.114 | 0.3459 |
| Skill Drill `mean_abs_d_dir` (deg/s) | 70 | -0.082 | 0.5022 | -0.105 | 0.3889 |
| Baseline `ngs_athleticism_score` | 70 | -0.093 | 0.4423 | -0.128 | 0.2891 |
| Baseline `forty` (s) | 56 | -0.017 | 0.8982 | -0.035 | 0.7999 |

### 3.2. Nhóm WR (`>= 50` snaps)

#### Target: `mean_sep_all` — Separation trung bình (Tất cả route)

| Đặc trưng | N | Pearson r | p-val (P) | Spearman ρ | p-val (S) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Skill Drill `max_s` (yd/s) | 69 | +0.248* | 0.0402 | +0.262* | 0.0294 |
| Skill Drill `max_a` (yd/s²) | 69 | -0.095 | 0.4393 | -0.012 | 0.9221 |
| Skill Drill `min_a` (signed ds/dt, yd/s²) | 69 | -0.044 | 0.7207 | -0.027 | 0.8239 |
| Skill Drill `mean_abs_d_dir` (deg/s) | 69 | +0.037 | 0.7617 | +0.047 | 0.7034 |
| Baseline `ngs_athleticism_score` | 69 | -0.106 | 0.3867 | -0.154 | 0.2064 |
| Baseline `forty` (s) | 64 | -0.147 | 0.2464 | -0.094 | 0.4600 |

#### Target: `mean_sep_break` — Separation trung bình (Break routes: `OUT, IN, HITCH, SLANT, CORNER`)

| Đặc trưng | N | Pearson r | p-val (P) | Spearman ρ | p-val (S) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Skill Drill `max_s` (yd/s) | 62 | +0.254* | 0.0465 | +0.276* | 0.0297 |
| Skill Drill `max_a` (yd/s²) | 62 | -0.028 | 0.8295 | -0.067 | 0.6056 |
| Skill Drill `min_a` (signed ds/dt, yd/s²) | 62 | -0.106 | 0.4117 | -0.059 | 0.6491 |
| Skill Drill `mean_abs_d_dir` (deg/s) | 62 | +0.102 | 0.4303 | +0.090 | 0.4865 |
| Baseline `ngs_athleticism_score` | 62 | -0.034 | 0.7929 | -0.050 | 0.6975 |
| Baseline `forty` (s) | 57 | -0.152 | 0.2605 | -0.196 | 0.1432 |

#### Target: `target_rate_all` — Tỷ lệ được target (Tất cả route)

| Đặc trưng | N | Pearson r | p-val (P) | Spearman ρ | p-val (S) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Skill Drill `max_s` (yd/s) | 69 | +0.081 | 0.5059 | +0.132 | 0.2801 |
| Skill Drill `max_a` (yd/s²) | 69 | +0.121 | 0.3222 | +0.159 | 0.1919 |
| Skill Drill `min_a` (signed ds/dt, yd/s²) | 69 | +0.004 | 0.9729 | -0.038 | 0.7547 |
| Skill Drill `mean_abs_d_dir` (deg/s) | 69 | +0.113 | 0.3537 | +0.134 | 0.2733 |
| Baseline `ngs_athleticism_score` | 69 | -0.044 | 0.7208 | +0.027 | 0.8246 |
| Baseline `forty` (s) | 64 | -0.109 | 0.3905 | -0.092 | 0.4687 |

#### Target: `target_rate_break` — Tỷ lệ được target (Break routes)

| Đặc trưng | N | Pearson r | p-val (P) | Spearman ρ | p-val (S) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Skill Drill `max_s` (yd/s) | 62 | +0.055 | 0.6725 | +0.074 | 0.5686 |
| Skill Drill `max_a` (yd/s²) | 62 | +0.097 | 0.4512 | +0.126 | 0.3295 |
| Skill Drill `min_a` (signed ds/dt, yd/s²) | 62 | +0.027 | 0.8350 | +0.001 | 0.9915 |
| Skill Drill `mean_abs_d_dir` (deg/s) | 62 | +0.035 | 0.7846 | -0.014 | 0.9135 |
| Baseline `ngs_athleticism_score` | 62 | -0.097 | 0.4544 | -0.085 | 0.5121 |
| Baseline `forty` (s) | 57 | -0.014 | 0.9188 | +0.048 | 0.7252 |

### 3.3. Nhóm DL (`pass_rush_snaps >= 50`)

#### Target: `mean_player_get_off` — Thời gian xuất phát sau snap (càng thấp càng nhanh)

| Đặc trưng | N | Pearson r | p-val (P) | Spearman ρ | p-val (S) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Skill Drill `max_s` (yd/s) | 79 | -0.668** | 0.0000 | -0.682** | 0.0000 |
| Skill Drill `max_a` (yd/s²) | 79 | -0.549** | 0.0000 | -0.550** | 0.0000 |
| Skill Drill `min_a` (signed ds/dt, yd/s²) | 79 | +0.413** | 0.0002 | +0.410** | 0.0002 |
| Skill Drill `mean_abs_d_dir` (deg/s) | 79 | +0.565** | 0.0000 | +0.568** | 0.0000 |
| Baseline `ngs_athleticism_score` | 79 | -0.210. | 0.0637 | -0.187. | 0.0998 |
| Baseline `forty` (s) | 63 | +0.758** | 0.0000 | +0.776** | 0.0000 |

#### Target: `mean_time_to_pressure` — Thời gian gây áp lực (tất cả DL có >= 1 pressure)

| Đặc trưng | N | Pearson r | p-val (P) | Spearman ρ | p-val (S) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Skill Drill `max_s` (yd/s) | 78 | -0.235* | 0.0386 | -0.181 | 0.1119 |
| Skill Drill `max_a` (yd/s²) | 78 | -0.156 | 0.1739 | -0.040 | 0.7277 |
| Skill Drill `min_a` (signed ds/dt, yd/s²) | 78 | +0.075 | 0.5125 | +0.100 | 0.3833 |
| Skill Drill `mean_abs_d_dir` (deg/s) | 78 | +0.078 | 0.4995 | +0.108 | 0.3484 |
| Baseline `ngs_athleticism_score` | 78 | -0.255* | 0.0244 | -0.186 | 0.1038 |
| Baseline `forty` (s) | 62 | +0.266* | 0.0363 | +0.154 | 0.2330 |

#### Target: `mean_time_to_pressure` — Kiểm tra độ nhạy trên nhóm có >= 10 pressures

| Đặc trưng | N | Pearson r | p-val (P) | Spearman ρ | p-val (S) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Skill Drill `max_s` (yd/s) | 51 | -0.283* | 0.0446 | -0.308* | 0.0278 |
| Skill Drill `max_a` (yd/s²) | 51 | -0.179 | 0.2094 | -0.061 | 0.6707 |
| Skill Drill `min_a` (signed ds/dt, yd/s²) | 51 | +0.151 | 0.2908 | +0.109 | 0.4463 |
| Skill Drill `mean_abs_d_dir` (deg/s) | 51 | +0.262. | 0.0638 | +0.278* | 0.0486 |
| Baseline `ngs_athleticism_score` | 51 | -0.302* | 0.0312 | -0.278* | 0.0481 |
| Baseline `forty` (s) | 39 | +0.361* | 0.0240 | +0.263 | 0.1060 |

#### Target: `quick_pressure_rate` — Tỷ lệ tạo quick pressure trên mỗi pass-rush snap

| Đặc trưng | N | Pearson r | p-val (P) | Spearman ρ | p-val (S) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Skill Drill `max_s` (yd/s) | 79 | +0.391** | 0.0004 | +0.416** | 0.0001 |
| Skill Drill `max_a` (yd/s²) | 79 | +0.365** | 0.0009 | +0.350** | 0.0016 |
| Skill Drill `min_a` (signed ds/dt, yd/s²) | 79 | -0.378** | 0.0006 | -0.353** | 0.0014 |
| Skill Drill `mean_abs_d_dir` (deg/s) | 79 | -0.293** | 0.0087 | -0.229* | 0.0428 |
| Baseline `ngs_athleticism_score` | 79 | +0.222* | 0.0490 | +0.217. | 0.0547 |
| Baseline `forty` (s) | 63 | -0.493** | 0.0000 | -0.491** | 0.0000 |

#### Target: `quick_pressure_share` — Tỷ lệ quick pressure có điều kiện (>= 10 pressures)

| Đặc trưng | N | Pearson r | p-val (P) | Spearman ρ | p-val (S) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Skill Drill `max_s` (yd/s) | 51 | +0.295* | 0.0354 | +0.334* | 0.0166 |
| Skill Drill `max_a` (yd/s²) | 51 | +0.166 | 0.2433 | +0.138 | 0.3338 |
| Skill Drill `min_a` (signed ds/dt, yd/s²) | 51 | -0.150 | 0.2944 | -0.146 | 0.3060 |
| Skill Drill `mean_abs_d_dir` (deg/s) | 51 | -0.318* | 0.0230 | -0.300* | 0.0325 |
| Baseline `ngs_athleticism_score` | 51 | +0.266. | 0.0589 | +0.264. | 0.0609 |
| Baseline `forty` (s) | 39 | -0.402* | 0.0111 | -0.357* | 0.0256 |

#### Kiểm tra tách nhóm DL: `EDGE` (`DE/OLB`) vs `DT/IDL` (`DT/NT`)

| Phân nhóm & Target | N | Best Skill (Pearson) | Best Skill (Spearman) | Best Baseline (Spearman) |
| :--- | ---: | :--- | :--- | :--- |
| EDGE — `mean_player_get_off` | 45 | `mean_abs_d_dir`: +0.456** (p=0.002) | `max_a`: -0.313* (p=0.036) | `forty`: +0.599** (p=0.000) |
| DT/IDL — `mean_player_get_off` | 34 | `max_s`: -0.448** (p=0.008) | `max_s`: -0.463** (p=0.006) | `forty`: +0.579** (p=0.001) |
| EDGE — `mean_time_to_pressure` | 44 | `max_s`: -0.284. (p=0.061) | `max_s`: -0.294. (p=0.053) | `forty`: +0.328. (p=0.058) |
| DT/IDL — `mean_time_to_pressure` | 34 | `mean_abs_d_dir`: -0.304. (p=0.081) | `mean_abs_d_dir`: -0.342* (p=0.048) | `ngs_athleticism_score`: -0.229 (p=0.192) |
| EDGE — `quick_pressure_rate` | 45 | `min_a`: -0.190 (p=0.211) | `min_a`: -0.141 (p=0.356) | `ngs_athleticism_score`: +0.237 (p=0.117) |
| DT/IDL — `quick_pressure_rate` | 34 | `min_a`: -0.418* (p=0.014) | `min_a`: -0.421* (p=0.013) | `forty`: -0.473* (p=0.011) |

### 3.4. Kiểm chứng nhanh: Toàn bộ Skill Drills vs Nhóm Drill chuyên biệt theo bài toán

- **OL chuyên biệt**: `PASS_PRO_MIRROR_DRILL, PASS_RUSH_DROP`.
- **WR chuyên biệt (Break/Route)**: `CURL_ROUTE_RIGHT, COMEBACK_ROUTE_RIGHT, SLANT_ROUTE_LEFT, SPEED_OUT_ROUTE_LEFT, DAGGER_ROUTE_LEFT`.
- **DL chuyên biệt (Pass Rush)**: `PASS_RUSH_DRILL, RUN_THE_HOOP_DRILL`.

| Hướng & Target | All Skill Drills (Spearman) | Drill chuyên biệt (Spearman) | Baseline (Spearman) |
| :--- | :--- | :--- | :--- |
| OL: `peak_pressure_prob_allowed` | `min_a`: +0.180 (p=0.136) | `max_s`: -0.068 (p=0.578) | `ngs_athleticism_score`: -0.095 (p=0.436) |
| OL: `pressure_allowed_rate` | `min_a`: +0.114 (p=0.346) | `min_a`: -0.089 (p=0.465) | `ngs_athleticism_score`: -0.128 (p=0.289) |
| WR: `mean_sep_break` | `max_s`: +0.276* (p=0.030) | `min_a`: -0.089 (p=0.493) | `forty`: -0.196 (p=0.143) |
| WR: `target_rate_break` | `max_a`: +0.126 (p=0.329) | `max_a`: +0.103 (p=0.427) | `ngs_athleticism_score`: -0.085 (p=0.512) |
| DL: `mean_player_get_off` | `max_s`: -0.682** (p=0.000) | `max_s`: -0.627** (p=0.000) | `forty`: +0.776** (p=0.000) |
| DL: `mean_time_to_pressure` | `max_s`: -0.181 (p=0.112) | `max_a`: -0.097 (p=0.396) | `ngs_athleticism_score`: -0.186 (p=0.104) |
| DL: `quick_pressure_rate` | `max_s`: +0.416** (p=0.000) | `max_s`: +0.305** (p=0.006) | `forty`: -0.491** (p=0.000) |

---

## 4. Tổng hợp so sánh tính khả thi 3 hướng đề tài (OL vs WR vs DL)

| Nhóm | Target chính | Cầu thủ (>=50 snaps) | Snaps REG | Best Skill (Pearson) | Best Skill (Spearman) | Best Baseline (Spearman) |
| :--- | :--- | ---: | ---: | :--- | :--- | :--- |
| OL | `peak_pressure_prob_allowed` | 70 | 46,492 | `max_s`: +0.116 (p=0.338) | `min_a`: +0.180 (p=0.136) | `ngs_athleticism_score`: -0.095 (p=0.436) |
| OL | `pressure_allowed_rate` | 70 | 46,492 | `min_a`: +0.086 (p=0.478) | `min_a`: +0.114 (p=0.346) | `ngs_athleticism_score`: -0.128 (p=0.289) |
| WR (All) | `mean_sep_all` | 69 | 35,427 | `max_s`: +0.248* (p=0.040) | `max_s`: +0.262* (p=0.029) | `ngs_athleticism_score`: -0.154 (p=0.206) |
| WR (Break) | `mean_sep_break` | 62 | 18,382 | `max_s`: +0.254* (p=0.047) | `max_s`: +0.276* (p=0.030) | `forty`: -0.196 (p=0.143) |
| WR (Break) | `target_rate_break` | 62 | 18,382 | `max_a`: +0.097 (p=0.451) | `max_a`: +0.126 (p=0.329) | `ngs_athleticism_score`: -0.085 (p=0.512) |
| DL (All) | `mean_player_get_off` | 79 | 28,970 | `max_s`: -0.668** (p=0.000) | `max_s`: -0.682** (p=0.000) | `forty`: +0.776** (p=0.000) |
| DL (All) | `mean_time_to_pressure` | 78 | 2,704 | `max_s`: -0.235* (p=0.039) | `max_s`: -0.181 (p=0.112) | `ngs_athleticism_score`: -0.186 (p=0.104) |
| DL (All) | `quick_pressure_rate` | 79 | 28,970 | `max_s`: +0.391** (p=0.000) | `max_s`: +0.416** (p=0.000) | `forty`: -0.491** (p=0.000) |

