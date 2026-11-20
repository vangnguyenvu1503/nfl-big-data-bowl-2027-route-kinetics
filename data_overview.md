# NFL Big Data Bowl 2027 — Data Audit Overview

- **Thời gian tạo báo cáo**: `2026-11-19 22:55:31`
- **Thư mục dữ liệu**: `D:\nfl-bdb-2027\data\raw`
- **Xử lý giá trị thiếu**: Đã cấu hình `null_values=["NA", ""]` trong `polars.scan_csv` để nhận diện chính xác chuỗi `"NA"` và chuỗi rỗng là missing values (giữ nguyên các nhãn hợp lệ như `"NONE"`, `"NO_ASSIGNMENT"`).

## 1. Tổng quan các file dữ liệu

| File | Size (MB) | Rows | Columns |
| :--- | ---: | ---: | ---: |
| `combine_results.csv` | 0.039 | 510 | 18 |
| `combine_tracking.csv` | 79.907 | 463,189 | 14 |
| `game_tracking_2023.csv` | 319.923 | 3,665,443 | 12 |
| `game_tracking_2024.csv` | 684.693 | 7,827,944 | 12 |
| `game_tracking_2025.csv` | 975.204 | 11,160,903 | 12 |
| `games.csv` | 0.041 | 1,002 | 7 |
| `player_career_successes.csv` | 0.013 | 510 | 9 |
| `player_play.csv` | 140.075 | 314,197 | 64 |
| `players.csv` | 0.045 | 510 | 10 |
| **Tổng cộng (9 files)** | **2199.940** | **23,434,208** | — |

## 2. Schema và tỷ lệ Missing Values từng file

### `combine_results.csv` (0.039 MB | 510 dòng | 18 cột)

| # | Column | Dtype | Missing Count | Missing (%) |
| ---: | :--- | :--- | ---: | ---: |
| 1 | `draft_year` | `Int64` | 0 | 0.00% |
| 2 | `nfl_id` | `Int64` | 0 | 0.00% |
| 3 | `combine_position` | `String` | 0 | 0.00% |
| 4 | `combine_height` | `Float64` | 0 | 0.00% |
| 5 | `combine_weight` | `Int64` | 0 | 0.00% |
| 6 | `hand_size` | `Float64` | 0 | 0.00% |
| 7 | `arm_length` | `Float64` | 0 | 0.00% |
| 8 | `wing_span` | `Float64` | 0 | 0.00% |
| 9 | `ten_yd_split` | `Float64` | 85 | 16.67% |
| 10 | `forty` | `Float64` | 86 | 16.86% |
| 11 | `vertical` | `Float64` | 72 | 14.12% |
| 12 | `broad_jump` | `Int64` | 87 | 17.06% |
| 13 | `three_cone` | `Float64` | 330 | 64.71% |
| 14 | `short_shuttle` | `Float64` | 309 | 60.59% |
| 15 | `bench_reps` | `Int64` | 318 | 62.35% |
| 16 | `ngs_athleticism_score` | `Int64` | 0 | 0.00% |
| 17 | `ngs_college_production_score` | `Int64` | 1 | 0.20% |
| 18 | `ngs_final_score` | `Int64` | 1 | 0.20% |

### `combine_tracking.csv` (79.907 MB | 463,189 dòng | 14 cột)

| # | Column | Dtype | Missing Count | Missing (%) |
| ---: | :--- | :--- | ---: | ---: |
| 1 | `draft_year` | `Int64` | 0 | 0.00% |
| 2 | `event_id` | `String` | 0 | 0.00% |
| 3 | `nfl_id` | `Int64` | 0 | 0.00% |
| 4 | `entity_type` | `String` | 0 | 0.00% |
| 5 | `time` | `String` | 0 | 0.00% |
| 6 | `drill_type` | `String` | 0 | 0.00% |
| 7 | `drill_name` | `String` | 0 | 0.00% |
| 8 | `attempt` | `Int64` | 0 | 0.00% |
| 9 | `x` | `Float64` | 0 | 0.00% |
| 10 | `y` | `Float64` | 0 | 0.00% |
| 11 | `s` | `Float64` | 0 | 0.00% |
| 12 | `a` | `Float64` | 0 | 0.00% |
| 13 | `dis` | `Float64` | 0 | 0.00% |
| 14 | `dir` | `Float64` | 0 | 0.00% |

### `game_tracking_2023.csv` (319.923 MB | 3,665,443 dòng | 12 cột)

| # | Column | Dtype | Missing Count | Missing (%) |
| ---: | :--- | :--- | ---: | ---: |
| 1 | `game_id` | `Int64` | 0 | 0.00% |
| 2 | `play_id` | `Int64` | 0 | 0.00% |
| 3 | `nfl_id` | `Int64` | 0 | 0.00% |
| 4 | `time` | `String` | 0 | 0.00% |
| 5 | `x` | `Float64` | 0 | 0.00% |
| 6 | `y` | `Float64` | 0 | 0.00% |
| 7 | `s` | `Float64` | 0 | 0.00% |
| 8 | `a` | `Float64` | 0 | 0.00% |
| 9 | `dis` | `Float64` | 0 | 0.00% |
| 10 | `o` | `Float64` | 0 | 0.00% |
| 11 | `dir` | `Float64` | 0 | 0.00% |
| 12 | `event` | `String` | 3,422,066 | 93.36% |

### `game_tracking_2024.csv` (684.693 MB | 7,827,944 dòng | 12 cột)

| # | Column | Dtype | Missing Count | Missing (%) |
| ---: | :--- | :--- | ---: | ---: |
| 1 | `game_id` | `Int64` | 0 | 0.00% |
| 2 | `play_id` | `Int64` | 0 | 0.00% |
| 3 | `nfl_id` | `Int64` | 0 | 0.00% |
| 4 | `time` | `String` | 0 | 0.00% |
| 5 | `x` | `Float64` | 0 | 0.00% |
| 6 | `y` | `Float64` | 0 | 0.00% |
| 7 | `s` | `Float64` | 0 | 0.00% |
| 8 | `a` | `Float64` | 0 | 0.00% |
| 9 | `dis` | `Float64` | 0 | 0.00% |
| 10 | `o` | `Float64` | 0 | 0.00% |
| 11 | `dir` | `Float64` | 0 | 0.00% |
| 12 | `event` | `String` | 7,306,249 | 93.34% |

### `game_tracking_2025.csv` (975.204 MB | 11,160,903 dòng | 12 cột)

| # | Column | Dtype | Missing Count | Missing (%) |
| ---: | :--- | :--- | ---: | ---: |
| 1 | `game_id` | `Int64` | 0 | 0.00% |
| 2 | `play_id` | `Int64` | 0 | 0.00% |
| 3 | `nfl_id` | `Int64` | 0 | 0.00% |
| 4 | `time` | `String` | 0 | 0.00% |
| 5 | `x` | `Float64` | 0 | 0.00% |
| 6 | `y` | `Float64` | 0 | 0.00% |
| 7 | `s` | `Float64` | 0 | 0.00% |
| 8 | `a` | `Float64` | 0 | 0.00% |
| 9 | `dis` | `Float64` | 0 | 0.00% |
| 10 | `o` | `Float64` | 0 | 0.00% |
| 11 | `dir` | `Float64` | 0 | 0.00% |
| 12 | `event` | `String` | 10,425,238 | 93.41% |

### `games.csv` (0.041 MB | 1,002 dòng | 7 cột)

| # | Column | Dtype | Missing Count | Missing (%) |
| ---: | :--- | :--- | ---: | ---: |
| 1 | `game_id` | `Int64` | 0 | 0.00% |
| 2 | `game_key` | `Int64` | 0 | 0.00% |
| 3 | `season` | `Int64` | 0 | 0.00% |
| 4 | `season_type` | `String` | 0 | 0.00% |
| 5 | `week` | `Int64` | 0 | 0.00% |
| 6 | `home_team_abbr` | `String` | 0 | 0.00% |
| 7 | `visitor_team_abbr` | `String` | 0 | 0.00% |

### `player_career_successes.csv` (0.013 MB | 510 dòng | 9 cột)

| # | Column | Dtype | Missing Count | Missing (%) |
| ---: | :--- | :--- | ---: | ---: |
| 1 | `nfl_id` | `Int64` | 0 | 0.00% |
| 2 | `career_offensive_snaps` | `Int64` | 0 | 0.00% |
| 3 | `career_defensive_snaps` | `Int64` | 0 | 0.00% |
| 4 | `career_special_teams_snaps` | `Int64` | 0 | 0.00% |
| 5 | `career_games_active` | `Int64` | 0 | 0.00% |
| 6 | `career_games_started` | `Int64` | 0 | 0.00% |
| 7 | `ap_all_pro_1st_team` | `Int64` | 0 | 0.00% |
| 8 | `ap_all_pro_2nd_team` | `Int64` | 0 | 0.00% |
| 9 | `pro_bowl_original_ballot` | `Int64` | 0 | 0.00% |

### `player_play.csv` (140.075 MB | 314,197 dòng | 64 cột)

| # | Column | Dtype | Missing Count | Missing (%) |
| ---: | :--- | :--- | ---: | ---: |
| 1 | `game_id` | `Int64` | 0 | 0.00% |
| 2 | `play_id` | `Int64` | 0 | 0.00% |
| 3 | `team_abbr` | `String` | 0 | 0.00% |
| 4 | `nfl_id` | `Int64` | 0 | 0.00% |
| 5 | `lined_up_position` | `String` | 1,383 | 0.44% |
| 6 | `play_description` | `String` | 0 | 0.00% |
| 7 | `quarter` | `Int64` | 0 | 0.00% |
| 8 | `down` | `Int64` | 0 | 0.00% |
| 9 | `yards_to_go` | `Int64` | 0 | 0.00% |
| 10 | `possession_team` | `String` | 0 | 0.00% |
| 11 | `yardline_side` | `String` | 4,102 | 1.31% |
| 12 | `yardline_number` | `Int64` | 0 | 0.00% |
| 13 | `game_clock` | `String` | 0 | 0.00% |
| 14 | `pre_snap_home_score` | `Int64` | 0 | 0.00% |
| 15 | `pre_snap_visitor_score` | `Int64` | 0 | 0.00% |
| 16 | `play_direction` | `String` | 50 | 0.02% |
| 17 | `pre_snap_home_team_win_probability` | `Float64` | 3 | 0.00% |
| 18 | `pre_snap_visitor_team_win_probability` | `Float64` | 3 | 0.00% |
| 19 | `home_team_win_probability_added` | `Float64` | 6 | 0.00% |
| 20 | `visitor_team_win_probility_added` | `Float64` | 6 | 0.00% |
| 21 | `expected_points_added` | `Float64` | 0 | 0.00% |
| 22 | `expected_points` | `Float64` | 0 | 0.00% |
| 23 | `offense_formation` | `String` | 5,335 | 1.70% |
| 24 | `receiver_alignment` | `String` | 4,597 | 1.46% |
| 25 | `pass_result` | `String` | 103,636 | 32.98% |
| 26 | `pass_length` | `Int64` | 129,458 | 41.20% |
| 27 | `play_nullified_by_penalty` | `String` | 0 | 0.00% |
| 28 | `penalty_yards` | `Int64` | 293,447 | 93.40% |
| 29 | `pre_penalty_yards_gained` | `Int64` | 0 | 0.00% |
| 30 | `yards_gained` | `Int64` | 0 | 0.00% |
| 31 | `team_coverage_man_zone` | `String` | 38,919 | 12.39% |
| 32 | `team_coverage_type` | `String` | 38,919 | 12.39% |
| 33 | `in_motion_at_ball_snap` | `Boolean` | 158,231 | 50.36% |
| 34 | `target` | `Boolean` | 0 | 0.00% |
| 35 | `rec_yards` | `Int64` | 307,937 | 98.01% |
| 36 | `rush_yards` | `Int64` | 314,175 | 99.99% |
| 37 | `yards_after_catch` | `Int64` | 307,937 | 98.01% |
| 38 | `fumble` | `Int64` | 314,124 | 99.98% |
| 39 | `fumble_lost` | `Int64` | 314,124 | 99.98% |
| 40 | `route_ran` | `String` | 260,889 | 83.03% |
| 41 | `separation_at_pass_forward` | `Float64` | 267,460 | 85.12% |
| 42 | `cushion` | `Float64` | 210,445 | 66.98% |
| 43 | `expected_yards_after_catch` | `Float64` | 305,221 | 97.14% |
| 44 | `dropback_duration` | `Float64` | 253,394 | 80.65% |
| 45 | `extended_sack_allowed` | `Float64` | 253,394 | 80.65% |
| 46 | `pass_rushers_encountered` | `Int64` | 253,394 | 80.65% |
| 47 | `peak_pressure_probability_allowed` | `Float64` | 256,169 | 81.53% |
| 48 | `pressure_allowed` | `Boolean` | 253,394 | 80.65% |
| 49 | `sack_allowed` | `Float64` | 253,394 | 80.65% |
| 50 | `time_to_pressure_allowed` | `Float64` | 309,227 | 98.42% |
| 51 | `quick_pressure` | `Boolean` | 310,434 | 98.80% |
| 52 | `unblocked_pressure` | `Boolean` | 227,359 | 72.36% |
| 53 | `blitzing` | `Boolean` | 113,470 | 36.11% |
| 54 | `player_get_off` | `Float64` | 270,195 | 86.00% |
| 55 | `sack` | `Float64` | 227,362 | 72.36% |
| 56 | `tackle` | `Int64` | 302,301 | 96.21% |
| 57 | `assist` | `Int64` | 302,301 | 96.21% |
| 58 | `caused_forced_fumble` | `Int64` | 302,301 | 96.21% |
| 59 | `recovered_fumble` | `Boolean` | 311,983 | 99.30% |
| 60 | `tackle_for_loss` | `Boolean` | 302,301 | 96.21% |
| 61 | `time_to_pressure` | `Float64` | 310,393 | 98.79% |
| 62 | `time_to_qb_hurry` | `Float64` | 310,318 | 98.77% |
| 63 | `coverage_assignment` | `String` | 241,545 | 76.88% |
| 64 | `coverage_assignment_at_snap` | `String` | 242,072 | 77.04% |

### `players.csv` (0.045 MB | 510 dòng | 10 cột)

| # | Column | Dtype | Missing Count | Missing (%) |
| ---: | :--- | :--- | ---: | ---: |
| 1 | `nfl_id` | `Int64` | 0 | 0.00% |
| 2 | `display_name` | `String` | 0 | 0.00% |
| 3 | `draft_year` | `Int64` | 0 | 0.00% |
| 4 | `nfl_position` | `String` | 0 | 0.00% |
| 5 | `birth_date` | `String` | 0 | 0.00% |
| 6 | `college_name` | `String` | 0 | 0.00% |
| 7 | `college_conference` | `String` | 0 | 0.00% |
| 8 | `draft_round` | `Int64` | 127 | 24.90% |
| 9 | `draft_pick_within_round` | `Int64` | 127 | 24.90% |
| 10 | `draft_overall_pick` | `Int64` | 127 | 24.90% |

## 3. Giá trị duy nhất của các cột phân loại & khóa liên kết

### `players.csv`

- Khóa `nfl_id`: **510** unique non-null / **510** dòng (null: 0).

#### Cột `draft_year` — 3 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| 2023 | 160 | 31.37% |
| 2024 | 177 | 34.71% |
| 2025 | 173 | 33.92% |

#### Cột `nfl_position` — 17 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| WR | 106 | 20.78% |
| CB | 60 | 11.76% |
| G | 51 | 10.00% |
| T | 50 | 9.80% |
| DT | 47 | 9.22% |
| TE | 42 | 8.24% |
| DE | 36 | 7.06% |
| OLB | 30 | 5.88% |
| SS | 28 | 5.49% |
| FS | 25 | 4.90% |
| C | 20 | 3.92% |
| DB | 5 | 0.98% |
| NT | 5 | 0.98% |
| ILB | 2 | 0.39% |
| FB | 1 | 0.20% |
| MLB | 1 | 0.20% |
| RB | 1 | 0.20% |

#### Cột `draft_round` — 7 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| 1 | 38 | 7.45% |
| 2 | 59 | 11.57% |
| 3 | 59 | 11.57% |
| 4 | 59 | 11.57% |
| 5 | 62 | 12.16% |
| 6 | 50 | 9.80% |
| 7 | 56 | 10.98% |
| `<NULL>` | 127 | 24.90% |

#### Cột `college_conference` — 21 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| Southeastern Conference | 142 | 27.84% |
| Big Ten Conference | 96 | 18.82% |
| Atlantic Coast Conference | 81 | 15.88% |
| Big Twelve Conference | 71 | 13.92% |
| Pacific Twelve Conference | 44 | 8.63% |
| American Athletic Conference | 14 | 2.75% |
| Independent | 14 | 2.75% |
| Mountain West Conference | 10 | 1.96% |
| Mid-American Conference | 8 | 1.57% |
| Conference USA | 6 | 1.18% |
| Missouri Valley Football Conference | 6 | 1.18% |
| Sun Belt Conference | 6 | 1.18% |
| Southwestern Athletic Conference | 3 | 0.59% |
| Patriot League | 2 | 0.39% |
| Big South - Ohio Valley Conference | 1 | 0.20% |
| Great Lakes Intercollegiate Athletic Conference | 1 | 0.20% |
| Ivy League | 1 | 0.20% |
| Mid-Eastern Athletic Conference | 1 | 0.20% |
| Southern Conference | 1 | 0.20% |
| Southland Conference | 1 | 0.20% |
| United Athletic Conference | 1 | 0.20% |

#### Cột `college_name` — 112 giá trị duy nhất (non-null) — hiển thị top 20

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| Georgia | 22 | 4.31% |
| Alabama | 16 | 3.14% |
| Louisiana State | 15 | 2.94% |
| Ohio State | 14 | 2.75% |
| South Carolina | 14 | 2.75% |
| Texas | 14 | 2.75% |
| Michigan | 12 | 2.35% |
| Arkansas | 11 | 2.16% |
| Florida | 11 | 2.16% |
| Florida State | 11 | 2.16% |
| Maryland | 11 | 2.16% |
| Miami | 11 | 2.16% |
| Mississippi | 11 | 2.16% |
| Oregon | 11 | 2.16% |
| Texas Christian | 10 | 1.96% |
| Iowa | 9 | 1.76% |
| Notre Dame | 9 | 1.76% |
| Penn State | 9 | 1.76% |
| Southern California | 9 | 1.76% |
| Iowa State | 8 | 1.57% |

### `combine_results.csv`

- Khóa `nfl_id`: **510** unique non-null / **510** dòng (null: 0).

#### Cột `draft_year` — 3 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| 2023 | 160 | 31.37% |
| 2024 | 177 | 34.71% |
| 2025 | 173 | 33.92% |

#### Cột `combine_position` — 5 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| DB | 122 | 23.92% |
| OL | 121 | 23.73% |
| DL | 118 | 23.14% |
| WR | 107 | 20.98% |
| TE | 42 | 8.24% |

### `combine_tracking.csv`

- Tổng số dòng: **463,189**
- Unique `nfl_id` (toàn bảng): **510**
- Unique `nfl_id` (`entity_type == "PLAYER"`): **510**
- Unique `nfl_id` (`entity_type == "BALL"`): **54**
- Unique `event_id`: **6,310**

#### Cột `draft_year` — 3 giá trị duy nhất

| draft_year | Rows | Unique nfl_id | Unique nfl_id (PLAYER) |
| :--- | ---: | ---: | ---: |
| 2023 | 136,097 | 160 | 160 |
| 2024 | 147,815 | 177 | 177 |
| 2025 | 179,277 | 173 | 173 |

#### Cột `entity_type` — 2 giá trị duy nhất

| entity_type | Rows | Unique nfl_id | Unique nfl_id (PLAYER) |
| :--- | ---: | ---: | ---: |
| PLAYER | 431,094 | 510 | 510 |
| BALL | 32,095 | 54 | 0 |

#### Cột `drill_type` — 9 giá trị duy nhất

| drill_type | Rows | Unique nfl_id | Unique nfl_id (PLAYER) |
| :--- | ---: | ---: | ---: |
| FORTY_YARD_DASH | 41,500 | 418 | 418 |
| SHORT_SHUTTLE | 11,690 | 170 | 170 |
| THREE_CONE_DRILL | 15,951 | 160 | 160 |
| SKILL_DRILLS_DB | 94,623 | 122 | 122 |
| SKILL_DRILLS_OL | 61,129 | 121 | 121 |
| SKILL_DRILLS_DL | 77,428 | 114 | 114 |
| SKILL_DRILLS_WR | 120,383 | 107 | 107 |
| SKILL_DRILLS_TE | 37,222 | 42 | 42 |
| SKILL_DRILLS_LB | 3,263 | 4 | 4 |

#### Cột `drill_name` — 56 giá trị duy nhất

| drill_name | Rows | Unique nfl_id | Unique nfl_id (PLAYER) |
| :--- | ---: | ---: | ---: |
| FORTY_YARD_DASH | 41,500 | 418 | 418 |
| GAUNTLET_DRILL | 38,228 | 267 | 267 |
| SHORT_SHUTTLE | 11,690 | 170 | 170 |
| THREE_CONE_DRILL | 15,951 | 160 | 160 |
| OVER_SHOULDER_ADJUST | 12,149 | 130 | 130 |
| BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION | 21,569 | 122 | 122 |
| BOX_DRILL | 9,439 | 122 | 122 |
| BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL | 8,422 | 122 | 122 |
| W_DRILL | 10,214 | 121 | 121 |
| FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE | 18,087 | 120 | 120 |
| PASS_RUSH_DROP | 10,752 | 120 | 120 |
| SCREEN_DRILL | 8,031 | 119 | 119 |
| OL_PULL_DRILL_FOLD_BLOCK_RIGHT | 4,706 | 119 | 119 |
| PASS_RUSH_DRILL | 10,285 | 118 | 118 |
| FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION | 12,845 | 112 | 112 |
| FOUR_BAG_AGILITY_DRILL | 11,709 | 112 | 112 |
| RUN_AND_CLUB_DRILL | 8,180 | 112 | 112 |
| TERYL_AUSTIN_DRILL_2 | 9,514 | 111 | 111 |
| BODY_CONTROL_DRILL | 6,611 | 111 | 111 |
| RUN_THE_HOOP_DRILL | 9,120 | 109 | 109 |
| DAGGER_ROUTE_LEFT | 9,917 | 107 | 107 |
| SLANT_ROUTE_LEFT | 5,320 | 107 | 107 |
| CURL_ROUTE_RIGHT | 7,993 | 103 | 103 |
| SLOT_STRIKE_ROUTE_LEFT | 8,457 | 101 | 101 |
| TERYL_AUSTIN_DRILL_1 | 7,765 | 101 | 101 |
| GO_ROUTE_RIGHT | 9,843 | 98 | 98 |
| SLOT_SAIL_ROUTE_RIGHT | 9,524 | 98 | 98 |
| COMEBACK_ROUTE_RIGHT | 9,266 | 98 | 98 |
| RED_ZONE_FADE_RIGHT | 6,256 | 98 | 98 |
| OL_PULL_DRILL_DEEP_SHORT_PULL_LEFT | 4,412 | 98 | 98 |
| POST_CORNER_ROUTE_RIGHT | 10,517 | 96 | 96 |
| OL_PULL_DRILL_LONG_PULL_RIGHT | 4,720 | 93 | 93 |
| LINE_DRILL | 11,895 | 85 | 85 |
| PASS_PRO-MIRROR_DRILL | 7,020 | 84 | 84 |
| SPEED_OUT_ROUTE_LEFT | 5,619 | 76 | 76 |
| POST_CORNER_ROUTE_LEFT | 4,289 | 70 | 70 |
| BACK_PEDAL_AND_REACT | 6,505 | 59 | 59 |
| FLAT_ROUTE_LEFT | 3,066 | 42 | 42 |
| HOOK_ROUTE_RIGHT | 3,986 | 41 | 41 |
| SHORT_ZONE_BREAKS | 9,044 | 40 | 40 |
| IN_ROUTE_LEFT | 3,842 | 40 | 40 |
| WHEEL_ROUTE_LEFT | 3,323 | 40 | 40 |
| BLOCK_EXPLOSION | 2,330 | 39 | 39 |
| CIRCUS_ROUTE_LEFT | 4,739 | 37 | 37 |
| PASS_PRO_MIRROR_DRILL | 3,401 | 37 | 37 |
| LINE | 3,552 | 33 | 33 |
| SPEED_OUT_LEFT | 1,675 | 31 | 31 |
| CORNER_ROUTE | 6,467 | 30 | 30 |
| END_ZONE_FADE_RIGHT | 2,290 | 28 | 28 |
| SHORT_ZONE_BREAK_FLAT_WHEEL | 2,374 | 20 | 20 |
| SHORT_ZONE_BREAK_FORWARD | 1,256 | 20 | 20 |
| SHORT_ZONE_BREAK_INSIDE | 1,534 | 18 | 18 |
| OVER_THE_SHOULDER_ADJUST | 762 | 12 | 12 |
| WAVE_DRILL | 477 | 4 | 4 |
| SHUFFLE_SPRINT_AND_COD_DRILL | 386 | 4 | 4 |
| FOUR_BAG_SHUFFLE_DRILL | 365 | 3 | 3 |

#### Cột `attempt` — 17 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| 1 | 247,474 | 53.43% |
| 2 | 75,934 | 16.39% |
| 3 | 23,012 | 4.97% |
| 4 | 13,486 | 2.91% |
| 5 | 14,046 | 3.03% |
| 6 | 14,237 | 3.07% |
| 7 | 14,697 | 3.17% |
| 8 | 12,616 | 2.72% |
| 9 | 10,527 | 2.27% |
| 10 | 11,056 | 2.39% |
| 11 | 8,639 | 1.87% |
| 12 | 6,335 | 1.37% |
| 13 | 4,980 | 1.08% |
| 14 | 3,572 | 0.77% |
| 15 | 1,836 | 0.40% |
| 16 | 668 | 0.14% |
| 17 | 74 | 0.02% |

### `games.csv`

- Unique `game_id`: **1,002** / 1,002 dòng; Unique `game_key`: **1,002**.

#### Cột `season` — 3 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| 2023 | 334 | 33.33% |
| 2024 | 334 | 33.33% |
| 2025 | 334 | 33.33% |

#### Cột `season_type` — 3 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| REG | 816 | 81.44% |
| PRE | 147 | 14.67% |
| POST | 39 | 3.89% |

#### Phân bố `season` × `season_type`

| season | season_type | Games | Week Range |
| ---: | :--- | ---: | :---: |
| 2023 | POST | 13 | 19..23 |
| 2023 | PRE | 49 | 0..3 |
| 2023 | REG | 272 | 1..18 |
| 2024 | POST | 13 | 19..23 |
| 2024 | PRE | 49 | 0..3 |
| 2024 | REG | 272 | 1..18 |
| 2025 | POST | 13 | 19..23 |
| 2025 | PRE | 49 | 0..3 |
| 2025 | REG | 272 | 1..18 |

#### Cột `week` — 23 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| 0 | 3 | 0.30% |
| 1 | 96 | 9.58% |
| 2 | 96 | 9.58% |
| 3 | 96 | 9.58% |
| 4 | 48 | 4.79% |
| 5 | 42 | 4.19% |
| 6 | 44 | 4.39% |
| 7 | 43 | 4.29% |
| 8 | 45 | 4.49% |
| 9 | 43 | 4.29% |
| 10 | 42 | 4.19% |
| 11 | 43 | 4.29% |
| 12 | 43 | 4.29% |
| 13 | 45 | 4.49% |
| 14 | 42 | 4.19% |
| 15 | 48 | 4.79% |
| 16 | 48 | 4.79% |
| 17 | 48 | 4.79% |
| 18 | 48 | 4.79% |
| 19 | 18 | 1.80% |
| 20 | 12 | 1.20% |
| 21 | 6 | 0.60% |
| 23 | 3 | 0.30% |

#### Các đội bóng (`home_team_abbr` & `visitor_team_abbr`)

- `home_team_abbr` (33 đội): `ARI`, `ATL`, `BAL`, `BUF`, `CAR`, `CHI`, `CIN`, `CLE`, `DAL`, `DEN`, `DET`, `GB`, `HOU`, `IND`, `JAX`, `KC`, `LA`, `LAC`, `LAR`, `LV`, `MIA`, `MIN`, `NE`, `NO`, `NYG`, `NYJ`, `PHI`, `PIT`, `SEA`, `SF`, `TB`, `TEN`, `WAS`
- `visitor_team_abbr` (33 đội): `ARI`, `ATL`, `BAL`, `BUF`, `CAR`, `CHI`, `CIN`, `CLE`, `DAL`, `DEN`, `DET`, `GB`, `HOU`, `IND`, `JAX`, `KC`, `LA`, `LAC`, `LAR`, `LV`, `MIA`, `MIN`, `NE`, `NO`, `NYG`, `NYJ`, `PHI`, `PIT`, `SEA`, `SF`, `TB`, `TEN`, `WAS`

### `player_career_successes.csv`

- Khóa `nfl_id`: **510** unique non-null / **510** dòng (null: 0).

#### Cột `ap_all_pro_1st_team` — 2 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| 0 | 504 | 98.82% |
| 1 | 6 | 1.18% |

#### Cột `ap_all_pro_2nd_team` — 2 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| 0 | 508 | 99.61% |
| 1 | 2 | 0.39% |

#### Cột `pro_bowl_original_ballot` — 3 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| 0 | 499 | 97.84% |
| 1 | 8 | 1.57% |
| 2 | 3 | 0.59% |

### `player_play.csv`

- Tổng số dòng: **314,197**
- Unique `nfl_id`: **497**
- Unique `game_id`: **1,001**
- Unique `(game_id, play_id)`: **114,954**

#### Độ phủ theo mùa giải (join `game_id` với `games.csv`)

| season | season_type | Rows | Unique game_id | Unique nfl_id |
| ---: | :--- | ---: | ---: | ---: |
| 2023 | POST | 1,560 | 13 | 31 |
| 2023 | PRE | 10,407 | 49 | 150 |
| 2023 | REG | 38,788 | 271 | 122 |
| 2024 | POST | 4,351 | 13 | 77 |
| 2024 | PRE | 18,611 | 49 | 296 |
| 2024 | REG | 85,562 | 272 | 247 |
| 2025 | POST | 7,258 | 13 | 115 |
| 2025 | PRE | 22,741 | 49 | 420 |
| 2025 | REG | 124,919 | 272 | 353 |

#### Cột `lined_up_position` — 17 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| G | 48,507 | 15.44% |
| T | 42,851 | 13.64% |
| EDGE | 38,398 | 12.22% |
| CB | 35,017 | 11.14% |
| INTERIOR_LINE | 30,301 | 9.64% |
| WR | 28,077 | 8.94% |
| HIGH_SAFETY | 25,608 | 8.15% |
| SLOT_WR | 18,755 | 5.97% |
| C | 14,959 | 4.76% |
| SLOT_CB | 14,537 | 4.63% |
| TE | 6,803 | 2.17% |
| OLB | 6,362 | 2.02% |
| MLB | 1,824 | 0.58% |
| `<NULL>` | 1,383 | 0.44% |
| RB | 320 | 0.10% |
| BOX_SAFETY | 266 | 0.08% |
| FB | 228 | 0.07% |
| EXTRA_OL | 1 | 0.00% |

#### Cột `team_abbr` — 33 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| HOU | 17,362 | 5.53% |
| ARI | 14,014 | 4.46% |
| PIT | 12,663 | 4.03% |
| CHI | 12,642 | 4.02% |
| NE | 11,862 | 3.78% |
| TEN | 11,798 | 3.75% |
| LAC | 11,164 | 3.55% |
| NYG | 10,829 | 3.45% |
| GB | 10,499 | 3.34% |
| TB | 10,447 | 3.32% |
| IND | 10,310 | 3.28% |
| DAL | 10,292 | 3.28% |
| JAX | 10,180 | 3.24% |
| SEA | 10,105 | 3.22% |
| SF | 9,873 | 3.14% |
| BUF | 9,859 | 3.14% |
| LA | 9,452 | 3.01% |
| PHI | 9,112 | 2.90% |
| CLE | 8,721 | 2.78% |
| LV | 8,669 | 2.76% |
| WAS | 8,462 | 2.69% |
| DET | 8,333 | 2.65% |
| KC | 8,333 | 2.65% |
| CIN | 8,161 | 2.60% |
| NO | 8,004 | 2.55% |
| ATL | 7,728 | 2.46% |
| CAR | 7,171 | 2.28% |
| MIA | 7,035 | 2.24% |
| LAR | 6,965 | 2.22% |
| NYJ | 6,583 | 2.10% |
| MIN | 5,895 | 1.88% |
| DEN | 5,881 | 1.87% |
| BAL | 5,793 | 1.84% |

#### Cột `play_direction` — 2 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| left | 157,872 | 50.25% |
| right | 156,275 | 49.74% |
| `<NULL>` | 50 | 0.02% |

#### Cột `play_nullified_by_penalty` — 2 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| N | 301,707 | 96.02% |
| Y | 12,490 | 3.98% |

#### Cột `offense_formation` — 7 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| SHOTGUN | 182,052 | 57.94% |
| SINGLEBACK | 69,220 | 22.03% |
| EMPTY | 22,754 | 7.24% |
| I_FORM | 16,406 | 5.22% |
| PISTOL | 14,880 | 4.74% |
| `<NULL>` | 5,335 | 1.70% |
| JUMBO | 2,708 | 0.86% |
| WILDCAT | 842 | 0.27% |

#### Cột `receiver_alignment` — 15 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| 2x2 | 130,055 | 41.39% |
| 3x1 | 126,352 | 40.21% |
| 2x1 | 27,989 | 8.91% |
| 3x2 | 21,103 | 6.72% |
| `<NULL>` | 4,597 | 1.46% |
| 4x1 | 1,974 | 0.63% |
| 1x1 | 1,100 | 0.35% |
| 0x0 | 643 | 0.20% |
| 3x0 | 173 | 0.06% |
| 4x0 | 136 | 0.04% |
| 2x0 | 46 | 0.01% |
| 3x3 | 14 | 0.00% |
| 5x2 | 6 | 0.00% |
| 4x2 | 4 | 0.00% |
| 4x3 | 4 | 0.00% |
| 1x0 | 1 | 0.00% |

#### Cột `pass_result` — 5 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| C | 115,903 | 36.89% |
| `<NULL>` | 103,636 | 32.98% |
| I | 65,187 | 20.75% |
| S | 13,888 | 4.42% |
| R | 11,326 | 3.60% |
| IN | 4,257 | 1.35% |

#### Cột `team_coverage_man_zone` — 2 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| ZONE_COVERAGE | 197,851 | 62.97% |
| MAN_COVERAGE | 77,427 | 24.64% |
| `<NULL>` | 38,919 | 12.39% |

#### Cột `team_coverage_type` — 8 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| COVER_3_ZONE | 97,938 | 31.17% |
| COVER_1_MAN | 56,381 | 17.94% |
| COVER_4_ZONE | 43,685 | 13.90% |
| `<NULL>` | 38,919 | 12.39% |
| COVER_2_ZONE | 32,999 | 10.50% |
| COVER_6_ZONE | 22,596 | 7.19% |
| COVER_0_MAN | 18,179 | 5.79% |
| COVER_2_MAN | 2,867 | 0.91% |
| PREVENT | 633 | 0.20% |

#### Cột `route_ran` — 12 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| `<NULL>` | 260,889 | 83.03% |
| GO | 9,715 | 3.09% |
| HITCH | 8,816 | 2.81% |
| CROSS | 6,756 | 2.15% |
| IN | 5,582 | 1.78% |
| OUT | 5,179 | 1.65% |
| POST | 4,865 | 1.55% |
| CORNER | 4,232 | 1.35% |
| SLANT | 3,413 | 1.09% |
| FLAT | 3,287 | 1.05% |
| SCREEN | 1,321 | 0.42% |
| WHEEL | 83 | 0.03% |
| ANGLE | 59 | 0.02% |

#### Cột `coverage_assignment` — 20 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| `<NULL>` | 241,545 | 76.88% |
| NO_ASSIGNMENT | 29,442 | 9.37% |
| MAN | 10,745 | 3.42% |
| CURL_FLAT_LEFT | 2,988 | 0.95% |
| DEEP_THIRD_RIGHT | 2,859 | 0.91% |
| CURL_FLAT_RIGHT | 2,781 | 0.89% |
| DEEP_THIRD_MIDDLE | 2,655 | 0.85% |
| DEEP_THIRD_LEFT | 2,065 | 0.66% |
| QUARTERS_OUTSIDE_RIGHT | 2,042 | 0.65% |
| QUARTERS_INSIDE_LEFT | 2,036 | 0.65% |
| QUARTERS_INSIDE_RIGHT | 1,771 | 0.56% |
| FLAT_RIGHT | 1,718 | 0.55% |
| DEEP_HALF_LEFT | 1,692 | 0.54% |
| DEEP_HALF_RIGHT | 1,481 | 0.47% |
| DEEP_FREE | 1,469 | 0.47% |
| HOOK_CURL_LEFT | 1,403 | 0.45% |
| FLAT_LEFT | 1,361 | 0.43% |
| HOOK_CURL_RIGHT | 1,358 | 0.43% |
| QUARTERS_OUTSIDE_LEFT | 1,335 | 0.42% |
| HOLE | 1,274 | 0.41% |
| PREVENT | 177 | 0.06% |

#### Cột `coverage_assignment_at_snap` — 20 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| `<NULL>` | 242,072 | 77.04% |
| NO_ASSIGNMENT | 29,153 | 9.28% |
| MAN | 10,744 | 3.42% |
| CURL_FLAT_LEFT | 3,098 | 0.99% |
| CURL_FLAT_RIGHT | 2,929 | 0.93% |
| DEEP_THIRD_RIGHT | 2,857 | 0.91% |
| DEEP_THIRD_MIDDLE | 2,650 | 0.84% |
| QUARTERS_INSIDE_LEFT | 2,355 | 0.75% |
| QUARTERS_OUTSIDE_RIGHT | 2,301 | 0.73% |
| QUARTERS_INSIDE_RIGHT | 2,132 | 0.68% |
| DEEP_THIRD_LEFT | 2,011 | 0.64% |
| DEEP_HALF_LEFT | 1,563 | 0.50% |
| FLAT_RIGHT | 1,546 | 0.49% |
| QUARTERS_OUTSIDE_LEFT | 1,418 | 0.45% |
| DEEP_HALF_RIGHT | 1,340 | 0.43% |
| DEEP_FREE | 1,297 | 0.41% |
| FLAT_LEFT | 1,172 | 0.37% |
| HOOK_CURL_LEFT | 1,151 | 0.37% |
| HOOK_CURL_RIGHT | 1,151 | 0.37% |
| HOLE | 1,071 | 0.34% |
| PREVENT | 186 | 0.06% |

### `game_tracking_2023.csv`

- Tổng số dòng: **3,665,443**
- Unique non-null `nfl_id`: **156** (dòng `nfl_id` null: 0)
- Unique `game_id`: **333**
- Unique `(game_id, play_id)`: **31,690**

#### Độ phủ theo mùa giải (join `game_id` với `games.csv`)

| season | season_type | Unique game_id | Unique nfl_id |
| ---: | :--- | ---: | ---: |
| 2023 | POST | 13 | 31 |
| 2023 | PRE | 49 | 150 |
| 2023 | REG | 271 | 122 |

#### Cột `event` — 44 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| `<NULL>` | 3,422,066 | 93.36% |
| ball_snap | 50,813 | 1.39% |
| pass_forward | 31,422 | 0.86% |
| tackle | 29,011 | 0.79% |
| first_contact | 28,345 | 0.77% |
| pass_arrived | 23,967 | 0.65% |
| pass_outcome_caught | 18,845 | 0.51% |
| handoff | 14,053 | 0.38% |
| pass_outcome_incomplete | 10,862 | 0.30% |
| play_action | 8,274 | 0.23% |
| out_of_bounds | 6,082 | 0.17% |
| autoevent_passinterrupted | 4,237 | 0.12% |
| run | 2,899 | 0.08% |
| qb_sack | 2,104 | 0.06% |
| man_in_motion | 1,827 | 0.05% |
| line_set | 1,610 | 0.04% |
| play_submit | 1,264 | 0.03% |
| touchdown | 1,174 | 0.03% |
| dropped_pass | 807 | 0.02% |
| pass_outcome_interception | 754 | 0.02% |
| pass_outcome_touchdown | 646 | 0.02% |
| pass_tipped | 561 | 0.02% |
| fumble | 496 | 0.01% |
| qb_kneel | 493 | 0.01% |
| qb_slide | 419 | 0.01% |
| shift | 397 | 0.01% |
| fumble_offense_recovered | 316 | 0.01% |
| fumble_defense_recovered | 304 | 0.01% |
| qb_sneak | 293 | 0.01% |
| qb_strip_sack | 249 | 0.01% |
| lateral | 151 | 0.00% |
| pass_shovel | 134 | 0.00% |
| snap_direct | 108 | 0.00% |
| autoevent_playaction | 96 | 0.00% |
| two_point_conversion | 91 | 0.00% |
| qb_spike | 77 | 0.00% |
| penalty_flag | 52 | 0.00% |
| run_attempt | 41 | 0.00% |
| touchback | 39 | 0.00% |
| huddle_break_offense | 25 | 0.00% |
| huddle_start_offense | 15 | 0.00% |
| safety | 10 | 0.00% |
| penalty_accepted | 6 | 0.00% |
| penalty_declined | 6 | 0.00% |
| field_goal_attempt | 2 | 0.00% |

### `game_tracking_2024.csv`

- Tổng số dòng: **7,827,944**
- Unique non-null `nfl_id`: **318** (dòng `nfl_id` null: 0)
- Unique `game_id`: **334**
- Unique `(game_id, play_id)`: **40,331**

#### Độ phủ theo mùa giải (join `game_id` với `games.csv`)

| season | season_type | Unique game_id | Unique nfl_id |
| ---: | :--- | ---: | ---: |
| 2024 | POST | 13 | 75 |
| 2024 | PRE | 49 | 296 |
| 2024 | REG | 272 | 247 |

#### Cột `event` — 46 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| `<NULL>` | 7,306,249 | 93.34% |
| ball_snap | 108,626 | 1.39% |
| pass_forward | 64,826 | 0.83% |
| first_contact | 64,745 | 0.83% |
| tackle | 64,072 | 0.82% |
| pass_arrived | 48,877 | 0.62% |
| pass_outcome_caught | 39,397 | 0.50% |
| handoff | 31,977 | 0.41% |
| pass_outcome_incomplete | 22,030 | 0.28% |
| play_action | 17,841 | 0.23% |
| out_of_bounds | 12,225 | 0.16% |
| autoevent_passinterrupted | 9,302 | 0.12% |
| run | 6,177 | 0.08% |
| qb_sack | 4,386 | 0.06% |
| man_in_motion | 3,054 | 0.04% |
| line_set | 2,942 | 0.04% |
| play_submit | 2,648 | 0.03% |
| touchdown | 2,604 | 0.03% |
| autoevent_playaction | 2,348 | 0.03% |
| dropped_pass | 1,718 | 0.02% |
| pass_outcome_touchdown | 1,451 | 0.02% |
| pass_outcome_interception | 1,369 | 0.02% |
| fumble | 1,309 | 0.02% |
| pass_tipped | 1,106 | 0.01% |
| qb_kneel | 1,018 | 0.01% |
| fumble_offense_recovered | 787 | 0.01% |
| qb_slide | 783 | 0.01% |
| qb_sneak | 743 | 0.01% |
| fumble_defense_recovered | 727 | 0.01% |
| run_attempt | 659 | 0.01% |
| lateral | 494 | 0.01% |
| qb_strip_sack | 480 | 0.01% |
| pass_shovel | 289 | 0.00% |
| qb_spike | 180 | 0.00% |
| snap_direct | 122 | 0.00% |
| touchback | 110 | 0.00% |
| two_point_conversion | 103 | 0.00% |
| penalty_flag | 92 | 0.00% |
| huddle_break_offense | 22 | 0.00% |
| safety | 21 | 0.00% |
| huddle_start_offense | 13 | 0.00% |
| penalty_accepted | 13 | 0.00% |
| kick_recovered | 4 | 0.00% |
| penalty_declined | 2 | 0.00% |
| punt_fake | 1 | 0.00% |
| punt_land | 1 | 0.00% |
| two_point_play | 1 | 0.00% |

### `game_tracking_2025.csv`

- Tổng số dòng: **11,160,903**
- Unique non-null `nfl_id`: **448** (dòng `nfl_id` null: 0)
- Unique `game_id`: **334**
- Unique `(game_id, play_id)`: **42,544**

#### Độ phủ theo mùa giải (join `game_id` với `games.csv`)

| season | season_type | Unique game_id | Unique nfl_id |
| ---: | :--- | ---: | ---: |
| 2025 | POST | 13 | 115 |
| 2025 | PRE | 49 | 420 |
| 2025 | REG | 272 | 353 |

#### Cột `event` — 42 giá trị duy nhất (non-null)

| Value | Count | % of Rows |
| :--- | ---: | ---: |
| `<NULL>` | 10,425,238 | 93.41% |
| ball_snap | 154,839 | 1.39% |
| first_contact | 91,278 | 0.82% |
| pass_forward | 91,231 | 0.82% |
| tackle | 91,186 | 0.82% |
| pass_arrived | 68,478 | 0.61% |
| pass_outcome_caught | 54,929 | 0.49% |
| handoff | 46,564 | 0.42% |
| pass_outcome_incomplete | 31,977 | 0.29% |
| play_action | 25,740 | 0.23% |
| out_of_bounds | 16,484 | 0.15% |
| autoevent_passinterrupted | 11,209 | 0.10% |
| run | 7,513 | 0.07% |
| qb_sack | 6,039 | 0.05% |
| man_in_motion | 5,653 | 0.05% |
| line_set | 4,207 | 0.04% |
| touchdown | 4,016 | 0.04% |
| play_submit | 3,869 | 0.03% |
| dropped_pass | 2,971 | 0.03% |
| pass_outcome_touchdown | 2,178 | 0.02% |
| pass_outcome_interception | 2,141 | 0.02% |
| pass_tipped | 1,984 | 0.02% |
| qb_kneel | 1,591 | 0.01% |
| fumble | 1,540 | 0.01% |
| qb_slide | 1,158 | 0.01% |
| qb_sneak | 1,073 | 0.01% |
| fumble_defense_recovered | 1,001 | 0.01% |
| fumble_offense_recovered | 956 | 0.01% |
| lateral | 701 | 0.01% |
| qb_strip_sack | 682 | 0.01% |
| run_attempt | 580 | 0.01% |
| snap_direct | 450 | 0.00% |
| pass_shovel | 334 | 0.00% |
| qb_spike | 307 | 0.00% |
| touchback | 232 | 0.00% |
| penalty_flag | 225 | 0.00% |
| two_point_conversion | 210 | 0.00% |
| safety | 43 | 0.00% |
| huddle_break_offense | 34 | 0.00% |
| shift | 26 | 0.00% |
| penalty_accepted | 20 | 0.00% |
| huddle_start_offense | 11 | 0.00% |
| penalty_declined | 5 | 0.00% |

## 4. Khóa liên kết `nfl_id` & Mức độ trùng khớp giữa Combine và mùa giải NFL

### 4.1. Tổng số cầu thủ duy nhất (`nfl_id`) theo từng bảng và nhóm nguồn

| Nguồn / Nhóm | Mô tả | Unique `nfl_id` |
| :--- | :--- | ---: |
| `players.csv` | Danh sách cầu thủ gốc | 510 |
| `combine_results.csv` | Chỉ số đo lường & điểm Combine | 510 |
| `combine_tracking.csv` (tất cả `entity_type`) | Tracking Combine (toàn bảng) | 510 |
| `combine_tracking.csv` (`PLAYER` only) | Tracking Combine (chỉ `entity_type == "PLAYER"`) | 510 |
| `combine_any` (`results ∪ tracking`) | Có ít nhất 1 nguồn Combine | 510 |
| `combine_both` (`results ∩ tracking`) | Có cả Combine results & Combine tracking | 510 |
| `player_play.csv` | Thống kê theo play NFL (2023–2025) | 497 |
| `game_tracking_2023.csv` | Tracking trận đấu 2023 | 156 |
| `game_tracking_2024.csv` | Tracking trận đấu 2024 | 318 |
| `game_tracking_2025.csv` | Tracking trận đấu 2025 | 448 |
| `game_tracking_all` (`2023 ∪ 2024 ∪ 2025`) | Có tracking trận đấu ít nhất 1 mùa | 497 |
| `nfl_season_any` (`player_play ∪ game_tracking_all`) | Có mặt trong dữ liệu mùa giải NFL | 497 |
| `nfl_season_both` (`player_play ∩ game_tracking_all`) | Có cả `player_play` và `game_tracking` | 497 |
| `player_career_successes.csv` | Tổng hợp thành tích sự nghiệp NFL | 510 |

- Kiểm tra `combine_tracking.csv`: tập `nfl_id` có `entity_type == "PLAYER"` gồm **510** ID; số ID chỉ xuất hiện với `BALL` (không có dòng `PLAYER`) là **0**.

### 4.2. Bảng đối chiếu trùng khớp Combine ↔ Mùa giải NFL

| Comparison (A ↔ B) | |A| | |B| | Matched |A ∩ B| | % of A | % of B | Only in A | Only in B |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `combine_results` ↔ `combine_tracking` | 510 | 510 | 510 | 100.00% | 100.00% | 0 | 0 |
| `combine_results` ↔ `player_play` | 510 | 497 | 497 | 97.45% | 100.00% | 13 | 0 |
| `combine_results` ↔ `game_tracking_2023` | 510 | 156 | 156 | 30.59% | 100.00% | 354 | 0 |
| `combine_results` ↔ `game_tracking_2024` | 510 | 318 | 318 | 62.35% | 100.00% | 192 | 0 |
| `combine_results` ↔ `game_tracking_2025` | 510 | 448 | 448 | 87.84% | 100.00% | 62 | 0 |
| `combine_results` ↔ `game_tracking_all` | 510 | 497 | 497 | 97.45% | 100.00% | 13 | 0 |
| `combine_results` ↔ `nfl_season_any` | 510 | 497 | 497 | 97.45% | 100.00% | 13 | 0 |
| `combine_results` ↔ `player_career_successes` | 510 | 510 | 510 | 100.00% | 100.00% | 0 | 0 |
| `combine_tracking` ↔ `player_play` | 510 | 497 | 497 | 97.45% | 100.00% | 13 | 0 |
| `combine_tracking` ↔ `game_tracking_2023` | 510 | 156 | 156 | 30.59% | 100.00% | 354 | 0 |
| `combine_tracking` ↔ `game_tracking_2024` | 510 | 318 | 318 | 62.35% | 100.00% | 192 | 0 |
| `combine_tracking` ↔ `game_tracking_2025` | 510 | 448 | 448 | 87.84% | 100.00% | 62 | 0 |
| `combine_tracking` ↔ `game_tracking_all` | 510 | 497 | 497 | 97.45% | 100.00% | 13 | 0 |
| `combine_tracking` ↔ `nfl_season_any` | 510 | 497 | 497 | 97.45% | 100.00% | 13 | 0 |
| `combine_tracking` ↔ `player_career_successes` | 510 | 510 | 510 | 100.00% | 100.00% | 0 | 0 |
| `combine_any` ↔ `nfl_season_any` | 510 | 497 | 497 | 97.45% | 100.00% | 13 | 0 |
| `combine_any` ↔ `player_career_successes` | 510 | 510 | 510 | 100.00% | 100.00% | 0 | 0 |
| `combine_both` ↔ `nfl_season_both` | 510 | 497 | 497 | 97.45% | 100.00% | 13 | 0 |

### 4.3. Phân tách trùng khớp theo `draft_year` (từ `players.csv`)

| draft_year | players | combine_results | combine_tracking | player_play | gt_2023 | gt_2024 | gt_2025 | gt_all | career_successes | combine_any ∩ nfl_any | combine_both ∩ nfl_both |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023 | 160 | 160 | 160 | 158 | 156 | 146 | 125 | 158 | 160 | 158 | 158 |
| 2024 | 177 | 177 | 177 | 175 | 0 | 172 | 159 | 175 | 177 | 175 | 175 |
| 2025 | 173 | 173 | 173 | 164 | 0 | 0 | 164 | 164 | 173 | 164 | 164 |

### 4.4. Ma trận giao `|Set_i ∩ Set_j|` giữa 8 bảng chứa `nfl_id`

| Table | `players` | `combine_results` | `combine_tracking` | `player_play` | `game_tracking_2023` | `game_tracking_2024` | `game_tracking_2025` | `player_career_successes` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `players` | 510 | 510 | 510 | 497 | 156 | 318 | 448 | 510 |
| `combine_results` | 510 | 510 | 510 | 497 | 156 | 318 | 448 | 510 |
| `combine_tracking` | 510 | 510 | 510 | 497 | 156 | 318 | 448 | 510 |
| `player_play` | 497 | 497 | 497 | 497 | 156 | 318 | 448 | 497 |
| `game_tracking_2023` | 156 | 156 | 156 | 156 | 156 | 145 | 123 | 156 |
| `game_tracking_2024` | 318 | 318 | 318 | 318 | 145 | 318 | 280 | 318 |
| `game_tracking_2025` | 448 | 448 | 448 | 448 | 123 | 280 | 448 | 448 |
| `player_career_successes` | 510 | 510 | 510 | 497 | 156 | 318 | 448 | 510 |

### 4.5. Kiểm tra tính toàn vẹn khóa `nfl_id` so với `players.csv`

| File | Unique `nfl_id` | Orphan IDs (ngoài `players.csv`) | ID trong `players.csv` vắng mặt | Ví dụ Orphan IDs |
| :--- | ---: | ---: | ---: | :--- |
| `combine_results.csv` | 510 | 0 | 0 | — |
| `combine_tracking.csv` | 510 | 0 | 0 | — |
| `player_play.csv` | 497 | 0 | 13 | — |
| `game_tracking_2023.csv` | 156 | 0 | 354 | — |
| `game_tracking_2024.csv` | 318 | 0 | 192 | — |
| `game_tracking_2025.csv` | 448 | 0 | 62 | — |
| `player_career_successes.csv` | 510 | 0 | 0 | — |

