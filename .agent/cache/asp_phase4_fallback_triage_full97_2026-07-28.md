# ASP Phase 4 — Full-Corpus Fallback Triage

Generated from `anime_stitch_20260728_013215.json` (54 fallback tests across 3 gate classes). Sorted by margin-over-limit (closest to passing first) within each class -- a triage aid for per-test policy review, not an automatic dispatch rule (see module docstring for why).

## `seam_vis_gate` (27 tests)

| test | seam_visibility | limit | margin | mean_post_warp_diff | pair_count | phases | likely root cause |
|------|------:|------:|------:|------:|------:|------:|------|
| asp_test87 | 35.4 | 35.0 | 0.4 | 13.0 | 47 | 4 | mixed/moderate |
| asp_test10 | 36.3 | 35.0 | 1.3 | 10.4 | 9 | 1 | mixed/moderate |
| asp_test71 | 38.3 | 35.0 | 3.3 | 6.1 | 8 | 2 | photometric-leaning (low post_warp_diff) |
| asp_test69 | 38.7 | 35.0 | 3.7 | 5.5 | 12 | 1 | photometric-leaning (low post_warp_diff) |
| asp_test32 | 39.0 | 35.0 | 4.0 | 18.3 | 26 | 3 | mixed/moderate |
| asp_test09 | 39.8 | 35.0 | 4.8 | 3.0 | 50 | 3 | photometric-leaning (low post_warp_diff) |
| asp_test57 | 40.5 | 35.0 | 5.5 | 12.0 | 44 | 2 | mixed/moderate |
| asp_test43 | 41.8 | 35.0 | 6.8 | 14.9 | 24 | 3 | mixed/moderate |
| asp_test01 | 42.5 | 35.0 | 7.5 | 7.4 | 36 | 3 | photometric-leaning (low post_warp_diff) |
| asp_test85 | 43.1 | 35.0 | 8.1 | 8.6 | 27 | 1 | photometric-leaning (low post_warp_diff) |
| asp_test56 | 43.2 | 35.0 | 8.2 | 17.9 | 6 | 2 | mixed/moderate |
| asp_test26 | 43.6 | 35.0 | 8.6 | 13.6 | 7 | 3 | mixed/moderate |
| asp_test54 | 43.7 | 35.0 | 8.7 | 12.3 | 14 | 2 | mixed/moderate |
| asp_test74 | 43.7 | 35.0 | 8.7 | 9.1 | 43 | 2 | photometric-leaning (low post_warp_diff) |
| asp_test80 | 45.9 | 35.0 | 10.9 | 4.9 | 45 | 5 | photometric-leaning (low post_warp_diff) |
| asp_test77 | 49.9 | 35.0 | 14.9 | 20.1 | 29 | 2 | mixed/moderate |
| asp_test94 | 59.5 | 35.0 | 24.5 | 11.8 | 7 | 2 | mixed/moderate |
| asp_test78 | 60.3 | 35.0 | 25.3 | 7.3 | 39 | 1 | photometric-leaning (low post_warp_diff) |
| asp_test51 | 60.6 | 35.0 | 25.6 | 16.5 | 6 | 3 | mixed/moderate |
| asp_test41 | 63.8 | 35.0 | 28.8 | 44.7 | 8 | 2 | pose-blend-leaning (high post_warp_diff) |
| asp_test63 | 64.0 | 35.0 | 29.0 | 10.9 | 22 | 5 | mixed/moderate |
| asp_test37 | 65.6 | 35.0 | 30.6 | 6.7 | 67 | 1 | photometric-leaning (low post_warp_diff) |
| asp_test23 | 65.7 | 35.0 | 30.7 | n/a | 12 | 1 | unknown |
| asp_test13 | 76.2 | 35.0 | 41.2 | 23.8 | 12 | 1 | mixed/moderate |
| asp_test92 | 82.1 | 35.0 | 47.1 | 6.7 | 55 | 1 | photometric-leaning (low post_warp_diff) |
| asp_test08 | 143.3 | 35.0 | 108.3 | 15.3 | 19 | 3 | mixed/moderate |
| asp_test82 | 155.5 | 35.0 | 120.5 | 13.4 | 43 | 4 | mixed/moderate |

## `composite_gate_sb` (26 tests)

| test | binding value | limit | margin | mean_post_warp_diff | pair_count | phases | sc/sb detail |
|------|------:|------:|------:|------:|------:|------:|------|
| asp_test25 | 36.4 | 35.0 | 1.4 | 13.2 | 7 | 1 | sc=41.4/88.5 sb=36.4/35.0 |
| asp_test16 | 37.0 | 35.0 | 2.0 | 23.2 | 15 | 4 | sc=26.5/70.8 sb=37.0/35.0 |
| asp_test11 | 37.3 | 35.0 | 2.3 | 13.6 | 11 | 2 | sc=23.8/38.0 sb=37.3/35.0 |
| asp_test24 | 37.8 | 35.0 | 2.8 | n/a | 3 | 1 | sc=30.1/38.9 sb=37.8/35.0 |
| asp_test53 | 37.9 | 35.0 | 2.9 | 72.6 | 2 | 1 | sc=26.7/88.7 sb=37.9/35.0 |
| asp_test42 | 38.0 | 35.0 | 3.0 | 40.8 | 14 | 5 | sc=16.5/38.0 sb=38.0/35.0 |
| asp_test91 | 38.0 | 35.0 | 3.0 | 20.9 | 40 | 2 | sc=32.5/52.2 sb=38.0/35.0 |
| asp_test90 | 39.2 | 35.0 | 4.2 | 14.4 | 29 | 1 | sc=23.8/38.0 sb=39.2/35.0 |
| asp_test66 | 40.8 | 35.0 | 5.8 | n/a | 13 | 2 | sc=20.9/38.5 sb=40.8/35.0 |
| asp_test62 | 41.6 | 35.0 | 6.6 | 11.9 | 19 | 2 | sc=36.4/38.6 sb=41.6/35.0 |
| asp_test76 | 41.7 | 35.0 | 6.7 | 31.0 | 7 | 1 | sc=25.4/38.0 sb=41.7/35.0 |
| asp_test30 | 41.9 | 35.0 | 6.9 | 13.5 | 20 | 1 | sc=32.6/66.2 sb=41.9/35.0 |
| asp_test60 | 42.4 | 35.0 | 7.4 | 16.6 | 26 | 1 | sc=30.1/64.3 sb=42.4/35.0 |
| asp_test75 | 44.6 | 35.0 | 9.6 | 10.5 | 27 | 4 | sc=52.0/123.0 sb=44.6/35.0 |
| asp_test67 | 45.0 | 35.0 | 10.0 | 12.0 | 16 | 2 | sc=28.6/39.9 sb=45.0/35.0 |
| asp_test48 | 45.4 | 35.0 | 10.4 | 33.0 | 13 | 1 | sc=46.7/66.0 sb=45.4/35.0 |
| asp_test84 | 46.3 | 35.0 | 11.3 | 14.9 | 21 | 2 | sc=21.0/38.0 sb=46.3/35.0 |
| asp_test72 | 46.5 | 35.0 | 11.5 | 8.6 | 18 | 4 | sc=27.1/52.3 sb=46.5/35.0 |
| asp_test49 | 47.1 | 35.0 | 12.1 | 20.3 | 28 | 3 | sc=39.3/55.4 sb=47.1/35.0 |
| asp_test52 | 51.1 | 35.0 | 16.1 | 11.4 | 16 | 2 | sc=24.7/63.1 sb=51.1/35.0 |
| asp_test93 | 54.8 | 35.0 | 19.8 | n/a | 7 | 1 | sc=18.7/65.3 sb=54.8/35.0 |
| asp_test68 | 59.1 | 35.0 | 24.1 | 7.2 | 9 | 1 | sc=28.3/38.0 sb=59.1/35.0 |
| asp_test20 | 60.9 | 35.0 | 25.9 | 11.2 | 21 | 1 | sc=27.9/38.0 sb=60.9/35.0 |
| asp_test21 | 68.3 | 35.0 | 33.3 | 7.8 | 8 | 2 | sc=30.7/50.8 sb=68.3/35.0 |
| asp_test70 | 70.0 | 35.0 | 35.0 | 19.3 | 22 | 2 | sc=29.7/53.8 sb=70.0/35.0 |
| asp_test89 | 135.8 | 35.0 | 100.8 | 21.4 | 21 | 2 | sc=39.3/85.4 sb=135.8/35.0 |

## `composite_gate_sc` (1 tests)

| test | binding value | limit | margin | mean_post_warp_diff | pair_count | phases | sc/sb detail |
|------|------:|------:|------:|------:|------:|------:|------|
| asp_test58 | 53.5 | 48.2 | 5.3 | 5.3 | 61 | 4 | sc=53.5/48.2 sb=39.3/35.0 |
