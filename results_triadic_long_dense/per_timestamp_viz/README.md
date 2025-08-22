# Per-Timestamp Influence Visualization

- Background: G_{t-1} (gray), Sampled new edges (blue dashed), Top-k influential (red), Highlighted supporting edges (gold) and common nodes w (yellow).
- k = 4 × (# new edges at t). Coverage = fraction of new edges triad-covered by top-k.

## Stats Table

| t | prev_edges | new_edges | k | coverage | sampled_cov | image |
|---|------------|-----------|---|----------|-------------|-------|
| 0 | 0 | 0 | 0 | 0.000 | 0/0 |  |
| 1 | 50 | 23 | 50 | 1.000 | 10/10 | t01_viz.png |
| 2 | 73 | 6 | 24 | 0.333 | 2/6 | t02_viz.png |
| 3 | 79 | 10 | 40 | 0.000 | 0/10 | t03_viz.png |
| 4 | 89 | 19 | 76 | 0.895 | 8/10 | t04_viz.png |
| 5 | 108 | 3 | 12 | 0.667 | 2/3 | t05_viz.png |
| 6 | 111 | 10 | 40 | 0.000 | 0/10 | t06_viz.png |
| 7 | 121 | 22 | 88 | 0.682 | 7/10 | t07_viz.png |
| 8 | 143 | 27 | 108 | 1.000 | 10/10 | t08_viz.png |
| 9 | 170 | 18 | 72 | 0.444 | 2/10 | t09_viz.png |
| 10 | 188 | 43 | 172 | 0.674 | 5/10 | t10_viz.png |
| 11 | 231 | 70 | 231 | 1.000 | 10/10 | t11_viz.png |
| 12 | 301 | 17 | 68 | 0.412 | 5/10 | t12_viz.png |
| 13 | 318 | 43 | 172 | 0.372 | 3/10 | t13_viz.png |
| 14 | 361 | 2 | 8 | 0.500 | 1/2 | t14_viz.png |
| 15 | 363 | 10 | 40 | 0.000 | 0/10 | t15_viz.png |
| 16 | 373 | 81 | 324 | 0.951 | 9/10 | t16_viz.png |
| 17 | 454 | 49 | 196 | 0.571 | 7/10 | t17_viz.png |
| 18 | 503 | 19 | 76 | 0.474 | 5/10 | t18_viz.png |
| 19 | 522 | 47 | 188 | 0.511 | 6/10 | t19_viz.png |
| 20 | 569 | 2 | 8 | 0.500 | 1/2 | t20_viz.png |
| 21 | 571 | 10 | 40 | 0.000 | 0/10 | t21_viz.png |
| 22 | 581 | 93 | 372 | 0.430 | 6/10 | t22_viz.png |
| 23 | 674 | 381 | 674 | 1.000 | 10/10 | t23_viz.png |
| 24 | 1055 | 25 | 100 | 0.000 | 0/10 | t24_viz.png |
| 25 | 1080 | 114 | 456 | 0.579 | 6/10 | t25_viz.png |
| 26 | 1194 | 223 | 892 | 0.928 | 9/10 | t26_viz.png |
| 27 | 1417 | 41 | 164 | 0.878 | 10/10 | t27_viz.png |
