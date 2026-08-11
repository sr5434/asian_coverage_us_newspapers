# Change-point stability report

Recovery means a fitted boundary occurred at the baseline location or within one observation. Annual specification tests vary algorithm, minimum segment length, and requested break count; annual leave-one-out tests omit one year at a time, with the observations adjacent to each boundary summarized separately. Event tests vary PELT penalty and minimum segment length, then compare fixed-count binary segmentation and dynamic programming.

| Analysis | Series | Baseline boundary | Stability | Main recovery | Secondary recovery |
|---|---|---:|---|---:|---:|
| Annual coverage | Chinese surname-match rate | 1791--1800 | partially stable | specifications 73% | leave-one-out 95%; boundary endpoints 1/2 |
| Annual coverage | Chinese surname-match rate | 1880--1890 | unstable/exploratory | specifications 18% | leave-one-out 74%; boundary endpoints 0/2 |
| Annual coverage | Chinese surname-match rate | 1900--1910 | partially stable | specifications 73% | leave-one-out 100%; boundary endpoints 2/2 |
| Annual coverage | Japanese surname-match rate | 1791--1800 | unstable/exploratory | specifications 18% | leave-one-out 89%; boundary endpoints 1/2 |
| Annual coverage | Japanese surname-match rate | 1820--1830 | partially stable | specifications 68% | leave-one-out 100%; boundary endpoints 2/2 |
| Annual coverage | Japanese surname-match rate | 1940--1950 | partially stable | specifications 73% | leave-one-out 95%; boundary endpoints 1/2 |
| Annual positivity margin | Asian | 1791--1800 | partially stable | specifications 73% | leave-one-out 100%; boundary endpoints 2/2 |
| Annual positivity margin | Asian | 1860--1870 | unstable/exploratory | specifications 45% | leave-one-out 89%; boundary endpoints 2/2 |
| Annual positivity margin | Asian | 1890--1900 | partially stable | specifications 73% | leave-one-out 100%; boundary endpoints 2/2 |
| Annual positivity margin | Chinese | 1791--1800 | partially stable | specifications 73% | leave-one-out 100%; boundary endpoints 2/2 |
| Annual positivity margin | Chinese | 1890--1900 | partially stable | specifications 73% | leave-one-out 100%; boundary endpoints 2/2 |
| Annual positivity margin | Chinese | 1910--1920 | unstable/exploratory | specifications 23% | leave-one-out 74%; boundary endpoints 1/2 |
| Annual positivity margin | Japanese | 1791--1800 | partially stable | specifications 73% | leave-one-out 100%; boundary endpoints 2/2 |
| Annual positivity margin | Japanese | 1810--1820 | partially stable | specifications 91% | leave-one-out 95%; boundary endpoints 1/2 |
| Annual positivity margin | Japanese | 1890--1900 | partially stable | specifications 64% | leave-one-out 95%; boundary endpoints 1/2 |
| Annual positivity margin | Korean | 1791--1800 | partially stable | specifications 73% | leave-one-out 100%; boundary endpoints 2/2 |
| Annual positivity margin | Korean | 1890--1900 | partially stable | specifications 73% | leave-one-out 100%; boundary endpoints 2/2 |
| Annual positivity margin | Korean | 1940--1950 | unstable/exploratory | specifications 23% | leave-one-out 42%; boundary endpoints 0/2 |
| Annual positivity margin | Oriental | 1791--1800 | partially stable | specifications 73% | leave-one-out 100%; boundary endpoints 2/2 |
| Annual positivity margin | Oriental | 1860--1870 | partially stable | specifications 73% | leave-one-out 100%; boundary endpoints 2/2 |
| Annual positivity margin | Oriental | 1890--1900 | unstable/exploratory | specifications 41% | leave-one-out 95%; boundary endpoints 2/2 |
| Opium War | Chinese surname-match rate | 1839-06--1839-07 | partially stable | penalty/min-size 60% | algorithm 100% |
| Opium War | Chinese surname-match rate | 1840-02--1840-03 | partially stable | penalty/min-size 60% | algorithm 100% |
| California Gold Rush | Chinese surname-match rate | 1850-08--1850-09 | partially stable | penalty/min-size 55% | algorithm 100% |
| California Gold Rush | Chinese surname-match rate | 1850-11--1850-12 | unstable/exploratory | penalty/min-size 45% | algorithm 38% |
| Chinese Exclusion Act | Chinese surname-match rate | None | stable | no-break result 80% | not applicable |
| Pearl Harbor | Japanese surname-match rate | 1941-09--1941-10 | unstable/exploratory | penalty/min-size 30% | algorithm 25% |
| Pearl Harbor | Japanese surname-match rate | 1941-12--1942-01 | partially stable | penalty/min-size 50% | algorithm 50% |

## Interpretation rules

- Stable: every applicable recovery rate is at least 80%.
- Partially stable: the lowest applicable recovery rate is 50% to 79%.
- Unstable/exploratory: at least one applicable recovery rate is below 50%.

These labels measure model-location stability, not data validity or historical causality. They do not replace article-level validation of unusual observations such as 1791 or 1950.
