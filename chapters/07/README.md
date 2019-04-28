# Chapter 7: "Power Analysis Using R"

These Python scripts are reproductions of the R scripts provided in
the [SIGIR 2016 pack](https://waseda.box.com/SIGIR2016PACK):

* future.sample.pairedt
* future.sample.unpairedt
* future.sample.1wayanova
* future.sample.2waynorep
* future.sample.2wayanova2

## Reliance on StatsModels

Unlike other chapters, this chapter does not describe the theory
behind power analysis in enough detail to reproduce the calculations
from scratch. The Python scripts here implement the power analysis
functions using the [power module in
statsmodels](https://www.statsmodels.org/stable/stats.html#power-and-sample-size-calculations) instead.

### Discrepency with R

Use the ANOVA scripts in this directory with caution: there are
discrepancies between the values statsmodels F-test power calculations
and those from R. See #1 for details.