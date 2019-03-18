# sakai-book

Implementaions of statistical methods from [Tetsuya
Sakai](http://sakailab.com/tetsuya/)'s [Laboratory Experiments in
Information Retrieval: Sample Sizes, Effect Sizes, and Statistical
Power](http://sakailab.com/leirbook/)
([AMZN](https://www.amazon.com/dp/9811311986))

# HOWTO

All scripts within the *chapters* directory should be able to run
standalone. Please set the PYTHONPATH appropriately:

```bash
$> git clone https://github.com/jerome-white/sakai-book.git
$> cd sakai-book
$> export PYTHONPATH=`pwd`:$PYTHONPATH
```

Unless otherwise noted, all scripts read results from `stdin`.

The book assumes results are stored in a matrix, where "systems" are
columns and "topics" are rows; from Section 2.2, for example:

> Let *x<sub>1j</sub>* denote the nDCG score of System 1 for the
> *j*-th topic; similarly, let *x<sub>2j</sub>* denote the nDCG score
> of System 2 for the *j*-th topic (*j* = 1, ..., *n*).

As such, this repository assumes results are CSV files with this
structure that come from `stdin`. It also assumes the first row of the
input to be neames of the systems.

# References

* [Discpower](http://research.nii.ac.jp/ntcir/tools/discpower-en.html)
