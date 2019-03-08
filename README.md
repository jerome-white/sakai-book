# sakai-book

Implementaions of statistical methods from (Tetsuya
Sakai)[http://sakailab.com/tetsuya/]'s (Laboratory Experiments in
Information Retrieval: Sample Sizes, Effect Sizes, and Statistical
Power)[http://sakailab.com/leirbook/]
((AMZN)[https://www.amazon.com/dp/9811311986])

# HOWTO

All scripts within the *chapters* directory should be able to run
standalone. Please set the PYTHONPATH appropriately:

```bash
$> git clone https://github.com/jerome-white/sakai-book.git
$> cd sakai-book
$> export PYTHONPATH=`pwd`:$PYTHONPATH
```

Unless otherwise noted, all scripts read results from `stdin`,
assuming a CSV format in which columns are systems and rows are
topics. The first row of the system is assumed to be the system name.

# References

* [Discpower](http://research.nii.ac.jp/ntcir/tools/discpower-en.html)
