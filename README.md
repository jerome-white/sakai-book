# sakai-book

Implementaions of statistical methods from [Tetsuya
Sakai](http://sakailab.com/tetsuya/)'s [Laboratory Experiments in
Information Retrieval: Sample Sizes, Effect Sizes, and Statistical
Power](http://sakailab.com/leirbook/)
([AMZN](https://www.amazon.com/dp/9811311986))

# HOWTO

All scripts within the *chapters* directory can be run standalone from
the command line. First, set the PYTHONPATH appropriately:

```bash
$> git clone https://github.com/jerome-white/sakai-book.git
$> cd sakai-book
$> export PYTHONPATH=`pwd`:$PYTHONPATH
```

## As scripts

Depending on the test/chapter that the script is implementing, the
exact command line arguments accepted by the script may differ. Use
`--help` to understand what those options are. As an example:

```bash
$> python chapters/02/t-test.py --help
```

Unless otherwise noted, all scripts read results from `stdin`. They
assume input is [tidy](http://vita.had.co.nz/papers/tidy-data.html)
CSV, where each row contains a system identifer, a topic identifer,
and a score. To distinguish which columns correspond to that
information, the first row should be the following header in any
order:

```
system,topic,score
```

## As API

Scripts wishing to use the methods and tests provided by this library
should first import irstats:

```python
import irstats as irs
```

From there a `Scores` object should be created. To do so, provide a
list of `Score`
[namedtuples](https://docs.python.org/3.7/library/collections.html#collections.namedtuple)
to the `Scores` constructor. Specific tests can be run by passing an
instance of that object.

# Dependencies

Python (3.7+), along with

* numpy
* scipy
* pandas
* statsmodels

# References

* [Discpower](http://research.nii.ac.jp/ntcir/tools/discpower-en.html)
