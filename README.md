# sakai-book

Implementaions of statistical methods from [Tetsuya
Sakai](http://sakailab.com/tetsuya/)'s [Laboratory Experiments in
Information Retrieval: Sample Sizes, Effect Sizes, and Statistical
Power](http://sakailab.com/leirbook/)
([AMZN](https://www.amazon.com/dp/9811311986))

# HOWTO

## As scripts

All scripts within the *chapters* directory can be run
standalone, from the command. First, set the PYTHONPATH appropriately:

```bash
$> git clone https://github.com/jerome-white/sakai-book.git
$> cd sakai-book
$> export PYTHONPATH=`pwd`:$PYTHONPATH
```

Depending on the test/chapter that the script is implementing, the exact command line arguments accepted by the script may differ. Use `--help` to understand what those options are. As an example:

```bash
$> python chapters/02/t-test.py --help
```

Unless otherwise noted, all scripts read results from `stdin`. They assume input is "tidy" CSV, where each row contains a system identifer, a topic identifer, and a score. To that end, the first row should be the following header:

```
system,topic,score
```

Column order does not matter.

## As API

Scripts wishing to use the methods and tests provided by this library should first import irstats:

```python
import irstats as irs
```

From there the a `Scores` object should be created. Provide a list of `Score` [namedtuples](https://docs.python.org/3.7/library/collections.html#collections.namedtuple) to the `Scores` constructor to instantiate an instance. Specific tests can be run by passing an instance of that object.

# References

* [Discpower](http://research.nii.ac.jp/ntcir/tools/discpower-en.html)
