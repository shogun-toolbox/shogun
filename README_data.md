To fetch the data required to run all the applications / examples / tests you
will need to download the shogun-data package (due to its size) available
separately. In case you use git - we maintain shogun-data as a git submodule.
It is therefore sufficient to call

```
git submodule update --init
```

from the shogun/ git directory to fetch all the data sets and later on to run

```
git submodule update .
```

In case you are using a pre-packaged version of shogun you will need to
manually download the latest data sets from

ftp://shogun-toolbox.org/shogun/data/
