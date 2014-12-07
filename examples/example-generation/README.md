# Shogun example meta-language parser
Running the parser requires python and pyparsing (included with many python distributions).

### Parsing a program:
i.e. retrieve its AST as JSON
```
$ python parse.py examples/knn.sg
```

### Parsing and translating a program
The AST can be translated to a target language by running
```
$ python parse.py examples/knn.sg | python translate.py
```

### Updating the list of Shogun types
A bash script to fetch the list of built-in Shogun types can be found in `types/gettypelist`. To update the `types/typelist` file run:
```
$ ./types/gettypelist > types/typelist
```

### Running unit tests
Unit tests are found under `tests/` and can be run with nose by changing to the `example-generation` folder and running:
```
$ nosetests
```
Which should return something like
```
..........
----------------------------------------------------------------------
Ran 10 tests in 0.008s

OK
```