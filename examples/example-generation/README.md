# Shogun example meta-language parser
Running the parser requires python and pyparsing (included with many python distributions).

### Translating all examples in “examples/“ folder
Running the following command will translate all programs in the “examples/” folder and generate translation for each target language. The translations are put in a new “outputs/“ folder.

```
./generate.py
```

### Parsing a program:
i.e. retrieve its AST as JSON (pretty printed)
```
$ python parse.py --pretty examples/knn.sg
```

### Parsing and translating a program
Firstly, parse the example
```
$ python parse.py examples/knn.sg > knn.ast
```
Then translate the AST into the target languages
```
python translate.py knn.ast --target=python > knn.py
python translate.py knn.ast --target=java > knn.java
python translate.py knn.ast --target=octave > knn.m
```

Note: it is also possible to pipe the parsing and translation to translate an example in one step. E.g.
```
python parse.py examples/gp.sg | python translate.py --target=python
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