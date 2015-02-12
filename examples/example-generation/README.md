# Shogun example meta-language parser
The example generation tool simplifies the maintenance of shogun API examples across all the target languages. Shogun interfaces to many target languages and a set of API examples for each language is maintained as part of the Shogun user documentation. Since all API examples are almost identical across the languages (they mostly only differ syntactically), the example generation tool implements a *very* simple meta-language that translates to all target languages. Consequently, each example only needs to be written once in the meta-language and can automatically be translated to each target language.

Running the parser requires python and pyparsing (included with many python distributions).

### Translating all examples in “examples/“ folder
Running the following command will translate all programs in the “examples/” folder and generate translation for each target language. The translations are put in a new “outputs/“ folder.

```
./generate.py
```
The translated examples can then be run for each target language. E.g.:

```
python outputs/python/knn.py

javac -d . -cp "/usr/local/share/java/*:/usr/share/java/*" outputs/java/knn.java
java -cp "/usr/local/share/java/*:/usr/share/java/*:." Example

R -f outputs/r/gp.R

octave outputs/octave/liblinear.m
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