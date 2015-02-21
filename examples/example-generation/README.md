# Shogun example meta-language parser
The example generation tool simplifies the maintenance of shogun API examples across all the target languages. Shogun interfaces to many target languages and a set of API examples for each language is maintained as part of the Shogun user documentation. Since all API examples are almost identical across the languages (they mostly only differ syntactically), the example generation tool implements a *very* simple meta-language that translates to all target languages. Consequently, each example only needs to be written once in the meta-language and can automatically be translated to each target language.

Running the parser requires python and pyparsing (included with many python distributions).

### Translating all examples in “examples/“ folder
Running the following command will translate all programs in the “examples/” folder and generate translation for each target language. The translations are put in a new “outputs/“ folder. (You may also specify specific targets to translate to - run `$ ./generate.py --help` for more info)

```
$ ./generate.py --input examples --output outputs 
```
The translated examples can then be run for each target language. E.g.:

```
$ python outputs/python/knn.py

$ javac -d . -cp "/usr/local/share/java/*:/usr/share/java/*" outputs/java/knn.java
$ java -cp "/usr/local/share/java/*:/usr/share/java/*:." knn

$ R -f outputs/r/gp.R

$ octave outputs/octave/liblinear.m
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
$ python translate.py knn.ast --target=python > knn.py
$ python translate.py knn.ast --target=java > knn.java
$ python translate.py knn.ast --target=octave > knn.m
```

Note: it is also possible to pipe the parsing and translation to translate an example in one step. E.g.
```
$ python parse.py examples/gp.sg | python translate.py --target=python
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
# Writing examples

The meta-language has a simple syntax. Statements in the language are either initialisations, assigments, expressions, or a print command. Shoguns modular class names are are built-in types of the language, along with the basic types `int`, `bool`, `float`, and `string`. Statements are separated by newlines. Single line comments are support by prepending the line with the symbol `#`.

It is important to notice that this is a meta-language which is never directly compiled/interpreted and executed. To test an example it is therefore necessary to translate it to the target languages and run each of these translations in the respective languages.

Note also that the language does not support common programming constructs such as loops and functions. The language is merely intended to showcase simple examples of the Shogun interfaces in different languages. This implies that you cannot write examples that generate dummy data on the fly. If you need data, please add it to the data repository (e.g. under `/data/toy`) or use an existing data file from there and import it in your example using shoguns APIs to do so (e.g. the `CSVFile` class).

### Initialisation
Object variables can be initialised in two ways - by constructing the object or by copying another object to the variable. When initialising a variable by construction, you must specify a name for the variable, its type, and any arguments to the constructor method. Below is an example of initialising the variable `train_labels` to be a `CSVFile` object constructed with a path to the CSV file.

```
CSVFile train_labels("label_train.csv”)
```

The alternative way of initialising a variable is by simply assigning the variable to some object (copying). Here you must also specify a name and a type for the variable. Below are to examples of this type of initialisation.

```
bool knn_train = knn.train()
RealVector output = test_labels.get_values()
```

### Literals
- The boolean literals are `True` and `False`
- Example of number literals are `1`, `0.000342`, `123.321`
- String literals are enclosed by double quotes. E.g. `"Hello world"`
- Shogun enums may be used by prepending the keyword `enum` and giving an enum value along with its type. e.g. `enum LIBLINEAR_SOLVER_TYPE.L2R_L2LOSS_SVC_DUAL`

### Assignment
Once you have initialised a variable you may assign to a another expression. For assignment, you do not specify the type of the variable. E.g.

```
knn_train = False
```

### Expressions
Expressions are used in assignment, function arguments, printing, and they can statements by themselves (used for side effects). Expressions are either a literal value, a method call, or an identifier. Below are examples of different expressions.

```
# Literal values
“this is a string”
2.2
enum ETransformType.T_LINEAR

# Method calls
svm.set_bias_enabled(True)
gp.get_variance_vector(feats_test)

# Single identifiers
myVariable
```

### Outputting values
The keyword for outputting values to stdout is `print`. E.g.

```
print knn.classify_for_multiple_k()
```

# Adding target languages

Target languages are added by adding a JSON file to the `targets` folder with the appropriate translation rules for the language. Please see the existing target files in the `targets` folder and `translate.py` for examples on how this is done.