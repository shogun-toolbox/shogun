# API examples

Shogun comes with automatically generated API examples in all interface languages, see [our website](http://shogun.ml/examples).
This readme describes their mechanics and how they are used for documentation.
See also [DEVELOPING.md](DEVELOPING.md) for their role in testing.

# Quicklinks
 * [Automagically generated examples](#meta_examples)
 * [HTML Cookbook](#cookbook)

# Automatically generated examples <a name="meta_examples"></a>
In Shogun, writing a single example file covers all interface languages at once, a subset can be seen on [our website](http://shogun.ml/examples).

The listings that can be found in `examples/meta/src/*/*.sg.in` contain example code in a meta-language that is specific to Shogun.
During the build, these are parsed and then translated with the (Python) machinery in `examples/meta/generator/*.py`.
The output is a code listing for each target language defined in `examples/meta/src/generator/targets/*.json`.
The process can be invoked with

    make meta_examples

This is only available when the cmake option `BUILD_META_EXAMPLES=ON` is set, and the Python requirements in `examples/meta/requirements.txt` are met.

The C++ examples are always available, you can compile them using `make` or more specifically

    make build_cpp_meta_examples

and run them as from their folder straight-away (you might have to set environmental variables, see [INTERFACES.md](INTERFACES.md))

    cd examples/meta/cpp/multiclass
    ./cpp-multiclass-k_nearest_neighbours

As the examples are part of the tests, you can easily run them as described in [DEVELOPING.md](DEVELOPING.md#testing).
Alternatively, you see [INTERFACES.md](INTERFACES.md) on how to run them manually.

For details, see `CMakeLists.txt` in `examples/meta/` for details, `generate.py` and `translate.py` in `examples/meta/generator/`.

## Adding new examples
It is extremely simple to add a new example: simply create another `*.sg.in` file.
We are currently porting all existing Python examples in the deprecated folder `examples/undocumented/python_modular` to the new system -- a copy-paste [entrance task](https://github.com/shogun-toolbox/shogun/issues/3555).

If you are porting a new example, great! Better yet, add the [data](https://github.com/shogun-toolbox/shogun-data) so that we can do integration tests automatically as described in [DEVELOPING.md](DEVELOPING.md#testing).

Please take inspiration from the existing examples, especially those that were written as part of the [Google Summer of Code](https://github.com/shogun-toolbox/shogun/wiki/GSoC-follow-up-blog-posts) 2016.

Please don't break the build. Always compile and run at least the C++ version of the example.
Check potential requirements of C++ guards that can make a class used in the example unavailable (`HAVE_LAPACK`, `HAVE_NLOPT`, `USE_GPL_SHOGUN`, etc); potentially add them [here](https://github.com/shogun-toolbox/shogun/blob/develop/cmake/FindMetaExamples.cmake).

# Website rendering of examples: the API cookbook <a name="cookbook"></a>

The [The Shogun API cookbook](http://shogun.ml/examples) is a web-based version that adds additional documentation to an existing example.
The idea is that code snippets (rather than the full listing) from the automatically generated listings can be embedded in a markdown page that describes details, math, and references for the example.
The pages are rendered with [our own plugin](https://github.com/shogun-toolbox/shogun/blob/develop/doc/cookbook/extensions/sgexample.py) for [Sphinx](http://www.sphinx-doc.org/).

## Adding a page
To add an entry for an existing example, create a markdown `*.rst` file with matching filename and directory.
E.g. for the example `examples/meta/src/multiclass/k_nearest_neighbours.sg.in`, this would be

    touch doc/cookbook/source/examples/multiclass/k_nearest_neighbours.rst

Edit the file so that it contains details on the API example and references to code snippets.
The point is to **not** show the full file listing but only snippets.
The file should furthermore contain basic math in the form of LaTeX, important references (Wikipedia, scientific paper references using BibTeX, other pages, etc).
Take inspiration from existing pages.

If you are adding a new topic (like "kernels" or "regression") you will also need to update the `index.rst` file in `doc/cookbook/source/`. Follow the template of the existing cookbooks.

### Tips for cookbook pages

 * Orient yourself closely to reference examples, especially those written during the [Google Summer of Code](https://github.com/shogun-toolbox/shogun/wiki/GSoC-follow-up-blog-posts) 2016.
 * Write a proper English. Pay attention to grammar, spelling, and punctuation.
 * Keep the example **specific**. Talk only about the particular algorithm and its interface, avoid general concepts (such as 'supervised learning').
 * Keep the example **local**. Only show code snippets that illustrate API usage, avoid showing the full listing.
 * Let the **code** speak for itself. Avoid useless statements that are clear from the code.
    Avoid statements like "we call the `train` method", but rather "we train the model via".
    This way the `.rst` file is also invariant to API changes.
 * Style: Please put every sentence in a line.
   This makes the diff easier to read in later changes.
 * All external weblinks are automatically checked and warnings are given if corrupt, but please ensure you do not put in dead links.
 * If you want to add a BibTeX reference, add it to [references.bib](https://github.com/shogun-toolbox/shogun/blob/develop/doc/cookbook/source/references.bib).
    Please do only use properly formated entries, follow the existing formatting.
 * If you need a custom LaTeX operator, simply add it to [mathconf.js](https://github.com/shogun-toolbox/shogun/blob/develop/doc/cookbook/source/static/mathconf.js)

## Rendering locally
In addition to the latest release [here](http://shogun.ml/examples), we automatically upload the development version of the cookbook, see [here](http://shogun.ml/examples/nightly/index.html).

Furthermore, if you send a PR, our [buildbot](http://buildbot.shogun-toolbox.org/builders/cookbook%20-%20PR) will automatically upload a preview of the cookbooks to a temporary location.
This is to make our life easier when reviewing the PR.

To make our life even easier, you should look at the cookbook before sending a PR.
You can render it with

    make cookbook

which is also part of `make doc`.
The target might not be available if the requirements in `doc/cookbook/requirements.txt` are not satisfied (in particular Sphinx), or if the meta examples are disabled.

In case the `cookbook` target is still missing then inspect the following.

Find out if the value of `SPHINX_EXECUTABLE` is set by searching for it in CMakeCache.txt which is present in the build directory. If it's not set, then it is the `sphinx-build` that has not been found by the cmake. In this case re-run the cmake script with an explicit path to the `sphinx-build` file. The file can be located by using the command `locate sphinx-build`. Once `sphinx-build` has been located, use the `-DSPHINX_EXECUTABLE="<path/to/sphinx-build>"` cmake flag to specify the `sphinx-build` location. In other words re-run cmake with the following options:

    cmake -DSPHINX_EXECUTABLE="<path/to/sphinx-build>" -DBUILD_META_EXAMPLES=ON [other cmake options] ..

After the cookbook has been rendered, you can view it for example with Python 2 running

    python -m SimpleHTTPServer

or with Python 3 running

    python -m http.server

in the `build/doc/cookbook/html` directory, and then open your browser at `localhost:8000`.
