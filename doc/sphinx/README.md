# The Shogun cookbook

This is the Shogun cookbook, an easy to maintain collection of API examples.
We use [Sphinx](http://www.sphinx-doc.org/en/stable/) and our own [meta-examples](https://github.com/shogun-toolbox/shogun/wiki/Example_Generation) to generate html pages with code snippets in all target languages.

This is achieved by the following high-level steps:

1. Use the automatically translated [API examples](https://github.com/shogun-toolbox/shogun/tree/feature/sphinxdoc/examples/meta/src/) in Shogun's meta example language. As these are part of our test process, they are guaranteed to be executable and errors (e.g. due to API changes) are easily caught.

2. [Cookbook pages](https://github.com/shogun-toolbox/shogun/tree/feature/sphinxdoc/doc/sphinx/source/examples/) contain text that uses snippets from the above code listings. Those are automatically extracted based on a simple markup system.

3. Automatically build a static html page with a global tab for each of Shogun's target languages. The user can toggle the shown language of the snippets without switching the page itself.

## Adding a page
To add an entry, only two files are needed:

 * A meta language example, e.g. ```shogun/examples/meta/src/classifier/knn.sg``` (which during the build process is automatically translated to ```build/examples/python/classifier/knn.py```, ```build/examples/R/classifier/knn.R```, etc). The file should contain a number of snippet start and end markers, see for example [knn.sg](https://github.com/shogun-toolbox/shogun/blob/feature/sphinxdoc/examples/meta/src/classifier/knn.sg).
 
 * A Sphinx markdown file with matching filename and directory, e.g. ```shogun/doc/sphinx/source/examples/classifier/knn.rst```. This file contains a description of the API example and references to code snippets. The point is to *not* show the full file listing but only a subset. The file should furthermore contain basic math in the form of LaTeX, important references (wikipedia, scientific paper references using BibTeX, etc), and links to the involved Shogun class documentation, github issues, etc. We use Sphinx tags to make this easy, see for example [knn.rst](https://github.com/shogun-toolbox/shogun/blob/feature/sphinxdoc/doc/sphinx/source/examples/classifier/knn.rst).

## Guidelines

 * Orient yourself as close as possible on the above reference example. Include the same elements as used thereim.
 * Write proper English. Pay attention to grammar, spelling, and punctuation.
 * Keep the example *specific*. Only talk about the particular algorithm and its interface. General concepts (for example 'supervised learning') should go to overview pages.
 * Keep the example *local*. Only show code snippets that illustrate API usage, avoid showing the full listing.
 
## Some useful tips

 * All external weblinks are automatically checked and warnings are given if corrupt, but please ensure you do not put in dead links.
 * If you want to add a BibTeX reference, add it to [references.bib](https://github.com/shogun-toolbox/shogun/blob/feature/sphinxdoc/doc/sphinx/source/references.bib). Please do only use properly formated entries.
 * It is very easy to re-structure the main page, or to put in general overview pages (i.e. a general description of the Classification interface of Shogun). Feel free to do so.
 
### Where to start
Our [Python examples](https://github.com/shogun-toolbox/shogun/tree/develop/examples/undocumented/python_modular) are most complete. Translate any example into the meta language (remove it afterwards, the meta examples are part of our tests), and add the corresponding cookbook page.


### Render locally
We automatically upload the cookbook to our webserver in the [nightly build](http://buildbot.shogun-toolbox.org/builders/nightly_all). However, you should have a look at your page before you send a pull request.

This can be done as running

```
make doc
```

This generates a ```doc/sphinx/html``` folder in your build folder. To see the results, run

```
python -m SimpleHTTPServer
```

in that directory and then open your browser at ```localhost:8000```.


## Python requirements

 * [sphinx](http://www.sphinx-doc.org/)
 * [sphinxcontrib-bibtex](https://sphinxcontrib-bibtex.readthedocs.org/)
