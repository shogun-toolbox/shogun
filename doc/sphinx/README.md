# A new Shogun web manual
## aka minimal examples 2.0

We attempt to get documentation such as [sckit's](http://scikit-learn.org/stable/user_guide.html).
We showcase our main strength via allowing users to switch the target language of the code snippets with one click.


# Prototype

This is a system to replace our modular examples, while automatically generating a more pretty documentation around them.
We combine easy API examples with a pretty web-documentation where we showcase Shogun's main strength: multiple language bindings with the same syntax.

The idea is to write a two files ```source/examples/classifier/knn.rst``` and ```source/examples/code/knn.sg```, where we describe a method (see 1st file), and can include snippets from a meta-language example (see 2nd file). We then use Sphinx with a custom plugin to generate a pretty page that shows these examples with one tab for each language. Everything automatic!

Run

```
make preview
```

and then open your browser at ```localhost:8000``` to see the demo.
