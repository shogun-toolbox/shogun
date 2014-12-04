# Shogun example meta-language parser
Running the parser requires python and pyparsing (included with many python distributions).

To run:

```
python parser.py program.sg
```

### Updating the list of Shogun types
A bash script to fetch the list of built-in Shogun types can be found in `gettypelist`. To update the `typelist` file run:
```
./gettypelist > typelist
```