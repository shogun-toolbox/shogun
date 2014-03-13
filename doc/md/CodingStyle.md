# SHOGUN-TOOLBOX Coding Style

This file documents the coding style used in shogun.  If someone pointed
you to this document, then your code was probably not well-formatted.
But no worries, all you need is to apply the following command to the
files to be committed:

```
$ astyle --style=allman --lineend=linux --indent=tab=8 \\
         --unpad-paren --pad-header --pad-oper \\
         --close-templates --add-brackets \\
         --align-pointer=middle --align-reference=name \\
         inputfile1.cpp ...
```

Currently, we are using `astyle`, the _Artistic Style_ code formatter.
It is available in almost every Linux distribution or can be found on
SourceForge: http://astyle.sourceforge.net/ .
