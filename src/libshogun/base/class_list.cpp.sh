#!/bin/sh

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Written (W) 2008-2009 Soeren Sonnenburg
# Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max Planck Society


TEMPL_FILE=$1
HEADERS=${*#$TEMPL_FILE}

# Search in headers for non-template class-names starting with `C'.
classes=\
`sed -n 's/^CLASSLIST_TODO[^t\/\*]*class \+[C]\([A-Z][A-Za-z0-9_]\+\)[^;]*$/\1/p;' \
    $HEADERS`

files=$(find . -false \
    `echo $classes | sed 's/\([^ \t\n]\+\)/-o -name \1.h/g'`)

includes=\
`echo $files | sed 's/[ \t\n]*\(\.\/\)\?\([^ \t\n]\+\)/#include "\2"\\\\n/g' \
    | sed -n '1h;1!H;${g;s/\n//g;p}'`
sed -i 's@REPLACE \+includes \+THIS@'"$includes"'@' $TEMPL_FILE

definitions=\
`echo $classes | sed 's/[ \t\n]*\([^ \t\n]\+\)/static CSGSerializable* __new_C\1(void) \{ return new C\1(); \}\\\\n/g' \
    | sed -n '1h;1!H;${g;s/\n//g;p}'`
sed -i 's@REPLACE \+definitions \+THIS@'"$definitions"'@' $TEMPL_FILE

structs=\
`echo $classes | sed 's/\([^ \t\n]\+\)/\t\{class_name: "\1", new_sgserializable: __new_C\1\},\\\\n/g' \
    | sed -n '1h;1!H;${g;s/\n//g;p}'`
sed -i 's@REPLACE \+structs \+THIS@'"$structs"'@' $TEMPL_FILE
