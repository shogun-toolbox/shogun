#!/bin/sh

TEMPL_FILE=$1
HEADERS=${*#$TEMPL_FILE}

# Search in headers for non-template class names start with `C'.
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
