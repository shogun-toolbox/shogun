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

# Search in headers for non-template/non-abstract class-names starting
# with `C'.
classes=$(for i in $HEADERS; do
    cat $i | sed -n -e \
's/^[ \t^t\/\*]*class \+[C]\([A-Z][A-Za-z0-9_]\+\)[^;]*$/sXXREPL\1/;
/^sXXREPL/h;/) *\(const\)\? *= *0 *\;/x;/^\} *\;/{g;s/^sXXREPL//p;x}';
    done;
)

# Search in headers for template(only one generic)/non-abstract
# class-names starting with `C'.
temp_classes=$(for i in $HEADERS; do
    cat $i | sed -n -e \
's/^[ \t^t\/\*]*template *<[^,]\+> *class \+[C]\([A-Z][A-Za-z0-9_]\+\)[^;]*$/sXXREPL\1/;
/^sXXREPL/h;/) *\(const\)\? *= *0 *\;/x;/^\} *\;/{g;s/^sXXREPL//p;x}';
    done;
)

files=$(find . -false \
    `echo $classes $temp_classes \
    | sed -e 's/\([^ \t\n]\+\)/-o -name \1.h/g'`)

includes=\
`echo $files | sed -e 's/[ \t\n]*\(\.\/\)\?\([^ \t\n]\+\)/#include \
"\2"\\\\n/g' | sed -n -e '1h;1!H;${g;s/\n//g;p}'`
sed -i'' -e 's@REPLACE \+includes \+THIS@'"$includes"'@' $TEMPL_FILE

definitions=\
`echo $classes | sed -e 's/[ \t\n]*\([^ \t\n]\+\)/static \
CSGObject* __new_C\1(EPrimitiveType g) \{ return g == \
PT_NOT_GENERIC? new C\1(): NULL; \}\\\\n/g' \
    | sed -n -e '1h;1!H;${g;s/\n//g;p}'`
sed -i'' -e 's@REPLACE \+definitions \+THIS@'"$definitions"'@' \
    $TEMPL_FILE

temp_definitions=\
`echo $temp_classes | sed -e 's/[ \t\n]*\([^ \t\n]\+\)/static \
CSGObject* __new_C\1(EPrimitiveType g) \{ \
switch (g) \{ \
case PT_BOOL: return new C\1<bool>(); \
case PT_CHAR: return new C\1<char>(); \
case PT_INT8: return new C\1<int8_t>(); \
case PT_UINT8: return new C\1<uint8_t>(); \
case PT_INT16: return new C\1<int16_t>(); \
case PT_UINT16: return new C\1<uint16_t>(); \
case PT_INT32: return new C\1<int32_t>(); \
case PT_UINT32: return new C\1<uint32_t>(); \
case PT_INT64: return new C\1<int64_t>(); \
case PT_UINT64: return new C\1<uint64_t>(); \
case PT_FLOAT32: return new C\1<float32_t>(); \
case PT_FLOAT64: return new C\1<float64_t>(); \
case PT_FLOATMAX: return new C\1<floatmax_t>(); \
case PT_SGOBJECT: return NULL; \
\} return NULL; \}\\\\n/g' \
    | sed -n -e '1h;1!H;${g;s/\n//g;p}'`
sed -i'' -e 's@REPLACE \+temp_definitions \+THIS@'"$temp_definitions"'@' \
    $TEMPL_FILE

structs=\
`echo $classes $temp_classes | sed -e 's/\([^ \t\n]\+\)/\t\{\
m_class_name: "\1", m_new_sgserializable: __new_C\1\},\\\\n/g' \
    | sed -n -e '1h;1!H;${g;s/\n//g;p}'`
sed -i'' -e 's@REPLACE \+structs \+THIS@'"$structs"'@' $TEMPL_FILE
