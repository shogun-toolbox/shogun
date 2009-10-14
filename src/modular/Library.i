/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
%define DOCSTR
"The `Library` module gathers all miscellaneous Objects in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Library

/* Documentation */
%feature("autodoc","0");

#ifdef HAVE_DOXYGEN
%include "Library_doxygen.i"
#endif

/* Include Module Definitions */
%include "SGBase.i"
%{
#include <shogun/lib/Cache.h>
#include <shogun/lib/File.h>
#include <shogun/lib/List.h>
#include <shogun/lib/Mathematics.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/SimpleFile.h>
#include <shogun/lib/Time.h>
#include <shogun/lib/Trie.h>
#include <shogun/lib/MemoryMappedFile.h>
#include <shogun/lib/DynamicArray.h>
#include <shogun/structure/PlifBase.h>
#include <shogun/lib/Hash.h>
#include <shogun/lib/Array.h>
#include <shogun/lib/Array2.h>
#include <shogun/lib/Array3.h>
#include <shogun/lib/GCArray.h>
%}

/* Remove C Prefix */
%rename(Cache) CCache;
%rename(File) CFile;
%rename(ListElement) CListElement;
%rename(List) CList;
%rename(Math) CMath;
%rename(Signal) CSignal;
%rename(SimpleFile) CSimpleFile;
%rename(Time) CTime;
%rename(Hash) CHash;
%rename(MemoryMappedFile) CMemoryMappedFile;

%ignore RADIX_STACK_SIZE;
%ignore NUMTRAPPEDSIGS;
%ignore TRIE_TERMINAL_CHARACTER;
%ignore NO_CHILD;

#pragma SWIG nowarn=312,362,389
%warnfilter(509) CArray;
%warnfilter(509) CArray2;
%warnfilter(509) CArray3;

/* Include Class Headers to make them visible from within the target language */

/* Template Class DynamicArray */
%include <shogun/lib/DynamicArray.h>
%template(DynamicCharArray) CDynamicArray<char>;
%template(DynamicByteArray) CDynamicArray<uint8_t>;
%template(DynamicShortArray) CDynamicArray<int16_t>;
%template(DynamicWordArray) CDynamicArray<uint16_t>;
%template(DynamicIntArray) CDynamicArray<int32_t>;
%template(DynamicUIntArray) CDynamicArray<uint32_t>;
%template(DynamicLongArray) CDynamicArray<int64_t>;
%template(DynamicULongArray) CDynamicArray<uint64_t>;
%template(DynamicShortRealArray) CDynamicArray<float32_t>;
%template(DynamicRealArray) CDynamicArray<float64_t>;
%template(DynamicPlifArray) CDynamicArray<CPlifBase*>;

/* Template Class GCArray */
%include <shogun/lib/GCArray.h>
%template(PlifGCArray) CGCArray<CPlifBase*>;

/* Hash */
%apply (uint8_t* IN_ARRAY1, int32_t DIM1) {(uint8_t* data, int32_t len)};
%include <shogun/lib/Hash.h>

/* Template Class Array */
%include <shogun/lib/Array.h>
%template(CharArray) CArray<char>;
%template(ByteArray) CArray<uint8_t>;
%template(ShortArray) CArray<int16_t>;
%template(WordArray) CArray<uint16_t>;
%template(IntArray) CArray<int32_t>;
%template(UIntArray) CArray<uint32_t>;
%template(LongArray) CArray<int64_t>;
%template(ULongArray) CArray<uint64_t>;
%template(ShortRealArray) CArray<float32_t>;
%template(RealArray) CArray<float64_t>;

/* Template Class Array2 */
%include <shogun/lib/Array2.h>
%template(CharArray2) CArray2<char>;
%template(ByteArray2) CArray2<uint8_t>;
%template(ShortArray2) CArray2<int16_t>;
%template(WordArray2) CArray2<uint16_t>;
%template(IntArray2) CArray2<int32_t>;
%template(UIntArray2) CArray2<uint32_t>;
%template(LongArray2) CArray2<int64_t>;
%template(ULongArray2) CArray2<uint64_t>;
%template(ShortRealArray2) CArray2<float32_t>;
%template(RealArray2) CArray2<float64_t>;

/* Template Class Array3 */
%include <shogun/lib/Array3.h>
%template(CharArray3) CArray3<char>;
%template(ByteArray3) CArray3<uint8_t>;
%template(ShortArray3) CArray3<int16_t>;
%template(WordArray3) CArray3<uint16_t>;
%template(IntArray3) CArray3<int32_t>;
%template(UIntArray3) CArray3<uint32_t>;
%template(LongArray3) CArray3<int64_t>;
%template(ULongArray3) CArray3<uint64_t>;
%template(ShortRealArray3) CArray3<float32_t>;
%template(RealArray3) CArray3<float64_t>;

%include <shogun/lib/Cache.h>
%include <shogun/lib/GCArray.h>
%include <shogun/lib/File.h>
%include <shogun/lib/List.h>
%include <shogun/lib/Mathematics.h>
%include <shogun/lib/Signal.h>
%include <shogun/lib/SimpleFile.h>
%include <shogun/lib/Time.h>
%include <shogun/lib/Trie.h>
%include <shogun/lib/MemoryMappedFile.h>
