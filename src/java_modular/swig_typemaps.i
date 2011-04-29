/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
  *
 * Written (W) 2011 Baozeng Ding
  *  
 */

/* TYPEMAP_IN macros
 *
 * This family of typemaps allows pure input C arguments of the form
 *
 *     (type* IN_ARRAY1, int32_t DIM1)
 *     (type* IN_ARRAY2, int32_t DIM1, int32_t DIM2)
 *
 * where "type" is any type supported by the numpy module, to be
 * called in python with an argument list of a single array (or any
 * python object that can be passed to the numpy.array constructor
 * to produce an arrayof te specified shape).  This can be applied to
 * a existing functions using the %apply directive:
 *
 *     %apply (float64_t* IN_ARRAY1, int32_t DIM1) {float64_t* series, int32_t length}
 *     %apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {float64_t* mx, int32_t rows, int32_t cols}
 *     float64_t sum(float64_t* series, int32_t length);
 *     float64_t max(float64_t* mx, int32_t rows, int32_t cols);
 *
 * or with
 *
 *     float64_t sum(float64_t* IN_ARRAY1, int32_t DIM1);
 *     float64_t max(float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2);
 */

/* One dimensional input arrays */
%define TYPEMAP_IN1(SGTYPE, JTYPE, JAVATYPE, JNITYPE)

%typemap(jni) (SGTYPE* IN_ARRAY1, int32_t DIM1)		%{JNITYPE##Array%}
%typemap(jtype) (SGTYPE* IN_ARRAY1, int32_t DIM1)		%{JTYPE[]%}
%typemap(jstype) (SGTYPE* IN_ARRAY1, int32_t DIM1)	%{JTYPE[]%}

%typemap(in) (SGTYPE* IN_ARRAY1, int32_t DIM1) (JNITYPE *jarr) {  
	int i;
	SGTYPE *array;
	if (!$input) {
		SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null array");
		return $null;	
	}
	$2 = JCALL1(GetArrayLength, jenv, $input);
	jarr = JCALL2(Get##JAVATYPE##ArrayElements, jenv, $input, 0);
	if (!jarr)
		return $null;
	array = new SGTYPE[$2];
	if (!array) {
    SWIG_JavaThrowException(jenv, SWIG_JavaOutOfMemoryError, "array memory allocation failed");
    return $null;
	}
	for (i = 0; i < $2; i++) {
		array[i] = jarr[i];	
	}
	$1 = array;
}

%typemap(out) (SGTYPE* IN_ARRAY1, int32_t DIM1) {
	JNITYPE *arr;
	int i;
	JNITYPE##Array jresult = JCALL1(New##JAVATYPE##Array, jenv, $2);
	if (!jresult)
		return NULL;
	arr = JCALL2(Get##JAVATYPE##ArrayElements, jenv, jresult, 0);
	if (!arr)
		return NULL;
	for (i=0; i < $2; i++)
		arr[i] = (JNITYPE)$1[i];
	JCALL3(Release##JAVATYPE##ArrayElements, jenv, jresult, arr, 0);
	$result = jresult;
}

%typemap(freearg) (SGTYPE* IN_ARRAY1, int32_t DIM1) { 
	delete [] $1;
}

%typemap(javain) (SGTYPE* IN_ARRAY1, int32_t DIM1) "$javainput"
%typemap(javaout) (SGTYPE* IN_ARRAY1, int32_t DIM1) {
	return $jnicall;
}

%enddef


TYPEMAP_IN1(bool, boolean, Boolean, jboolean)
TYPEMAP_IN1(char, byte, Byte, jbyte)
TYPEMAP_IN1(uint8_t, short, Short, jshort)
TYPEMAP_IN1(int16_t, short, Short, jshort)
TYPEMAP_IN1(uint16_t, int, Int, jint)
TYPEMAP_IN1(int32_t, int, Int, jint)
TYPEMAP_IN1(uint32_t, long, Long, jlong)
TYPEMAP_IN1(int64_t, int, Int, jint)
TYPEMAP_IN1(uint64_t, long, Long, jlong)
TYPEMAP_IN1(long long, long, Long, jlong)
TYPEMAP_IN1(float32_t, float, Float, jfloat)
TYPEMAP_IN1(float64_t, double, Double, jdouble)

#undef TYPEMAP_IN1

/* TYPEMAP_ARGOUT macros
 *
 * This family of typemaps allows output C arguments of the form
 *
 *     (type** ARGOUT_ARRAY)
 *
 * where "type" is any type supported by the numpy module, to be
 * called in python with an argument list of a single contiguous
 * numpy array.  This can be applied to an existing function using
 * the %apply directive:
 *
 *     %apply (float64_t** ARGOUT_ARRAY1, {(float64_t** series, int32_t* len)}
 *     %apply (float64_t** ARGOUT_ARRAY2, {(float64_t** matrix, int32_t* d1, int32_t* d2)}
 *
 * with
 *
 *     void sum(float64_t* series, int32_t* len);
 *     void sum(float64_t** series, int32_t* len);
 *     void sum(float64_t** matrix, int32_t* d1, int32_t* d2);
 *
 * where sum mallocs the array and assigns dimensions and the pointer
 *
 */

 /* One dimensional input/output arrays */
%define TYPEMAP_ARRAYOUT1(SGTYPE, JTYPE, JAVATYPE, JNITYPE)

%typemap(jni) (SGTYPE** ARGOUT1, int32_t* DIM1)		%{JNITYPE##Array%}
%typemap(jtype) (SGTYPE** ARGOUT1, int32_t* DIM1)		%{JTYPE[]%}
%typemap(jstype) (SGTYPE** ARGOUT1, int32_t* DIM1)	%{JTYPE[]%}

%typemap(in) (SGTYPE** ARGOUT1, int32_t* DIM1) {
    $1 = (SGTYPE**) malloc(sizeof(SGTYPE*));
    $2 = (int32_t*) malloc(sizeof(int32_t));
}

%typemap(argout) (SGTYPE** ARGOUT1, int32_t* DIM1) {
	SGTYPE* vec = *$1;
	JNITYPE *arr;
	int i;
	arr = JCALL2(Get##JAVATYPE##ArrayElements, jenv, $input, 0);
	if (!arr)
		return;
	for (i=0; i < *$2; i++)
		arr[i] = (JNITYPE)vec[i];
	JCALL3(Release##JAVATYPE##ArrayElements, jenv, $input, arr, 0);
}

%typemap(javain) (SGTYPE** ARGOUT1, int32_t* DIM1) "$javainput"

%enddef


TYPEMAP_ARRAYOUT1(bool, boolean, Boolean, jboolean)
TYPEMAP_ARRAYOUT1(char, byte, Byte, jbyte)
TYPEMAP_ARRAYOUT1(uint8_t, short, Short, jshort)
TYPEMAP_ARRAYOUT1(int16_t, short, Short, jshort)
TYPEMAP_ARRAYOUT1(uint16_t, int, Int, jint)
TYPEMAP_ARRAYOUT1(int32_t, int, Int, jint)
TYPEMAP_ARRAYOUT1(uint32_t, long, Long, jlong)
TYPEMAP_ARRAYOUT1(int64_t, int, Int, jint)
TYPEMAP_ARRAYOUT1(uint64_t, long, Long, jlong)
TYPEMAP_ARRAYOUT1(long long, long, Long, jlong)
TYPEMAP_ARRAYOUT1(float32_t, float, Float, jfloat)
TYPEMAP_ARRAYOUT1(float64_t, double, Double, jdouble)

#undef TYPEMAP_ARRAYOUT1
