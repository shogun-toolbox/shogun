/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
  *
 * Written (W) 2011 Baozeng Ding
  *  
 */
%{
#include <lib/DataType.h>
%}
//%include "lib/DataType.h"

/* One dimensional input/output arrays */

%define TYPEMAP_SGVECTOR(SGTYPE, JTYPE, JAVATYPE, JNITYPE)

%typemap(jni) shogun::SGVector<SGTYPE>		%{JNITYPE##Array%}
%typemap(jtype) shogun::SGVector<SGTYPE>		%{JTYPE[]%}
%typemap(jstype) shogun::SGVector<SGTYPE> 	%{JTYPE[]%}

%typemap(in) shogun::SGVector<SGTYPE> (JNITYPE *jarr) {
	int32_t i, len;
	SGTYPE *array;
	if (!$input) {
		SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null array");
		return $null;	
	}
	
	len = JCALL1(GetArrayLength, jenv, $input);
	jarr = JCALL2(Get##JAVATYPE##ArrayElements, jenv, $input, 0);
	if (!jarr)
		return $null;
	
	array = new SGTYPE[len];
	if (!array) {
		SWIG_JavaThrowException(jenv, SWIG_JavaOutOfMemoryError, "array memory allocation failed");
		return $null;
	}
	for (i = 0; i < len; i++) {
		array[i] = jarr[i];	
	}
	
	$1 = shogun::SGVector<SGTYPE>((SGTYPE *)array, len);
}

%typemap(out) shogun::SGVector<SGTYPE> {
	JNITYPE *arr;
	int32_t i;
	JNITYPE##Array res = JCALL1(New##JAVATYPE##Array, jenv, $1.length);
	if (!res)
		return NULL;
	
	arr = JCALL2(Get##JAVATYPE##ArrayElements, jenv, res, 0);
	if (!arr)
		return NULL;
	
	for (i=0; i < $1.length; i++)
		arr[i] = (JNITYPE)$1.vector[i];
	
	JCALL3(Release##JAVATYPE##ArrayElements, jenv, res, arr, 0);	
	$result = res;
}

%typemap(javain) shogun::SGVector<SGTYPE> "$javainput"
%typemap(javaout) shogun::SGVector<SGTYPE> {
	return $jnicall;
}

%enddef

/* Define concrete examples of the TYPEMAP_SGVECTOR macros */
TYPEMAP_SGVECTOR(bool, boolean, Boolean, jboolean)
TYPEMAP_SGVECTOR(char, byte, Byte, jbyte)
TYPEMAP_SGVECTOR(uint8_t, short, Short, jshort)
TYPEMAP_SGVECTOR(int16_t, short, Short, jshort)
TYPEMAP_SGVECTOR(uint16_t, int, Int, jint)
TYPEMAP_SGVECTOR(int32_t, int, Int, jint)
TYPEMAP_SGVECTOR(uint32_t, long, Long, jlong)
TYPEMAP_SGVECTOR(int64_t, int, Int, jint)
TYPEMAP_SGVECTOR(uint64_t, long, Long, jlong)
TYPEMAP_SGVECTOR(long long, long, Long, jlong)
TYPEMAP_SGVECTOR(float32_t, float, Float, jfloat)
TYPEMAP_SGVECTOR(float64_t, double, Double, jdouble)

#undef TYPEMAP_SGVECTOR

/* Two dimensional input/output arrays */
%define TYPEMAP_SGMATRIX(SGTYPE, JTYPE, JAVATYPE, JNITYPE, JNIDESC)

%typemap(jni) shogun::SGMatrix<SGTYPE>		%{jobjectArray%}
%typemap(jtype) shogun::SGMatrix<SGTYPE>		%{JTYPE[][]%}
%typemap(jstype) shogun::SGMatrix<SGTYPE> 	%{JTYPE[][]%}

%typemap(in) shogun::SGMatrix<SGTYPE>
{
	int32_t i,j;
	int32_t rows, cols;
	SGTYPE *array;
	##JNITYPE##Array jarr;
	JNITYPE *element;
	if (!$input) {
		SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null array");
		return $null;	
	}
	
	rows = JCALL1(GetArrayLength, jenv, $input);
	cols = 0;
	for (i = 0; i < rows; i++) {
		jarr = (##JNITYPE##Array)JCALL2(GetObjectArrayElement, jenv, $input, i);
		if (!jarr) {
			SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null array");
			return $null;
		}
		if (cols == 0) {
			cols = JCALL1(GetArrayLength, jenv, jarr);
			array = new SGTYPE[rows * cols];
			if (!array) {
    			SWIG_JavaThrowException(jenv, SWIG_JavaOutOfMemoryError, "array memory allocation failed");
    			return $null;
			}
		}
		element = JCALL2(Get##JAVATYPE##ArrayElements, jenv, jarr, 0);
		for (j = 0; j < cols; j++) {
			array[i * cols + j] = element[j];
		}
	}

    $1 = shogun::SGMatrix<SGTYPE>((SGTYPE*)array, rows, cols);
}

%typemap(out) shogun::SGMatrix<SGTYPE>
{
	int32_t rows = $1.num_rows;
	int32_t cols = $1.num_cols;
	int32_t i, j;
	
	jclass cls = JCALL1(FindClass, jenv, JNIDESC);
	$result = JCALL3(NewObjectArray, jenv, rows, cls, NULL);	
	if (!$result) {
		return NULL;
	}

	for (i = 0; i < rows; i++) {
		JNITYPE arr[cols];
		##JNITYPE##Array jarr = (##JNITYPE##Array)JCALL1(New##JAVATYPE##Array, jenv, cols);
		if (!jarr) {
			return NULL;		
		}
		
		for (j = 0; j < cols; j++) {
			arr[j] = (JNITYPE)$1.matrix[i * cols + j];
		}
		
		JCALL4(Set##JAVATYPE##ArrayRegion, jenv, jarr, 0, cols, arr);
		JCALL3(SetObjectArrayElement, jenv, $result, i, jarr);
		JCALL1(DeleteLocalRef, jenv, jarr);
		 
	}
}
%enddef

/* Define concrete examples of the TYPEMAP_SGMATRIX macros */
//TYPEMAP_SGMATRIX(bool, boolean, Boolean, jboolean, "[Z")
//TYPEMAP_SGMATRIX(char, byte, Byte, jbyte, "[B")
TYPEMAP_SGMATRIX(uint8_t, short, Short, jshort, "[S")
TYPEMAP_SGMATRIX(int16_t, short, Short, jshort, "[S")
TYPEMAP_SGMATRIX(uint16_t, int, Int, jint, "[I")
TYPEMAP_SGMATRIX(int32_t, int, Int, jint, "[I")
TYPEMAP_SGMATRIX(uint32_t, long, Long, jlong, "[J")
TYPEMAP_SGMATRIX(int64_t, int, Int, jint, "[I")
TYPEMAP_SGMATRIX(uint64_t, long, Long, jlong, "[J")
//TYPEMAP_SGMATRIX(long long, long, Long, jlong, "[J")
TYPEMAP_SGMATRIX(float32_t, float, Float, jfloat, "[F")
TYPEMAP_SGMATRIX(float64_t, double, Double, jdouble, "[D")

#undef TYPEMAP_SGMATRIX
