/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Baozeng Ding
 *
 */

%include <java/enumtypeunsafe.swg>

#ifdef HAVE_JBLAS
%pragma(java) jniclassimports=%{
import org.jblas.*;
%}
%typemap(javaimports) SWIGTYPE%{
import java.io.Serializable;
import org.jblas.*;
%}
#else
#ifdef HAVE_UJMP
%pragma(java) jniclassimports=%{
import org.ujmp.core.*;
import org.ujmp.core.doublematrix.impl.DefaultDenseDoubleMatrix2D;
import org.ujmp.core.floatmatrix.impl.DefaultDenseFloatMatrix2D;
import org.ujmp.core.intmatrix.impl.DefaultDenseIntMatrix2D;
import org.ujmp.core.longmatrix.impl.DefaultDenseLongMatrix2D;
import org.ujmp.core.shortmatrix.impl.DefaultDenseShortMatrix2D;
import org.ujmp.core.bytematrix.impl.DefaultDenseByteMatrix2D;
import org.ujmp.core.booleanmatrix.impl.DefaultDenseBooleanMatrix2D;
%}
%typemap(javaimports) SWIGTYPE%{
import org.ujmp.core.*;
import org.ujmp.core.doublematrix.impl.DefaultDenseDoubleMatrix2D;
import org.ujmp.core.floatmatrix.impl.DefaultDenseFloatMatrix2D;
import org.ujmp.core.intmatrix.impl.DefaultDenseIntMatrix2D;
import org.ujmp.core.longmatrix.impl.DefaultDenseLongMatrix2D;
import org.ujmp.core.shortmatrix.impl.DefaultDenseShortMatrix2D;
import org.ujmp.core.bytematrix.impl.DefaultDenseByteMatrix2D;
import org.ujmp.core.booleanmatrix.impl.DefaultDenseBooleanMatrix2D;
%}
#endif
#endif
/* One dimensional input/output arrays */
#ifdef HAVE_JBLAS
/* Two dimensional input/output arrays */
%define TYPEMAP_SGVECTOR(SGTYPE, JTYPE, JAVATYPE, JNITYPE, TOARRAY, CLASSDESC, CONSTRUCTOR)

%typemap(jni) shogun::SGVector<SGTYPE>		%{jobject%}
%typemap(jtype) shogun::SGVector<SGTYPE>		%{##JAVATYPE##Matrix%}
%typemap(jstype) shogun::SGVector<SGTYPE>	%{##JAVATYPE##Matrix%}

%typemap(in) shogun::SGVector<SGTYPE>
{
	jclass cls;
	jmethodID mid;
	SGTYPE *array;
	##JNITYPE##Array jarr;
	JNITYPE *carr;
	int32_t i, cols;
	bool isVector;

	if (!$input) {
		SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null array");
		return $null;
	}

	cls = JCALL1(GetObjectClass, jenv, $input);
	if (!cls)
		return $null;

	mid = JCALL3(GetMethodID, jenv, cls, "isVector", "()Z");
	if (!mid)
		return $null;

	isVector = (int32_t)JCALL2(CallIntMethod, jenv, $input, mid);
	if (!isVector) {
		SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "vector expected");
		return $null;
	}

	mid = JCALL3(GetMethodID, jenv, cls, "getColumns", "()I");
	if (!mid)
		return $null;

	cols = (int32_t)JCALL2(CallIntMethod, jenv, $input, mid);
	if (cols < 1) {
		SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null vector");
		return $null;
	}

	mid = JCALL3(GetMethodID, jenv, cls, "toArray", TOARRAY);
	if (!mid)
		return $null;

	jarr = (##JNITYPE##Array)JCALL2(CallObjectMethod, jenv, $input, mid);
	carr = JCALL2(Get##JAVATYPE##ArrayElements, jenv, jarr, 0);
	array = SG_MALLOC(SGTYPE, cols);
	for (i = 0; i < cols; i++) {
		array[i] = (SGTYPE)carr[i];
	}

	JCALL3(Release##JAVATYPE##ArrayElements, jenv, jarr, carr, 0);

    $1 = shogun::SGVector<SGTYPE>((SGTYPE *)array, cols);
}

%typemap(out) shogun::SGVector<SGTYPE>
{
	int32_t rows = 1;
	int32_t cols = $1.vlen;
	JNITYPE* arr = SG_MALLOC(JNITYPE,cols);
	jobject res;
	int32_t i;

	jclass cls;
	jmethodID mid;

	cls = JCALL1(FindClass, jenv, CLASSDESC);
	if (!cls)
		return $null;

	mid = JCALL3(GetMethodID, jenv, cls, "<init>", CONSTRUCTOR);
	if (!mid)
		return $null;

	##JNITYPE##Array jarr = (##JNITYPE##Array)JCALL1(New##JAVATYPE##Array, jenv, cols);
	if (!jarr)
		return $null;

	for (i = 0; i < cols; i++) {
		arr[i] = (JNITYPE)$1.vector[i];
	}

	JCALL4(Set##JAVATYPE##ArrayRegion, jenv, jarr, 0, cols, arr);

	res = JCALL5(NewObject, jenv, cls, mid, rows, cols, jarr);
        SG_FREE(arr);
	$result = (jobject)res;
}

%typemap(javain) shogun::SGVector<SGTYPE> "$javainput"
%typemap(javaout) shogun::SGVector<SGTYPE> {
	return $jnicall;
}

%enddef

/*Define concrete examples of the TYPEMAP_SGVector macros */
TYPEMAP_SGVECTOR(bool, boolean, Double, jdouble, "()[D", "org/jblas/DoubleMatrix", "(II[D)V")
TYPEMAP_SGVECTOR(char, byte, Double, jdouble, "()[D", "org/jblas/DoubleMatrix", "(II[D)V")
TYPEMAP_SGVECTOR(uint8_t, byte, Double, jdouble, "()[D", "org/jblas/DoubleMatrix", "(II[D)V")
TYPEMAP_SGVECTOR(int16_t, short, Double, jdouble, "()[D", "org/jblas/DoubleMatrix", "(II[D)V")
TYPEMAP_SGVECTOR(uint16_t, int, Double, jdouble, "()[D", "org/jblas/DoubleMatrix", "(II[D)V")
TYPEMAP_SGVECTOR(int32_t, int, Double, jdouble, "()[D", "org/jblas/DoubleMatrix", "(II[D)V")
TYPEMAP_SGVECTOR(uint32_t, long, Double, jdouble, "()[D", "org/jblas/DoubleMatrix", "(II[D)V")
TYPEMAP_SGVECTOR(int64_t, int, Double, jdouble, "()[D", "org/jblas/DoubleMatrix", "(II[D)V")
TYPEMAP_SGVECTOR(uint64_t, long, Double, jdouble, "()[D", "org/jblas/DoubleMatrix", "(II[D)V")
TYPEMAP_SGVECTOR(long long, long, Double, jdouble, "()[D", "org/jblas/DoubleMatrix", "(II[D)V")
TYPEMAP_SGVECTOR(float32_t, float, Float, jfloat, "()[F", "org/jblas/FloatMatrix","(II[F)V")
TYPEMAP_SGVECTOR(float64_t, double, Double, jdouble, "()[D", "org/jblas/DoubleMatrix", "(II[D)V")

#undef TYPEMAP_SGVECTOR

#else
#ifdef HAVE_UJMP
/* Two dimensional input/output arrays */
%define TYPEMAP_SGVECTOR(SGTYPE, JTYPE, JAVATYPE, JNITYPE, TOARRAYMETHOD, TOARRAYDESC, CLASSDESC, CONSTRUCTOR)

%typemap(jni) shogun::SGVector<SGTYPE>		%{jobject%}
%typemap(jtype) shogun::SGVector<SGTYPE>		%{Matrix%}
%typemap(jstype) shogun::SGVector<SGTYPE>	%{Matrix%}

%typemap(in) shogun::SGVector<SGTYPE>
{
	jclass cls;
	jmethodID mid;
	SGTYPE *array;
	jobjectArray jobj;
	##JNITYPE##Array jarr;
	JNITYPE *carr;
	bool isVector;
	int32_t i, rows, cols;

	if (!$input) {
		SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null array");
		return $null;
	}

	cls = JCALL1(GetObjectClass, jenv, $input);
	if (!cls)
		return $null;

	mid = JCALL3(GetMethodID, jenv, cls, "isRowVector", "()Z");
	if (!mid)
		return $null;

	isVector = (int32_t)JCALL2(CallIntMethod, jenv, $input, mid);
	if (!isVector) {
		SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "vector expected");
		return $null;
	}

	mid = JCALL3(GetMethodID, jenv, cls, " getColumnCount", "()I");
	if (!mid)
		return $null;

	cols = (int32_t)JCALL2(CallIntMethod, jenv, $input, mid);
	if (cols < 1) {
		SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null vector");
		return $null;
	}

	mid = JCALL3(GetMethodID, jenv, cls, TOARRAYMETHOD, TOARRAYDESC);
	if (!mid)
		return $null;

	jobj = (jobjectArray)JCALL2(CallObjectMethod, jenv, $input, mid);
	jarr = (JNITYPE##Array)JCALL2(GetObjectArrayElement, jenv, jobj, 0);
	if (!jarr) {
		SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null array");
		return $null;
	}

	carr = JCALL2(Get##JAVATYPE##ArrayElements, jenv, jarr, 0);
	array = SG_MALLOC(SGTYPE, cols);
	if (!array) {
	SWIG_JavaThrowException(jenv, SWIG_JavaOutOfMemoryError, "array memory allocation failed");
	return $null;
	}

	for (i = 0; i < cols; i++) {
		array[j] = carr[i];
	}

	JCALL3(Release##JAVATYPE##ArrayElements, jenv, jarr, carr, 0);

	$1 = shogun::SGVector<SGTYPE>((SGTYPE *)array, cols);
}

%typemap(out) shogun::SGVector<SGTYPE>
{
	int32_t rows = 1;
	int32_t cols = $1.vlen;
	JNITYPE* arr = SG_MALLOC(JNITYPE,cols);
	jobject res;
	int32_t i;

	jclass cls;
	jmethodID mid;

	cls = JCALL1(FindClass, jenv, CLASSDESC);
	if (!cls)
		return $null;

	mid = JCALL3(GetMethodID, jenv, cls, "<init>", CONSTRUCTOR);
	if (!mid)
		return $null;

	##JNITYPE##Array jarr = (##JNITYPE##Array)JCALL1(New##JAVATYPE##Array, jenv, cols);
	if (!jarr)
		return $null;

	for (i = 0; i < cols; i++) {
		arr[i] = (JNITYPE)$1.vector[i];
	}

	JCALL4(Set##JAVATYPE##ArrayRegion, jenv, jarr, 0, cols, arr);

	res = JCALL5(NewObject, jenv, cls, mid, jarr, rows, cols);

        SG_FREE(arr);
	$result = (jobject)res;
}

%typemap(javain) shogun::SGVector<SGTYPE> "$javainput"
%typemap(javaout) shogun::SGVector<SGTYPE> {
	return $jnicall;
}

%enddef

/*Define concrete examples of the TYPEMAP_SGVECTOR macros */
TYPEMAP_SGVECTOR(bool, boolean, Boolean, jboolean, "toBooleanArray", "()[[Z", "org/ujmp/core/booleanmatrix/impl/DefaultDenseBooleanMatrix2D", "([BII)V")
TYPEMAP_SGVECTOR(char, byte, Byte, jbyte, "toByteArray", "()[[B", "org/ujmp/core/bytematrix/impl/DefaultDenseByteMatrix2D", "([BII)V")
TYPEMAP_SGVECTOR(uint8_t, byte, Byte, jbyte, "toByteArray", "()[[B", "org/ujmp/core/bytematrix/impl/DefaultDenseByteMatrix2D", "([BII)V")
TYPEMAP_SGVECTOR(int16_t, short, Short, jshort, "toShortArray", "()[[S", "org/ujmp/core/shortmatrix/impl/DefaultDenseShortMatrix2D", "([SII)V")
TYPEMAP_SGVECTOR(uint16_t, int, Int, jint, "toIntArray", "()[[I", "org/ujmp/core/intmatrix/impl/DefaultDenseIntMatrix2D", "([III)V")
TYPEMAP_SGVECTOR(int32_t, int, Int, jint, "toIntArray", "()[[I", "org/ujmp/core/intmatrix/impl/DefaultDenseIntMatrix2D", "([III)V")
TYPEMAP_SGVECTOR(uint32_t, long, Long, jlong, "toLongArray", "()[[J", "org/ujmp/core/longmatrix/impl/DefaultDenseLongMatrix2D", "([JII)V")
TYPEMAP_SGVECTOR(int64_t, int, Int, jint, "toIntArray", "()[[I", "org/ujmp/core/intmatrix/impl/DefaultDenseIntMatrix2D", "([III)V")
TYPEMAP_SGVECTOR(uint64_t, long, Long, jlong, "toLongArray", "()[[J", "org/ujmp/core/longmatrix/impl/DefaultDenseLongMatrix2D", "([JII)V")
TYPEMAP_SGVECTOR(long long, long, Long, jlong, "toLongArray", "()[[J", "org/ujmp/core/longmatrix/impl/DefaultDenseLongMatrix2D", "([JII)V")
TYPEMAP_SGVECTOR(float32_t, float, Float, jfloat, "toFloatArray", "()[[F", "org/ujmp/core/floatmatrix/impl/DefaultDenseFloatMatrix2D", "([FII)V")
TYPEMAP_SGVECTOR(float64_t, double, Double, jdouble, "toDoubleArray", "()[[D", "org/ujmp/core/doublematrix/impl/DefaultDenseDoubleMatrix2D", "([DII)V")

#undef TYPEMAP_SGVECTOR

/* Two dimensional input/output arrays */
%define TYPEMAP_SGVECTOR_REF(SGTYPE, JTYPE, JAVATYPE, JNITYPE, TOARRAYMETHOD, TOARRAYDESC, CLASSDESC, CONSTRUCTOR)

%typemap(jni) shogun::SGVector<SGTYPE>&, const shogun::SGVector<SGTYPE>&		%{jobject%}
%typemap(jtype) shogun::SGVector<SGTYPE>&, const shogun::SGVector<SGTYPE>&		%{Matrix%}
%typemap(jstype) shogun::SGVector<SGTYPE>&, const shogun::SGVector<SGTYPE>&	%{Matrix%}

%typemap(in) shogun::SGVector<SGTYPE>& (SGVector<SGTYPE> temp), const shogun::SGVector<SGTYPE>& (SGVector<SGTYPE> temp)
{
	jclass cls;
	jmethodID mid;
	SGTYPE *array;
	jobjectArray jobj;
	##JNITYPE##Array jarr;
	JNITYPE *carr;
	bool isVector;
	int32_t i, rows, cols;

	if (!$input) {
		SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null array");
		return $null;
	}

	cls = JCALL1(GetObjectClass, jenv, $input);
	if (!cls)
		return $null;

	mid = JCALL3(GetMethodID, jenv, cls, "isRowVector", "()Z");
	if (!mid)
		return $null;

	isVector = (int32_t)JCALL2(CallIntMethod, jenv, $input, mid);
	if (!isVector) {
		SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "vector expected");
		return $null;
	}

	mid = JCALL3(GetMethodID, jenv, cls, " getColumnCount", "()I");
	if (!mid)
		return $null;

	cols = (int32_t)JCALL2(CallIntMethod, jenv, $input, mid);
	if (cols < 1) {
		SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null vector");
		return $null;
	}

	mid = JCALL3(GetMethodID, jenv, cls, TOARRAYMETHOD, TOARRAYDESC);
	if (!mid)
		return $null;

	jobj = (jobjectArray)JCALL2(CallObjectMethod, jenv, $input, mid);
	jarr = (JNITYPE##Array)JCALL2(GetObjectArrayElement, jenv, jobj, 0);
	if (!jarr) {
		SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null array");
		return $null;
	}

	carr = JCALL2(Get##JAVATYPE##ArrayElements, jenv, jarr, 0);
	array = SG_MALLOC(SGTYPE, cols);
	if (!array) {
	SWIG_JavaThrowException(jenv, SWIG_JavaOutOfMemoryError, "array memory allocation failed");
	return $null;
	}

	for (i = 0; i < cols; i++) {
		array[j] = carr[i];
	}

	JCALL3(Release##JAVATYPE##ArrayElements, jenv, jarr, carr, 0);

	temp = shogun::SGVector<SGTYPE>((SGTYPE *)array, cols);
	$1 = &temp;
}

%typemap(javain) shogun::SGVector<SGTYPE>&, const shogun::SGVector<SGTYPE>& "$javainput"

%enddef

/*Define concrete examples of the TYPEMAP_SGVECTOR_REF macros */
TYPEMAP_SGVECTOR_REF(bool, boolean, Boolean, jboolean, "toBooleanArray", "()[[Z", "org/ujmp/core/booleanmatrix/impl/DefaultDenseBooleanMatrix2D", "([BII)V")
TYPEMAP_SGVECTOR_REF(char, byte, Byte, jbyte, "toByteArray", "()[[B", "org/ujmp/core/bytematrix/impl/DefaultDenseByteMatrix2D", "([BII)V")
TYPEMAP_SGVECTOR_REF(uint8_t, byte, Byte, jbyte, "toByteArray", "()[[B", "org/ujmp/core/bytematrix/impl/DefaultDenseByteMatrix2D", "([BII)V")
TYPEMAP_SGVECTOR_REF(int16_t, short, Short, jshort, "toShortArray", "()[[S", "org/ujmp/core/shortmatrix/impl/DefaultDenseShortMatrix2D", "([SII)V")
TYPEMAP_SGVECTOR_REF(uint16_t, int, Int, jint, "toIntArray", "()[[I", "org/ujmp/core/intmatrix/impl/DefaultDenseIntMatrix2D", "([III)V")
TYPEMAP_SGVECTOR_REF(int32_t, int, Int, jint, "toIntArray", "()[[I", "org/ujmp/core/intmatrix/impl/DefaultDenseIntMatrix2D", "([III)V")
TYPEMAP_SGVECTOR_REF(uint32_t, long, Long, jlong, "toLongArray", "()[[J", "org/ujmp/core/longmatrix/impl/DefaultDenseLongMatrix2D", "([JII)V")
TYPEMAP_SGVECTOR_REF(int64_t, int, Int, jint, "toIntArray", "()[[I", "org/ujmp/core/intmatrix/impl/DefaultDenseIntMatrix2D", "([III)V")
TYPEMAP_SGVECTOR_REF(uint64_t, long, Long, jlong, "toLongArray", "()[[J", "org/ujmp/core/longmatrix/impl/DefaultDenseLongMatrix2D", "([JII)V")
TYPEMAP_SGVECTOR_REF(long long, long, Long, jlong, "toLongArray", "()[[J", "org/ujmp/core/longmatrix/impl/DefaultDenseLongMatrix2D", "([JII)V")
TYPEMAP_SGVECTOR_REF(float32_t, float, Float, jfloat, "toFloatArray", "()[[F", "org/ujmp/core/floatmatrix/impl/DefaultDenseFloatMatrix2D", "([FII)V")
TYPEMAP_SGVECTOR_REF(float64_t, double, Double, jdouble, "toDoubleArray", "()[[D", "org/ujmp/core/doublematrix/impl/DefaultDenseDoubleMatrix2D", "([DII)V")

#undef TYPEMAP_SGVECTOR_REF

#endif
#endif

#ifdef HAVE_JBLAS
/* Two dimensional input/output arrays */
%define TYPEMAP_SGMATRIX(SGTYPE, JTYPE, JAVATYPE, JNITYPE, TOARRAY, CLASSDESC, CONSTRUCTOR)

%typemap(jni) shogun::SGMatrix<SGTYPE>		%{jobject%}
%typemap(jtype) shogun::SGMatrix<SGTYPE>		%{##JAVATYPE##Matrix%}
%typemap(jstype) shogun::SGMatrix<SGTYPE>	%{##JAVATYPE##Matrix%}

%typemap(in) shogun::SGMatrix<SGTYPE>
{
	jclass cls;
	jmethodID mid;
	SGTYPE *array;
	##JNITYPE##Array jarr;
	JNITYPE *carr;
	int32_t i,len, rows, cols;

	if (!$input) {
		SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null array");
		return $null;
	}

	cls = JCALL1(GetObjectClass, jenv, $input);
	if (!cls)
		return $null;

	mid = JCALL3(GetMethodID, jenv, cls, "toArray", TOARRAY);
	if (!mid)
		return $null;

	jarr = (##JNITYPE##Array)JCALL2(CallObjectMethod, jenv, $input, mid);
	carr = JCALL2(Get##JAVATYPE##ArrayElements, jenv, jarr, 0);
	len = JCALL1(GetArrayLength, jenv, jarr);
	array = SG_MALLOC(SGTYPE, len);
	for (i = 0; i < len; i++) {
		array[i] = (SGTYPE)carr[i];
	}

	JCALL3(Release##JAVATYPE##ArrayElements, jenv, jarr, carr, 0);

	mid = JCALL3(GetMethodID, jenv, cls, "getRows", "()I");
	if (!mid)
		return $null;

	rows = (int32_t)JCALL2(CallIntMethod, jenv, $input, mid);

	mid = JCALL3(GetMethodID, jenv, cls, "getColumns", "()I");
	if (!mid)
		return $null;

	cols = (int32_t)JCALL2(CallIntMethod, jenv, $input, mid);

	$1 = shogun::SGMatrix<SGTYPE>((SGTYPE*)array, rows, cols, true);
}

%typemap(out) shogun::SGMatrix<SGTYPE>
{
	int32_t rows = $1.num_rows;
	int32_t cols = $1.num_cols;
	int64_t len = int64_t(rows) * cols;
	JNITYPE* arr = SG_MALLOC(JNITYPE, len);
	jobject res;

	jclass cls;
	jmethodID mid;

	cls = JCALL1(FindClass, jenv, CLASSDESC);
	if (!cls)
		return $null;

	mid = JCALL3(GetMethodID, jenv, cls, "<init>", CONSTRUCTOR);
	if (!mid)
		return $null;

	##JNITYPE##Array jarr = (##JNITYPE##Array)JCALL1(New##JAVATYPE##Array, jenv, len);
	if (!jarr)
		return $null;

	for (int64_t i = 0; i < len; i++) {
		arr[i] = (JNITYPE)$1.matrix[i];
	}
	JCALL4(Set##JAVATYPE##ArrayRegion, jenv, jarr, 0, len, arr);

	res = JCALL5(NewObject, jenv, cls, mid, rows, cols, jarr);
        SG_FREE(arr);
	$result = (jobject)res;
}

%typemap(javain) shogun::SGMatrix<SGTYPE> "$javainput"
%typemap(javaout) shogun::SGMatrix<SGTYPE> {
	return $jnicall;
}

%enddef

/*Define concrete examples of the TYPEMAP_SGMATRIX macros */
TYPEMAP_SGMATRIX(bool, boolean, Double, jdouble, "()[D", "org/jblas/DoubleMatrix", "(II[D)V")
TYPEMAP_SGMATRIX(char, byte, Double, jdouble, "()[D", "org/jblas/DoubleMatrix", "(II[D)V")
TYPEMAP_SGMATRIX(uint8_t, byte, Double, jdouble, "()[D", "org/jblas/DoubleMatrix", "(II[D)V")
TYPEMAP_SGMATRIX(int16_t, short, Double, jdouble, "()[D", "org/jblas/DoubleMatrix", "(II[D)V")
TYPEMAP_SGMATRIX(uint16_t, int, Double, jdouble, "()[D", "org/jblas/DoubleMatrix", "(II[D)V")
TYPEMAP_SGMATRIX(int32_t, int, Double, jdouble, "()[D", "org/jblas/DoubleMatrix", "(II[D)V")
TYPEMAP_SGMATRIX(uint32_t, long, Double, jdouble, "()[D", "org/jblas/DoubleMatrix", "(II[D)V")
TYPEMAP_SGMATRIX(int64_t, int, Double, jdouble, "()[D", "org/jblas/DoubleMatrix", "(II[D)V")
TYPEMAP_SGMATRIX(uint64_t, long, Double, jdouble, "()[D", "org/jblas/DoubleMatrix", "(II[D)V")
TYPEMAP_SGMATRIX(long long, long, Double, jdouble, "()[D", "org/jblas/DoubleMatrix", "(II[D)V")
TYPEMAP_SGMATRIX(float32_t, float, Float, jfloat, "()[F", "org/jblas/FloatMatrix","(II[F)V")
TYPEMAP_SGMATRIX(float64_t, double, Double, jdouble, "()[D", "org/jblas/DoubleMatrix", "(II[D)V")

#undef TYPEMAP_SGMATRIX

#else
#ifdef HAVE_UJMP
/* Two dimensional input/output arrays */
%define TYPEMAP_SGMATRIX(SGTYPE, JTYPE, JAVATYPE, JNITYPE, TOARRAYMETHOD, TOARRAYDESC, CLASSDESC, CONSTRUCTOR)

%typemap(jni) shogun::SGMatrix<SGTYPE>		%{jobject%}
%typemap(jtype) shogun::SGMatrix<SGTYPE>		%{Matrix%}
%typemap(jstype) shogun::SGMatrix<SGTYPE>	%{Matrix%}

%typemap(in) shogun::SGMatrix<SGTYPE>
{
	jclass cls;
	jmethodID mid;
	SGTYPE *array;
	jobjectArray jobj;
	##JNITYPE##Array jarr;
	JNITYPE *carr;
	int32_t i, j, rows, cols;

	if (!$input) {
		SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null array");
		return $null;
	}

	cls = JCALL1(GetObjectClass, jenv, $input);
	if (!cls)
		return $null;

	mid = JCALL3(GetMethodID, jenv, cls, TOARRAYMETHOD, TOARRAYDESC);
	if (!mid)
		return $null;

	jobj = (jobjectArray)JCALL2(CallObjectMethod, jenv, $input, mid);
	rows = JCALL1(GetArrayLength, jenv, jobj);
	cols = 0;

	for (i = 0; i < rows; i++) {
		jarr = (JNITYPE##Array)JCALL2(GetObjectArrayElement, jenv, jobj, i);
		if (!jarr) {
			SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null array");
			return $null;
		}
		if (cols == 0) {
			cols = JCALL1(GetArrayLength, jenv, jarr);
			array = SG_MALLOC(SGTYPE, rows * cols);
			if (!array) {
			SWIG_JavaThrowException(jenv, SWIG_JavaOutOfMemoryError, "array memory allocation failed");
			return $null;
			}
		}
		carr = JCALL2(Get##JAVATYPE##ArrayElements, jenv, jarr, 0);
		for (j = 0; j < cols; j++) {
			array[i * cols + j] = carr[j];
		}
		JCALL3(Release##JAVATYPE##ArrayElements, jenv, jarr, carr, 0);
	}

	$1 = shogun::SGMatrix<SGTYPE>((SGTYPE*)array, rows, cols);
}

%typemap(out) shogun::SGMatrix<SGTYPE>
{
	int32_t rows = $1.num_rows;
	int32_t cols = $1.num_cols;
	int32_t len = rows * cols;
	JNITYPE* arr = SG_MALLOC(JNITYPE,len);
	jobject res;
	int32_t i;

	jclass cls;
	jmethodID mid;

	cls = JCALL1(FindClass, jenv, CLASSDESC);
	if (!cls)
		return $null;

	mid = JCALL3(GetMethodID, jenv, cls, "<init>", CONSTRUCTOR);
	if (!mid)
		return $null;

	##JNITYPE##Array jarr = (##JNITYPE##Array)JCALL1(New##JAVATYPE##Array, jenv, len);
	if (!jarr)
		return $null;

	for (i = 0; i < len; i++) {
		arr[i] = (JNITYPE)$1.matrix[i];
	}
	JCALL4(Set##JAVATYPE##ArrayRegion, jenv, jarr, 0, len, arr);

	res = JCALL5(NewObject, jenv, cls, mid, jarr, rows, cols);
        SG_FREE(arr);
	$result = (jobject)res;
}

%typemap(javain) shogun::SGMatrix<SGTYPE> "$javainput"
%typemap(javaout) shogun::SGMatrix<SGTYPE> {
	return $jnicall;
}

%enddef

/*Define concrete examples of the TYPEMAP_SGMATRIX macros */
TYPEMAP_SGMATRIX(bool, boolean, Boolean, jboolean, "toBooleanArray", "()[[Z", "org/ujmp/core/booleanmatrix/impl/DefaultDenseBooleanMatrix2D", "([BII)V")
TYPEMAP_SGMATRIX(char, byte, Byte, jbyte, "toByteArray", "()[[B", "org/ujmp/core/bytematrix/impl/DefaultDenseByteMatrix2D", "([BII)V")
TYPEMAP_SGMATRIX(uint8_t, byte, Byte, jbyte, "toByteArray", "()[[B", "org/ujmp/core/bytematrix/impl/DefaultDenseByteMatrix2D", "([BII)V")
TYPEMAP_SGMATRIX(int16_t, short, Short, jshort, "toShortArray", "()[[S", "org/ujmp/core/shortmatrix/impl/DefaultDenseShortMatrix2D", "([SII)V")
TYPEMAP_SGMATRIX(uint16_t, int, Int, jint, "toIntArray", "()[[I", "org/ujmp/core/intmatrix/impl/DefaultDenseIntMatrix2D", "([III)V")
TYPEMAP_SGMATRIX(int32_t, int, Int, jint, "toIntArray", "()[[I", "org/ujmp/core/intmatrix/impl/DefaultDenseIntMatrix2D", "([III)V")
TYPEMAP_SGMATRIX(uint32_t, long, Long, jlong, "toLongArray", "()[[J", "org/ujmp/core/longmatrix/impl/DefaultDenseLongMatrix2D", "([JII)V")
TYPEMAP_SGMATRIX(int64_t, int, Int, jint, "toIntArray", "()[[I", "org/ujmp/core/intmatrix/impl/DefaultDenseIntMatrix2D", "([III)V")
TYPEMAP_SGMATRIX(uint64_t, long, Long, jlong, "toLongArray", "()[[J", "org/ujmp/core/longmatrix/impl/DefaultDenseLongMatrix2D", "([JII)V")
TYPEMAP_SGMATRIX(long long, long, Long, jlong, "toLongArray", "()[[J", "org/ujmp/core/longmatrix/impl/DefaultDenseLongMatrix2D", "([JII)V")
TYPEMAP_SGMATRIX(float32_t, float, Float, jfloat, "toFloatArray", "()[[F", "org/ujmp/core/floatmatrix/impl/DefaultDenseFloatMatrix2D", "([FII)V")
TYPEMAP_SGMATRIX(float64_t, double, Double, jdouble, "toDoubleArray", "()[[D", "org/ujmp/core/doublematrix/impl/DefaultDenseDoubleMatrix2D", "([DII)V")

#undef TYPEMAP_SGMATRIX

#endif
#endif

/* input/output typemap for CStringFeatures */
%define TYPEMAP_STRINGFEATURES(SGTYPE, JTYPE, JAVATYPE, JNITYPE, JNIDESC, CLASSDESC)

%typemap(jni) shogun::SGStringList<SGTYPE>	%{jobjectArray%}
%typemap(jtype) shogun::SGStringList<SGTYPE>		%{JTYPE[][]%}
%typemap(jstype) shogun::SGStringList<SGTYPE>	%{JTYPE[][]%}

%typemap(in) shogun::SGStringList<SGTYPE> {
	int32_t size = 0;
	int32_t i;
	int32_t len, max_len = 0;

	if (!$input) {
		SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null array");
		return $null;
	}

	size = JCALL1(GetArrayLength, jenv, $input);
	shogun::SGString<SGTYPE>* strings=SG_MALLOC(shogun::SGString<SGTYPE>, size);

	for (i = 0; i < size; i++) {
		##JNITYPE##Array jarr = (##JNITYPE##Array)JCALL2(GetObjectArrayElement, jenv, $input, i);
		len = JCALL1(GetArrayLength, jenv, jarr);
		max_len = shogun::CMath::max(len, max_len);

		strings[i].slen=len;
		strings[i].string=NULL;

		if (len >0) {
			strings[i].string = SG_MALLOC(SGTYPE, len);
			memcpy(strings[i].string, jarr, len * sizeof(SGTYPE));
		}
	}

	SGStringList<SGTYPE> sl;
	sl.strings=strings;
	sl.num_strings=size;
	sl.max_string_length=max_len;
	$1 = sl;
}

%typemap(out) shogun::SGStringList<SGTYPE> {
	shogun::SGString<SGTYPE>* str = $1.strings;
	int32_t i, j, num = $1.num_strings;
	jclass cls;
	jobjectArray res;

	cls = JCALL1(FindClass, jenv, CLASSDESC);
	res = JCALL3(NewObjectArray, jenv, num, cls, NULL);

	for (i = 0; i < num; i++) {
		SGTYPE* data = SG_MALLOC(SGTYPE, str[i].slen);
		memcpy(data, str[i].string, str[i].slen * sizeof(SGTYPE));

		##JNITYPE##Array jarr = (##JNITYPE##Array)JCALL1(New##JAVATYPE##Array, jenv, str[i].slen);

		JNITYPE* arr = SG_MALLOC(JNITYPE, str[i].slen);
		for (j = 0; j < str[i].slen; j++) {
			arr[j] = (JNITYPE)data[j];
		}
		JCALL4(Set##JAVATYPE##ArrayRegion, jenv, jarr, 0, str[i].slen, arr);
		JCALL3(SetObjectArrayElement, jenv, res, i, jarr);
		JCALL1(DeleteLocalRef, jenv, jarr);

		SG_FREE(str[i].string);
		SG_FREE(arr);
	}
	SG_FREE(str);
	$result = res;
}

%typemap(javain) shogun::SGStringList<SGTYPE> "$javainput"
%typemap(javaout) shogun::SGStringList<SGTYPE> {
	return $jnicall;
}

%enddef

TYPEMAP_STRINGFEATURES(bool, boolean, Boolean, jboolean, "Boolen[][]", "[[Z")
TYPEMAP_STRINGFEATURES(uint8_t, byte, Byte, jbyte, "Byte[][]", "[[S")
TYPEMAP_STRINGFEATURES(int16_t, short, Short, jshort, "Short[][]", "[[S")
TYPEMAP_STRINGFEATURES(uint16_t, int, Int, jint, "Int[][]", "[[I")
TYPEMAP_STRINGFEATURES(int32_t, int, Int, jint, "Int[][]", "[[I")
TYPEMAP_STRINGFEATURES(uint32_t, long, Long, jlong, "Long[][]", "[[J")
TYPEMAP_STRINGFEATURES(int64_t, int, Int, jint, "Int[][]", "[[I")
TYPEMAP_STRINGFEATURES(uint64_t, long, Long, jlong, "Long[][]", "[[J")
TYPEMAP_STRINGFEATURES(long long, long, Long, jlong, "Long[][]", "[[J")
TYPEMAP_STRINGFEATURES(float32_t, float, Float, jfloat, "Float[][]", "[[F")
TYPEMAP_STRINGFEATURES(float64_t, double, Double, jdouble, "Doulbe[][]", "[[D")

#undef TYPEMAP_STRINGFEATURES

/* input/output typemap for SGStringList<char> */
%typemap(jni) shogun::SGStringList<char>	%{jobjectArray%}
%typemap(jtype) shogun::SGStringList<char>		%{String []%}
%typemap(jstype) shogun::SGStringList<char>	%{String []%}

%typemap(in) shogun::SGStringList<char> {
	int32_t size = 0;
	int32_t i;
	int32_t len, max_len = 0;

	if (!$input) {
		SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null array");
		return $null;
	}

	size = JCALL1(GetArrayLength, jenv, $input);
	shogun::SGString<char>* strings=SG_MALLOC(shogun::SGString<char>, size);

	for (i = 0; i < size; i++) {
		jstring jstr = (jstring)JCALL2(GetObjectArrayElement, jenv, $input, i);

		len = JCALL1(GetStringUTFLength, jenv, jstr);
		max_len = shogun::CMath::max(len, max_len);
		const char *str = (char *)JCALL2(GetStringUTFChars, jenv, jstr, 0);

		strings[i].slen = len;
		strings[i].string = NULL;

		if (len > 0) {
			strings[i].string = SG_MALLOC(char, len);
			memcpy(strings[i].string, str, len);
		}
		JCALL2(ReleaseStringUTFChars, jenv, jstr, str);
	}

	SGStringList<char> sl;
	sl.strings = strings;
	sl.num_strings = size;
	sl.max_string_length = max_len;
	$1 = sl;
}

%typemap(out) shogun::SGStringList<char> {
	shogun::SGString<char>* str = $1.strings;
	int32_t i, j, num = $1.num_strings;
	jclass cls;
	jobjectArray res;

	cls = JCALL1(FindClass, jenv, "java/lang/String");
	res = JCALL3(NewObjectArray, jenv, num, cls, NULL);

	for (i = 0; i < num; i++) {
		jstring jstr = (jstring)JCALL1(NewStringUTF, jenv, (char *)str[i].string);
		JCALL3(SetObjectArrayElement, jenv, res, i, jstr);
		JCALL1(DeleteLocalRef, jenv, jstr);
		SG_FREE(str[i].string);
	}
	SG_FREE(str);
	$result = res;
}

%typemap(javain) shogun::SGStringList<char> "$javainput"
%typemap(javaout) shogun::SGStringList<char> {
	return $jnicall;
}
