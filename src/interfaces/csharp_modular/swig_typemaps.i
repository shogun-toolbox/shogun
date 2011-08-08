/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 *  Written (W) 2011 Baozeng Ding
 *
 */

%define TYPEMAP_SGVECTOR(SGTYPE, CTYPE, CSHARPTYPE)

%typemap(ctype, out="CTYPE*") shogun::SGVector<SGTYPE>		%{int size, CTYPE*%}
%typemap(imtype, out="IntPtr", inattributes="int size, [MarshalAs(UnmanagedType.LPArray)]") shogun::SGVector<SGTYPE>		%{CSHARPTYPE[]%}
%typemap(cstype) shogun::SGVector<SGTYPE> 	%{CSHARPTYPE[]%}

%typemap(in) shogun::SGVector<SGTYPE> {
	int32_t i;
	SGTYPE *array;

	if (!$input) {
		SWIG_CSharpSetPendingException(SWIG_CSharpNullReferenceException, "null array");
		return $null;	
	}

	array = SG_MALLOC(SGTYPE, size);

	if (!array) {
		SWIG_CSharpSetPendingException(SWIG_CSharpOutOfMemoryException, "array memory allocation failed");
		return $null;
	}
	for (i = 0; i < size; i++) {
		array[i] = (SGTYPE)$input[i];	
	}
	
	$1 = shogun::SGVector<SGTYPE>((SGTYPE *)array, size);
}


%typemap(out) shogun::SGVector<SGTYPE> {

	int32_t i;
	CTYPE *res;
	int len = $1.vlen;

	res = SG_MALLOC(CTYPE, len + 1);
	res[0] = len;

	for (i = 0; i < len; i++) {
		res[i + 1] = (CTYPE) $1.vector[i];
	}

	$1.free_vector();

	$result = res;
}

%typemap(csin) shogun::SGVector<SGTYPE> "$csinput.Length, $csinput"
%typemap(csout) shogun::SGVector<SGTYPE> {
		IntPtr ptr = $imcall;$excode
		CSHARPTYPE[] size = new CSHARPTYPE[1];
		Marshal.Copy(ptr, size, 0, 1);

		int len = (int)size[0];

		CSHARPTYPE[] ret = new CSHARPTYPE[len];

		Marshal.Copy(new IntPtr(ptr.ToInt32() + Marshal.SizeOf(typeof(CSHARPTYPE))), ret, 0, len);
		return ret;
}
%enddef


TYPEMAP_SGVECTOR(char, signed char, byte)
TYPEMAP_SGVECTOR(uint8_t, unsigned char, byte)
TYPEMAP_SGVECTOR(int16_t, short, short)
TYPEMAP_SGVECTOR(uint16_t, unsigned short, short)
TYPEMAP_SGVECTOR(int32_t, int, int)
TYPEMAP_SGVECTOR(uint32_t, unsigned int, int)
TYPEMAP_SGVECTOR(int64_t, long, int)
TYPEMAP_SGVECTOR(uint64_t, unsigned long, long)
TYPEMAP_SGVECTOR(long long, long long, long)
TYPEMAP_SGVECTOR(float32_t, float, float)
TYPEMAP_SGVECTOR(float64_t, double, double)

#undef TYPEMAP_SGVECTOR

%define TYPEMAP_SGMATRIX(SGTYPE, CTYPE, CSHARPTYPE)

%typemap(ctype, out="CTYPE*") shogun::SGMatrix<SGTYPE>		%{int rows, int cols, CTYPE*%}
%typemap(imtype, out="IntPtr", inattributes="int rows, int cols, [MarshalAs(UnmanagedType.LPArray)]") shogun::SGMatrix<SGTYPE>		%{CSHARPTYPE[,]%}
%typemap(cstype) shogun::SGMatrix<SGTYPE> 	%{CSHARPTYPE[,]%}

%typemap(in) shogun::SGMatrix<SGTYPE>
{
	int32_t i,j;
	SGTYPE *array;   

	if (!$input) {
		SWIG_CSharpSetPendingException(SWIG_CSharpNullReferenceException, "null array");
		return $null;
	}
 
	array = SG_MALLOC(SGTYPE, rows * cols);
	if (!array) {
		SWIG_CSharpSetPendingException(SWIG_CSharpOutOfMemoryException, "array memory allocation failed");
		return $null;
	}

	for (i = 0; i < rows * cols; i++) {
		array[i] = (SGTYPE)$input[i];
	}

	$1 = shogun::SGMatrix<SGTYPE>(array, rows, cols);
}

%typemap(out) shogun::SGMatrix<SGTYPE>
{
	int32_t i;
	int32_t rows = $1.num_rows;
	int32_t cols = $1.num_cols;
	int32_t len = rows * cols;
	CTYPE *res;

	res = SG_MALLOC(CTYPE, len + 2);
	res[0] = rows;
	res[1] = cols;

	for (i = 0; i < len; i++) {
		res[i + 2] = (CTYPE) $1.matrix[i];
	}

	$1.free_matrix();

	$result = res;
}

%typemap(csin) shogun::SGMatrix<SGTYPE> "$csinput.GetLength(0), $csinput.GetLength(1), $csinput"
%typemap(csout) shogun::SGMatrix<SGTYPE> {
	IntPtr ptr = $imcall;$excode
	CSHARPTYPE[] ranks = new CSHARPTYPE[2];
	Marshal.Copy(ptr, ranks, 0, 2);

	int rows = (int)ranks[0];
	int cols = (int)ranks[1];
	int len = rows * cols;

	CSHARPTYPE[] ret = new CSHARPTYPE[len];

	Marshal.Copy(new IntPtr(ptr.ToInt32() + 2 * Marshal.SizeOf(typeof(CSHARPTYPE))), ret, 0, len);

	CSHARPTYPE[,] result = new CSHARPTYPE[rows, cols];
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result[i, j] = ret[i * cols + j];
		}
	}
	return result;
}
%enddef

TYPEMAP_SGMATRIX(char, signed char, byte)
TYPEMAP_SGMATRIX(uint8_t, unsigned char, byte)
TYPEMAP_SGMATRIX(int16_t, short, short)
TYPEMAP_SGMATRIX(uint16_t, unsigned short, short)
TYPEMAP_SGMATRIX(int32_t, int, int)
TYPEMAP_SGMATRIX(uint32_t, unsigned int, int)
TYPEMAP_SGMATRIX(int64_t, long, int)
TYPEMAP_SGMATRIX(uint64_t, unsigned long, long)
TYPEMAP_SGMATRIX(long long, long long, long)
TYPEMAP_SGMATRIX(float32_t, float, float)
TYPEMAP_SGMATRIX(float64_t, double, double)

#undef TYPEMAP_SGMATRIX
