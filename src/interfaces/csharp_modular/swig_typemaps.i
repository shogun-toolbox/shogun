/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Baozeng Ding
 */

%define TYPEMAP_SGVECTOR(SGTYPE, CTYPE, CSHARPTYPE)

%typemap(ctype, out="CTYPE*") shogun::SGVector<SGTYPE> %{int size_$1, CTYPE*%}
%typemap(imtype, out="IntPtr", inattributes="int size_$1, [MarshalAs(UnmanagedType.LPArray)]") shogun::SGVector<SGTYPE> %{CSHARPTYPE[]%}
%typemap(cstype) shogun::SGVector<SGTYPE> %{CSHARPTYPE[]%}

%typemap(in) shogun::SGVector<SGTYPE> {
	int32_t i;
	SGTYPE *array;

	if (!$input) {
		SWIG_CSharpSetPendingException(SWIG_CSharpNullReferenceException, "null array");
		return $null;
	}

	array = SG_MALLOC(SGTYPE, size_$1);

	if (!array) {
		SWIG_CSharpSetPendingException(SWIG_CSharpOutOfMemoryException, "array memory allocation failed");
		return $null;
	}

	for (i = 0; i < size_$1; i++) {
		array[i] = (SGTYPE)$input[i];
	}

	$1 = shogun::SGVector<SGTYPE>((SGTYPE *)array, size_$1);
}


%typemap(out) shogun::SGVector<SGTYPE> {

	int32_t i;
	char *res;
	int len = $1.vlen;

	res = SG_MALLOC(char, sizeof(CTYPE) * len + sizeof(int32_t));
	((int32_t*)res)[0] = len;

	CTYPE * dst = (CTYPE *)(res + sizeof(int32_t));

	for (i = 0; i < len; i++) {
		dst[i ] = (CTYPE) $1.vector[i];
	}

	$result = (CTYPE *)res;
}

%typemap(csin) shogun::SGVector<SGTYPE> "$csinput.Length, $csinput"
%typemap(csout, excode=SWIGEXCODE) shogun::SGVector<SGTYPE> {
	IntPtr ptr = $imcall;$excode

	int[] size = new int[1];
	Marshal.Copy(ptr, size, 0, 1);

	int len = size[0];

	CSHARPTYPE[] ret = new CSHARPTYPE[len];

	Marshal.Copy(new IntPtr(ptr.ToInt64() + Marshal.SizeOf(typeof(int))), ret, 0, len);
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

%typemap(ctype, out="CTYPE*") shogun::SGMatrix<SGTYPE> %{int rows_$1, int cols_$1, CTYPE*%}
%typemap(imtype, out="IntPtr", inattributes="int rows_$1, int cols_$1, [MarshalAs(UnmanagedType.LPArray)]") shogun::SGMatrix<SGTYPE> %{CSHARPTYPE[,]%}
%typemap(cstype) shogun::SGMatrix<SGTYPE> %{CSHARPTYPE[,]%}

%typemap(in) shogun::SGMatrix<SGTYPE>
{
	int32_t i,j;
	SGTYPE *array;

	if (!$input) {
		SWIG_CSharpSetPendingException(SWIG_CSharpNullReferenceException, "null array");
		return $null;
	}

	array = SG_MALLOC(SGTYPE, rows_$1 * cols_$1);
	if (!array) {
		SWIG_CSharpSetPendingException(SWIG_CSharpOutOfMemoryException, "array memory allocation failed");
		return $null;
	}

	for (i = 0; i < rows_$1 * cols_$1; i++) {
		array[i] = (SGTYPE)$input[i];
	}

	$1 = shogun::SGMatrix<SGTYPE>(array, rows_$1, cols_$1, true);
}

%typemap(out) shogun::SGMatrix<SGTYPE>
{
	int32_t i;
	int32_t rows = $1.num_rows;
	int32_t cols = $1.num_cols;
	int32_t len = rows * cols;
	char *res;

	res = SG_MALLOC(char, sizeof(CTYPE) * len + 2 * sizeof(int32_t));
	((int32_t*) res)[0] = rows;
	((int32_t*) res)[1] = cols;

	CTYPE *dst = (CTYPE *)(res + 2 * sizeof(int32_t));

	for (i = 0; i < len; i++) {
		dst[i] = (CTYPE) $1.matrix[i];
	}

	$result = (CTYPE *)res;
}

%typemap(csin) shogun::SGMatrix<SGTYPE> "$csinput.GetLength(0), $csinput.GetLength(1), $csinput"
%typemap(csout, excode=SWIGEXCODE) shogun::SGMatrix<SGTYPE> {
	IntPtr ptr = $imcall;$excode
	int[] ranks = new int[2];
	Marshal.Copy(ptr, ranks, 0, 2);

	int rows = ranks[0];
	int cols = ranks[1];
	int len = rows * cols;

	CSHARPTYPE[] ret = new CSHARPTYPE[len];

	Marshal.Copy(new IntPtr(ptr.ToInt64() + 2 * Marshal.SizeOf(typeof(int))), ret, 0, len);

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


/* input/output typemap for CStringFeatures */
%define TYPEMAP_STRINGFEATURES(SGTYPE, CTYPE, CSHARPTYPE)

%typemap(ctype, out="CTYPE*") shogun::SGStringList<SGTYPE> %{int rows_$1, int cols_$1, CTYPE*%}
%typemap(imtype, out="IntPtr", inattributes="int rows, int cols, [MarshalAs(UnmanagedType.LPArray)]") shogun::SGStringList<SGTYPE> %{CSHARPTYPE[,]%}
%typemap(cstype) shogun::SGStringList<SGTYPE> %{CSHARPTYPE[,]%}

%typemap(in) shogun::SGStringList<SGTYPE> {
	int32_t i;
	int32_t len, max_len = 0;
	CTYPE * array = $input;

	if (!$input) {
		SWIG_CSharpSetPendingException(SWIG_CSharpNullReferenceException, "null array");
		return $null;
	}

	shogun::SGString<SGTYPE>* strings=SG_MALLOC(shogun::SGString<SGTYPE>, rows_$1);

	for (i = 0; i < rows_$1; i++) {
		len = cols_$1;
		max_len = shogun::CMath::max(len, max_len);

		strings[i].slen = len;
		strings[i].string = NULL;

		if (len >0) {
			strings[i].string = SG_MALLOC(SGTYPE, len);
			memcpy(strings[i].string, array, len * sizeof(SGTYPE));
		}
		array = array + len;
	}

	SGStringList<SGTYPE> sl;
	sl.strings=strings;
	sl.num_strings=rows_$1;
	sl.max_string_length=max_len;
	$1 = sl;
}

%typemap(out) shogun::SGStringList<SGTYPE> {
	shogun::SGString<SGTYPE>* str = $1.strings;
	int32_t i, j;
	int32_t rows = $1.num_strings;
	int32_t cols = str[0].slen;
	int32_t len = rows * cols;

	CTYPE *res = SG_MALLOC(CTYPE, len + 2);
	res[0] = rows;
	res[1] = cols;

	res = res + 2;

	for (i = 0; i < rows; i++) {
		memcpy(res, str[i].string, str[i].slen * sizeof(SGTYPE));
		res = res + cols;
		SG_FREE(str[i].string);
	}
	SG_FREE(str);
	$result = res;
}

%typemap(csin) shogun::SGStringList<SGTYPE> "$csinput.GetLength(0), $csinput.GetLength(1), $csinput"
%typemap(csout, excode=SWIGEXCODE) shogun::SGStringList<SGTYPE> {
	IntPtr ptr = $imcall;$excode
	CSHARPTYPE[] ranks = new CSHARPTYPE[2];
	Marshal.Copy(ptr, ranks, 0, 2);

	int rows = (int)ranks[0];
	int cols = (int)ranks[1];
	int len = rows * cols;

	CSHARPTYPE[] ret = new CSHARPTYPE[len];

	Marshal.Copy(new IntPtr(ptr.ToInt64() + 2 * Marshal.SizeOf(typeof(CSHARPTYPE))), ret, 0, len);

	CSHARPTYPE[,] result = new CSHARPTYPE[rows, cols];
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result[i, j] = ret[i * cols + j];
		}
	}
	return result;
}

%enddef

//TYPEMAP_STRINGFEATURES(char, signed char, byte)
TYPEMAP_STRINGFEATURES(uint8_t, unsigned char, byte)
TYPEMAP_STRINGFEATURES(int16_t, short, short)
TYPEMAP_STRINGFEATURES(uint16_t, unsigned short, short)
TYPEMAP_STRINGFEATURES(int32_t, int, int)
TYPEMAP_STRINGFEATURES(uint32_t, unsigned int, int)
TYPEMAP_STRINGFEATURES(int64_t, long, int)
TYPEMAP_STRINGFEATURES(uint64_t, unsigned long, long)
TYPEMAP_STRINGFEATURES(long long, long long, long)
TYPEMAP_STRINGFEATURES(float32_t, float, float)
TYPEMAP_STRINGFEATURES(float64_t, double, double)

/* input/output typemap for SGStringList<char> */
%typemap(ctype, out="char **") shogun::SGStringList<char> %{int size_$1, char **%}
%typemap(imtype, out="IntPtr", inattributes="int size, [MarshalAs(UnmanagedType.LPArray)]") shogun::SGStringList<char> %{string []%}
%typemap(cstype) shogun::SGStringList<char> %{string []%}

%typemap(in) shogun::SGStringList<char> {
	int32_t i;
	int32_t len, max_len = 0;
	char * str;

	if (!$input) {
		SWIG_CSharpSetPendingException(SWIG_CSharpNullReferenceException, "null array");
		return $null;
	}

	shogun::SGString<char>* strings=SG_MALLOC(shogun::SGString<char>, size_$1);

	for (i = 0; i < size_$1; i++) {
		str = $input[i];
		len = strlen(str);
		max_len = shogun::CMath::max(len, max_len);

		strings[i].slen = len;
		strings[i].string = NULL;

		if (len > 0) {
			strings[i].string = SG_MALLOC(char, len);
			memcpy(strings[i].string, str, len);
		}
	}

	SGStringList<char> sl;
	sl.strings = strings;
	sl.num_strings = size_$1;
	sl.max_string_length = max_len;
	$1 = sl;
}

%typemap(out) shogun::SGStringList<char> {
	shogun::SGString<char>* str = $1.strings;
	int32_t i, j;
	int32_t size = $1.num_strings;
	int32_t max_size = 32;

	char ** res = SG_MALLOC(char*, size + 1);
	res[0] = SG_MALLOC(char, max_size);
	sprintf(res[0], "%d", size);

	for (i = 0; i < size; i++) {
		res[i + 1] = SG_MALLOC(char, str[i].slen);
		memcpy(res[i + 1], str[i].string, str[i].slen * sizeof(char));
	}
	$result = res;
}

%typemap(csin) shogun::SGStringList<char> "$csinput.Length, $csinput"
%typemap(csout, excode=SWIGEXCODE) shogun::SGStringList<char> {
	IntPtr ptr = $imcall;$excode

	IntPtr[] ranks = new IntPtr[1];
	Marshal.Copy(ptr, ranks, 0, 1);

	string len = Marshal.PtrToStringAnsi(ranks[0]);
	int size = Convert.ToInt32(len);
	IntPtr[] ptrarray = new IntPtr[size + 1];
	Marshal.Copy(ptr, ptrarray, 0, size + 1);

	string[] result = new string[size];
	for (int i = 0; i < size; i++) {
			result[i] = Marshal.PtrToStringAnsi(ptrarray[i + 1]);
	}

	Marshal.FreeCoTaskMem(ranks[0]);
	Marshal.FreeCoTaskMem(ptr);
	return result;
}
