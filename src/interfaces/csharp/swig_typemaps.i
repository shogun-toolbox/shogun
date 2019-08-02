/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
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

%typemap(ctype, out="CTYPE*") std::vector<shogun::SGVector<SGTYPE>>, std::vector<shogun::SGVector<SGTYPE>>& %{int rows_$1, int cols_$1, CTYPE*%}
%typemap(imtype, out="IntPtr", inattributes="int rows, int cols, [MarshalAs(UnmanagedType.LPArray)]") std::vector<shogun::SGVector<SGTYPE>>, std::vector<shogun::SGVector<SGTYPE>>& %{CSHARPTYPE[,]%}
%typemap(cstype) std::vector<shogun::SGVector<SGTYPE>>, std::vector<shogun::SGVector<SGTYPE>>& %{CSHARPTYPE[,]%}

%typemap(in) std::vector<shogun::SGVector<SGTYPE>>& {
	$1 = nullptr;
	int32_t i;
	int32_t len, max_len = 0;
	CTYPE * array = $input;

	if (!$input) {
		SWIG_CSharpSetPendingException(SWIG_CSharpNullReferenceException, "null array");
		return $null;
	}

	std::vector<shogun::SGVector<SGTYPE>> strings;
	strings.reserve(rows_$1);

	for (i = 0; i < rows_$1; i++) {
		len = cols_$1;

		strings.emplace_back(len);
		sg_memcpy(strings.back().vector, array, len * sizeof(SGTYPE));

		array = array + len;
	}

	$1 = new std::vector<shogun::SGVector<SGTYPE>>(std::move(strings));
}

%typemap(freearg) std::vector<shogun::SGVector<SGTYPE>>& {
    delete $1;
}

%typemap(out) std::vector<shogun::SGVector<SGTYPE>> {
	std::vector<shogun::SGVector<SGTYPE>>& strings = $1;
	int32_t rows = strings.size();
	int32_t cols = 0;
	for(auto& str : strings)
		cols = std::max(cols, str.vlen);
	int32_t len = rows * cols;

	CTYPE *res = SG_MALLOC(CTYPE, len + 2);
	res[0] = rows;
	res[1] = cols;

	res = res + 2;

	for (auto& str : strings) {
		sg_memcpy(res, str.vector, str.vlen * sizeof(SGTYPE));
		res = res + cols;
	}
	$result = res;
}

%typemap(csin) std::vector<shogun::SGVector<SGTYPE>>, std::vector<shogun::SGVector<SGTYPE>>& "$csinput.GetLength(0), $csinput.GetLength(1), $csinput"
%typemap(csout, excode=SWIGEXCODE) std::vector<shogun::SGVector<SGTYPE>>, std::vector<shogun::SGVector<SGTYPE>>& {
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

/* input/output typemap for std::vector<shogun::SGVector<SGTYPE>><char> */
%typemap(ctype, out="char **") std::vector<shogun::SGVector<char>>, std::vector<shogun::SGVector<char>>& %{int size_$1, char **%}
%typemap(imtype, out="IntPtr", inattributes="int size, [MarshalAs(UnmanagedType.LPArray)]") std::vector<shogun::SGVector<char>>, std::vector<shogun::SGVector<char>>& %{string []%}
%typemap(cstype) std::vector<shogun::SGVector<char>>, std::vector<shogun::SGVector<char>>& %{string []%}

%typemap(in) std::vector<shogun::SGVector<char>>& {
	$1 = nullptr;
	int32_t i;
	int32_t len, max_len = 0;
	char * str;

	if (!$input) {
		SWIG_CSharpSetPendingException(SWIG_CSharpNullReferenceException, "null array");
		return $null;
	}

	std::vector<shogun::SGVector<char>> strings;
	strings.reserve(size_$1);

	for (i = 0; i < size_$1; i++) {
		str = $input[i];
		len = strlen(str);

		strings.emplace_back(len);

		sg_memcpy(strings.back().vector, str, len);
	}

	$1 = new std::vector<shogun::SGVector<char>>(std::move(strings));
}

%typemap(freearg) std::vector<shogun::SGVector<char>>& {
    delete $1;
}

%typemap(out) std::vector<shogun::SGVector<char>> {
	std::vector<shogun::SGVector<char>>& strings = $1;
	int32_t size = strings.size();
	int32_t max_size = 0;
	for(auto& str : strings)
		max_size = std::max(max_size, str.vlen);

	char ** res = SG_MALLOC(char*, size + 1);
	res[0] = SG_MALLOC(char, max_size);
	sprintf(res[0], "%d", size);
	
	for (int i = 0; i < size; i++) {
		res[i + 1] = SG_MALLOC(char, strings[i].vlen + 1);
		sg_memcpy(res[i + 1], strings[i].vector, strings[i].vlen * sizeof(char));
		
		// null terminate string as C#'s Marshal.PtrToStringAnsi expects that
		res[i+1][strings[i].vlen] = '\0';
	}
	$result = res;
}

%typemap(csin) std::vector<shogun::SGVector<char>>, std::vector<shogun::SGVector<char>>& "$csinput.Length, $csinput"
%typemap(csout, excode=SWIGEXCODE) std::vector<shogun::SGVector<char>>, std::vector<shogun::SGVector<char>>& {
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
