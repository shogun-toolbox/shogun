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

TYPEMAP_SGVECTOR(bool, bool, byte)
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

	for (int32_t i = 0; i < rows_$1 * cols_$1; i++) {
		array[i] = (SGTYPE)$input[i];
	}

	$1 = shogun::SGMatrix<SGTYPE>(array, rows_$1, cols_$1, true);
}

%typemap(out) shogun::SGMatrix<SGTYPE>
{
	int32_t rows = $1.num_rows;
	int32_t cols = $1.num_cols;
	int32_t len = rows * cols;
	char *res;

	res = SG_MALLOC(char, sizeof(CTYPE) * len + 2 * sizeof(int32_t));
	((int32_t*) res)[0] = rows;
	((int32_t*) res)[1] = cols;

	CTYPE *dst = (CTYPE *)(res + 2 * sizeof(int32_t));

	for (int32_t i = 0; i < len; i++) {
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

TYPEMAP_SGMATRIX(bool, bool, byte)
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

%pragma(csharp) modulecode=%{
	public static int[] __get_dims<T>(T[][] arr)
	{
		int[] dims = new int[arr.Length];
		for(int i = 0; i < arr.Length; i++)
			dims[i] = arr[i].Length;
		return dims;
	}

	public static T[] __flatten<T>(T[][] arr)
	{
		int len = 0;
		for(int i = 0; i < arr.Length; i++)
			len += arr[i].Length;

		T[] flat_arr = new T[len];
		int index = 0;
		for(int i = 0; i < arr.Length; i++)
		{
			for(int j = 0; j < arr[i].Length; j++)
			{
				flat_arr[index] = arr[i][j];
				index++;
			}
		}
		return flat_arr;
	}
%}

/* input/output typemap for CStringFeatures */
%define TYPEMAP_STRINGFEATURES(SGTYPE, CTYPE, CSHARPTYPE)

%typemap(ctype, out="CTYPE*") std::vector<shogun::SGVector<SGTYPE>> %{int rows_$1, int* cols_$1, CTYPE*%}
%typemap(imtype, out="IntPtr", inattributes="int rows, [MarshalAs(UnmanagedType.LPArray)] int[] cols, [MarshalAs(UnmanagedType.LPArray)]") std::vector<shogun::SGVector<SGTYPE>> %{CSHARPTYPE[]%}
%typemap(cstype) std::vector<shogun::SGVector<SGTYPE>> %{CSHARPTYPE[][]%}

%fragment(SWIG_AsVal_frag(std::vector<shogun::SGVector<SGTYPE>>), "header")
{
    int SWIG_AsVal_dec(std::vector<shogun::SGVector<SGTYPE>>)
		(int rows, int* cols, CTYPE* array, std::vector<shogun::SGVector<SGTYPE>>& strings)
    {
		if (!array) {
			SWIG_CSharpSetPendingException(SWIG_CSharpNullReferenceException, "null array");
			return SWIG_ERROR;
		}

		strings.reserve(rows);

		for (int32_t i = 0; i < rows; i++) {
			strings.emplace_back(cols[i]);
			sg_memcpy(strings.back().vector, array, cols[i] * sizeof(SGTYPE));

			array = array + cols[i];
		}
		return SWIG_OK;
    }
}

%fragment(SWIG_From_frag(std::vector<shogun::SGVector<SGTYPE>>), "header")
{
    CTYPE* SWIG_From_dec(std::vector<shogun::SGVector<SGTYPE>>)
    	(const std::vector<shogun::SGVector<SGTYPE>>& strings)
    {
		int32_t rows = strings.size();

		int32_t len = 0;
		for(auto& str : strings)
			len += str.vlen;

		char *res = SG_MALLOC(char, len * sizeof(CTYPE) + (rows + 1) * sizeof(int32_t));

		((int32_t *) res)[0] = rows;
		for(int32_t i = 0; i < rows; i++)
			((int32_t *) res)[i + 1] = strings[i].size();

		CTYPE *data = (CTYPE *)((int32_t *) res + rows + 1);

		for (auto& str : strings) {
			sg_memcpy(data, str.vector, str.vlen * sizeof(SGTYPE));
			data = data + str.size();
		}
		return (CTYPE *) res;
    }
}

%typemap(csin) std::vector<shogun::SGVector<SGTYPE>> "$csinput.Length, shogun.__get_dims($csinput), shogun.__flatten($csinput)"
%typemap(csout, excode=SWIGEXCODE) std::vector<shogun::SGVector<SGTYPE>> {
	IntPtr ptr = $imcall;$excode
	Int32[] rows_ = new Int32[1];
	Marshal.Copy(ptr, rows_, 0, 1);
	Int32 rows = rows_[0];

	Int32[] lengths = new Int32[rows];
	ptr = new IntPtr(ptr.ToInt64() + 1 * Marshal.SizeOf(typeof(int)));
	Marshal.Copy(ptr, lengths, 0, rows);
	ptr = new IntPtr(ptr.ToInt64() + rows * Marshal.SizeOf(typeof(int)));

	CSHARPTYPE[][] result = new CSHARPTYPE[rows][];
	for (int i = 0; i < rows; i++) {
		result[i] = new CSHARPTYPE[lengths[i]];
		Marshal.Copy(ptr, result[i], 0, lengths[i]);
		ptr = new IntPtr(ptr.ToInt64() + (lengths[i]) * Marshal.SizeOf(typeof(CSHARPTYPE)));
	}
	return result;
}

%apply std::vector<shogun::SGVector<SGTYPE>> { const std::vector<shogun::SGVector<SGTYPE>>& };
%val_in_typemap_with_args(std::vector<shogun::SGVector<SGTYPE>>, rows_$1, cols_$1);
%val_out_typemap(std::vector<shogun::SGVector<SGTYPE>>);

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

/* input/output typemap for std::vector<shogun::SGVector<<char>> */
%typemap(ctype, out="char **") std::vector<shogun::SGVector<char>> %{int size_$1, char **%}
%typemap(imtype, out="IntPtr", inattributes="int size, [MarshalAs(UnmanagedType.LPArray)]") std::vector<shogun::SGVector<char>> %{string []%}
%typemap(cstype) std::vector<shogun::SGVector<char>> %{string []%}

%fragment(SWIG_AsVal_frag(std::vector<shogun::SGVector<char>>), "header")
{
    int SWIG_AsVal_dec(std::vector<shogun::SGVector<char>>)
    	(int size, char** array, std::vector<shogun::SGVector<char>>& strings)
    {
		if (!array) {
			SWIG_CSharpSetPendingException(SWIG_CSharpNullReferenceException, "null array");
			return SWIG_ERROR;
		}

		strings.reserve(size);

		for (int32_t i = 0; i < size; i++) {
			char * str = array[i];
			int32_t len = strlen(str);

			strings.emplace_back(len);

			sg_memcpy(strings.back().vector, str, len);
		}

		return SWIG_OK;
    }
}

%fragment(SWIG_From_frag(std::vector<shogun::SGVector<char>>), "header")
{
    char** SWIG_From_dec(std::vector<shogun::SGVector<char>>)
    	(const std::vector<shogun::SGVector<char>>& strings)
    {
		int32_t size = strings.size();

		char ** res = SG_MALLOC(char*, size + 1);
		res[0] = (char *) SG_MALLOC(int32_t, 1);
		*((int32_t *) res[0]) = size;
		
		for (int i = 0; i < size; i++) {
			res[i + 1] = SG_MALLOC(char, strings[i].vlen + 1);
			sg_memcpy(res[i + 1], strings[i].vector, strings[i].vlen * sizeof(char));
			
			// null terminate string as C#'s Marshal.PtrToStringAnsi expects that
			res[i+1][strings[i].vlen] = '\0';
		}

		return res;
    }
}

%typemap(csin) std::vector<shogun::SGVector<char>> "$csinput.Length, $csinput"
%typemap(csout, excode=SWIGEXCODE) std::vector<shogun::SGVector<char>> {
	IntPtr ptr = $imcall;$excode

	IntPtr[] ranks = new IntPtr[1];
	Marshal.Copy(ptr, ranks, 0, 1);

	Int32[] size_arr = new Int32[1];
	Marshal.Copy(ranks[0], size_arr, 0, 1);
	Int32 size = size_arr[0];

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

%apply std::vector<shogun::SGVector<char>> { const std::vector<shogun::SGVector<char>>& }
%val_in_typemap_with_args(std::vector<shogun::SGVector<char>>, size_$1);
%val_out_typemap(std::vector<shogun::SGVector<char>>);
