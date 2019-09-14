/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

%typemap("rtypecheck") int, int &, long, long &
%{ (is.integer($arg) || is.numeric($arg)) && length($arg) == 1 && $arg == as.integer($arg) %}

%typemap(scoercein) float, float*, float &, double, double *, double &
%{ $input = as.numeric($input); %}

/* One dimensional input arrays */
%define TYPEMAP_SGVECTOR(sg_type, r_type, r_cast, r_type_string, condition)

%typemap(in) shogun::SGVector<sg_type>
{
    SEXP rvec = $input;
    int length = Rf_length(rvec);
    $1 = shogun::SGVector<sg_type>(length);

    SEXP coerced;
    PROTECT(coerced = Rf_coerceVector(rvec, r_type));
    std::copy_n(r_cast(coerced), length, $1.vector);
    UNPROTECT(1);
}

%typemap(freearg) shogun::SGVector<sg_type>
{
}

%typemap(out) shogun::SGVector<sg_type>
{
    sg_type* vec = $1.vector;
    int32_t len = $1.vlen;

    PROTECT($result = Rf_allocVector(r_type, len));
    std::copy_n(vec, len, r_cast($result));
    UNPROTECT(1);
}

%typemap("rtype") shogun::SGVector<sg_type>   r_type_string
%typemap("rtypecheck") shogun::SGVector<sg_type> %{ (condition) && is.vector($arg) %}

%typemap("scoerceout") shogun::SGVector<sg_type>
%{ %}

%enddef

TYPEMAP_SGVECTOR(bool, LGLSXP, LOGICAL_POINTER, "logical", is.logical($arg))
TYPEMAP_SGVECTOR(char, INTSXP, INTEGER_POINTER, "integer", is.integer($arg))
TYPEMAP_SGVECTOR(uint8_t, INTSXP, INTEGER_POINTER, "integer", is.integer($arg))
TYPEMAP_SGVECTOR(int16_t, INTSXP, INTEGER_POINTER, "integer", is.integer($arg))
TYPEMAP_SGVECTOR(uint16_t, INTSXP, INTEGER_POINTER, "integer", is.integer($arg))
TYPEMAP_SGVECTOR(int32_t, INTSXP, INTEGER_POINTER, "integer", is.integer($arg))
TYPEMAP_SGVECTOR(uint32_t, INTSXP, INTEGER_POINTER, "integer", is.integer($arg))
TYPEMAP_SGVECTOR(int64_t, INTSXP, INTEGER_POINTER, "integer", is.integer($arg))
TYPEMAP_SGVECTOR(uint64_t, INTSXP, INTEGER_POINTER, "integer", is.integer($arg))
TYPEMAP_SGVECTOR(float32_t, REALSXP, NUMERIC_POINTER, "numeric", is.numeric($arg))
TYPEMAP_SGVECTOR(float64_t, REALSXP, NUMERIC_POINTER, "numeric", is.numeric($arg))
TYPEMAP_SGVECTOR(floatmax_t, REALSXP, NUMERIC_POINTER, "numeric", is.numeric($arg))

#undef TYPEMAP_SGVECTOR

%define TYPEMAP_SGMATRIX(sg_type, r_type, r_cast, condition)

%typemap(in) shogun::SGMatrix<sg_type>
{
    SEXP rvec = $input;
    $1 = shogun::SGMatrix<sg_type>(Rf_nrows(rvec), Rf_ncols(rvec));

    SEXP coerced;
    PROTECT(coerced = Rf_coerceVector(rvec, r_type));
    std::copy_n(r_cast(coerced), $1.size(), $1.matrix);
    UNPROTECT(1);
}

%typemap(freearg) shogun::SGMatrix<sg_type>
{
}

%typemap(out) shogun::SGMatrix<sg_type>
{
    int32_t rows = $1.num_rows;
    int32_t cols = $1.num_cols;

    PROTECT($result = Rf_allocMatrix(r_type, rows, cols));
    std::copy_n($1.matrix, $1.size(), r_cast($result));
    UNPROTECT(1);
}

%typemap("rtype") shogun::SGMatrix<sg_type>   "matrix"
%typemap("rtypecheck") shogun::SGMatrix<sg_type> %{ (condition) && is.matrix($arg) %}

%typemap("scoerceout") shogun::SGMatrix<sg_type>
%{ %}

%enddef

TYPEMAP_SGMATRIX(bool, LGLSXP, LOGICAL_POINTER, is.logical($arg))
TYPEMAP_SGMATRIX(char, INTSXP, INTEGER_POINTER, is.integer($arg))
TYPEMAP_SGMATRIX(uint8_t, INTSXP, INTEGER_POINTER, is.integer($arg))
TYPEMAP_SGMATRIX(int16_t, INTSXP, INTEGER_POINTER, is.integer($arg))
TYPEMAP_SGMATRIX(uint16_t, INTSXP, INTEGER_POINTER, is.integer($arg))
TYPEMAP_SGMATRIX(int32_t, INTSXP, INTEGER_POINTER, is.integer($arg))
TYPEMAP_SGMATRIX(uint32_t, INTSXP, INTEGER_POINTER, is.integer($arg))
TYPEMAP_SGMATRIX(int64_t, INTSXP, INTEGER_POINTER, is.integer($arg))
TYPEMAP_SGMATRIX(uint64_t, INTSXP, INTEGER_POINTER, is.integer($arg))
TYPEMAP_SGMATRIX(float32_t, REALSXP, NUMERIC_POINTER, is.integer($arg) || is.numeric($arg))
TYPEMAP_SGMATRIX(float64_t, REALSXP, NUMERIC_POINTER, is.integer($arg) || is.numeric($arg))
TYPEMAP_SGMATRIX(floatmax_t, REALSXP, NUMERIC_POINTER, is.integer($arg) || is.numeric($arg))

#undef TYPEMAP_SGMATRIX

/* TODO INND ARRAYS */

/* input typemap for CStringFeatures<char> etc */
%define TYPEMAP_STRINGFEATURES(sg_type, r_type, r_cast)

%fragment(SWIG_AsVal_frag(std::vector<shogun::SGVector<sg_type>>), "header")
{
    int SWIG_AsVal_dec(std::vector<shogun::SGVector<sg_type>>)
        (const SEXP& obj, std::vector<shogun::SGVector<sg_type>>& strings)
    {
        unsigned int sexpsz = Rf_length(obj);
        strings.reserve(sexpsz);
        for (int listpos = 0; listpos < sexpsz; listpos++)
        {
            int vecsize = Rf_length(VECTOR_ELT(obj, listpos));
            strings.emplace_back(vecsize);
            std::copy_n(r_cast(VECTOR_ELT(obj, listpos)), vecsize, strings.back().vector);
        }
        return SWIG_OK;
    }
}

%fragment(SWIG_From_frag(std::vector<shogun::SGVector<sg_type>>), "header")
{
    SEXP SWIG_From_dec(std::vector<shogun::SGVector<sg_type>>)
        (const std::vector<shogun::SGVector<sg_type>>& vec_arr)
    {
        SEXP result;
        PROTECT(result = Rf_allocVector(VECSXP, vec_arr.size()));
        for (int pos = 0; pos < vec_arr.size(); pos++)
        {
            SET_VECTOR_ELT(result, pos, Rf_allocVector(r_type, vec_arr[pos].size()));
            std::copy_n(vec_arr[pos].vector, vec_arr[pos].size(), r_cast(VECTOR_ELT(result, pos)));
        }
        UNPROTECT(1);
        return result;
    }
}

%typemap("rtypecheck") std::vector<shogun::SGVector<sg_type>>, const std::vector<shogun::SGVector<sg_type>>&
%{ is.list($arg) && all(sapply($arg , is.integer) || sapply($arg, is.numeric)) %}

%typemap("scoerceout") std::vector<shogun::SGVector<sg_type>>, const std::vector<shogun::SGVector<sg_type>>&
%{ %}

%val_in_typemap(std::vector<shogun::SGVector<sg_type>>);
%val_out_typemap(std::vector<shogun::SGVector<sg_type>>);

%enddef

TYPEMAP_STRINGFEATURES(uint8_t, INTSXP, INTEGER_POINTER)
TYPEMAP_STRINGFEATURES(int16_t, INTSXP, INTEGER_POINTER)
TYPEMAP_STRINGFEATURES(uint16_t, INTSXP, INTEGER_POINTER)
TYPEMAP_STRINGFEATURES(int32_t, INTSXP, INTEGER_POINTER)
TYPEMAP_STRINGFEATURES(uint32_t, INTSXP, INTEGER_POINTER)
TYPEMAP_STRINGFEATURES(int64_t, INTSXP, INTEGER_POINTER)
TYPEMAP_STRINGFEATURES(uint64_t, INTSXP, INTEGER_POINTER)
TYPEMAP_STRINGFEATURES(float32_t, REALSXP, NUMERIC_POINTER)
TYPEMAP_STRINGFEATURES(float64_t, REALSXP, NUMERIC_POINTER)
TYPEMAP_STRINGFEATURES(floatmax_t, REALSXP, NUMERIC_POINTER)

#undef TYPEMAP_STRINGFEATURES_IN

%fragment(SWIG_AsVal_frag(std::vector<shogun::SGVector<char>>), "header")
{
    int SWIG_AsVal_dec(std::vector<shogun::SGVector<char>>)
        (const SEXP& obj, std::vector<shogun::SGVector<char>>& vec_arr)
    {
        const size_t MAX_LEN = 1 << 16;
        int sexpsz = Rf_length(obj);
        vec_arr.reserve(sexpsz);
        SEXP coerced;
        PROTECT(coerced = Rf_coerceVector(obj, STRSXP));
        for (int pos = 0; pos < sexpsz; pos++)
        {
            const char* cstr = CHAR(STRING_ELT(coerced, pos));
            size_t len = strnlen(cstr, MAX_LEN);
            if (len == MAX_LEN)
            {
                UNPROTECT(1);
                return SWIG_ERROR;
            }
            vec_arr.emplace_back(len);
            std::copy_n(cstr, len, vec_arr.back().vector);
        }
        UNPROTECT(1);
        return SWIG_OK;
    }
}

%fragment(SWIG_From_frag(std::vector<shogun::SGVector<char>>), "header")
{
    SEXP SWIG_From_dec(std::vector<shogun::SGVector<char>>)
        (const std::vector<shogun::SGVector<char>>& vec_arr)
    {
        SEXP result;
        PROTECT(result = Rf_allocVector(STRSXP, vec_arr.size()));
        for (size_t pos = 0; pos < vec_arr.size(); pos++)
        {
            std::stringstream ss;
            ss.write(vec_arr[pos].vector, vec_arr[pos].vlen);
            std::string str = ss.str();
            CHARACTER_POINTER(result)[pos] = Rf_mkChar(str.c_str());
        }
        UNPROTECT(1);
        return result;
    }
}

%typemap("scoerceout") std::vector<shogun::SGVector<char>>, const std::vector<shogun::SGVector<char>>&
%{ %}

%typemap("rtypecheck") std::vector<shogun::SGVector<char>>, const std::vector<shogun::SGVector<char>>&
%{ is.character($arg) %}

%val_in_typemap(std::vector<shogun::SGVector<char>>);
%val_out_typemap(std::vector<shogun::SGVector<char>>);
