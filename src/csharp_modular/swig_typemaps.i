
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 *  Written by Daniel Korn
 *  Based upon code by Baozeng Ding for the modular java interface
 *
 */

%define TYPEMAP_SGVECTOR(SGTYPE, CTYPE, CSHARPTYPE)

%typemap(ctype) shogun::SGVector<SGTYPE>		%{CTYPE*%}         // ctype is the C# equivalent of the javatypemap jni
%typemap(imtype) shogun::SGVector<SGTYPE>		%{CSHARPTYPE[]%}   // imtype is the C# equivalent of the java typemap jtype
%typemap(cstype) shogun::SGVector<SGTYPE> 	%{CSHARPTYPE[]%}           // cstype is the C# equivalent of the java typemap cstype

%typemap(in) shogun::SGVector<SGTYPE> (CSHARPTYPE *jarr) {
	int32_t i, len;
	SGTYPE *array;
	if (!$input) {
		SWIG_CSharpSetPendingException(SWIG_CSharpNullReferenceException, "null array");
		return $null;	
	}
	
	len = ((sizeof($input)) / (sizeof($input[0])));		

	array = new SGTYPE[len];

	if (!array) {
		SWIG_CSharpSetPendingException(SWIG_CSharpOutOfMemoryException, "array memory allocation failed");
		return $null;
	}
	for (i = 0; i < len; i++) {
		array[i] = jarr[i];	
	}
	
	$1 = shogun::SGVector<SGTYPE>((SGTYPE *)array, len);
}


%typemap(out) shogun::SGVector<SGTYPE> {

	int32_t i;
	CSHARPTYPE res[$1.vlen];

	for (i=0; i < $1.vlen; i++)
		res[i] = (CSHARPTYPE)$1.vector[i];

	if (!res)
		return NULL;
	
	$result = res;
}
%enddef


/* Define concrete examples of the TYPEMAP_SGVECTOR macros */
TYPEMAP_SGVECTOR(bool, bool, unsigned int)
TYPEMAP_SGVECTOR(char, char, char)
TYPEMAP_SGVECTOR(uint8_t, uint8_t, unsigned char)
TYPEMAP_SGVECTOR(int16_t, short, short)
TYPEMAP_SGVECTOR(uint16_t, ushort, unsigned short)
TYPEMAP_SGVECTOR(int32_t, int, int)
TYPEMAP_SGVECTOR(uint32_t, uint, unsigned int)
TYPEMAP_SGVECTOR(int64_t, int64_t, long)
TYPEMAP_SGVECTOR(uint64_t, uint64_t, unsigned long)
TYPEMAP_SGVECTOR(long long, long, long long)
TYPEMAP_SGVECTOR(float32_t, float, float)
TYPEMAP_SGVECTOR(float64_t, double, double)

#undef TYPEMAP_SGVECTOR


////////////////////////////////// Vector Typemaps - End /////////////////////////////////////////////


/////////////////////////////////  SGMATRIX Typemaps - Begin ////////////////////////////////////////

/* Two dimensional input/output arrays */
//%define TYPEMAP_SGMATRIX(SGTYPE, JTYPE, JAVATYPE, JNITYPE, TOARRAY, CLASSDESC, CONSTRUCTOR)
%define TYPEMAP_SGMATRIX(SGTYPE, CTYPE, CSHARPTYPE)

%typemap(ctype) shogun::SGMatrix<SGTYPE> %{CSHARPTYPE**%}  //  CTYPE
%typemap(imtype) shogun::SGMatrix<SGTYPE> %{CSHARPTYPE[][]%}
%typemap(cstype) shogun::SGMatrix<SGTYPE> %{CSHARPTYPE[][]%}

%typemap(in) shogun::SGMatrix<SGTYPE>
{


    int32_t i,j;
    int32_t rows, cols;
    SGTYPE *array;   
    //##JNITYPE##Array jarr;
    //JNITYPE *carr;
    //JNITYPE *element;

    if (!$input || !$input[0]) {
        SWIG_CSharpSetPendingException(SWIG_CSharpNullReferenceException, "null array");
        return $null;
    }
   
   
    rows = (sizeof($input) / sizeof($input[0])); //  Array First Dimmension Length
    cols = (sizeof($input[0]) / sizeof($input[0][0])); // Array Second Dimension Length
   
    for (i = 1; i < rows; i++){
   
        if (!$input[i]){
            SWIG_CSharpSetPendingException(SWIG_CSharpNullReferenceException, "null array");
            return $null;
        }
   
        if (sizeof($input[0]) != sizeof($input[i])){
            SWIG_CSharpSetPendingException(SWIG_CSharpNullReferenceException, "inconsistent array collum length");
            return $null;
        }       
    }
   
    array = new SGTYPE[rows * cols];
   
    if (!array) {
        SWIG_CSharpSetPendingException(SWIG_CSharpOutOfMemoryException, "array memory allocation failed");
        return $null;
    }
   
   
       
    for (i = 0; i < rows; i++) {
            for (j = 0; j < cols; j++) {
                    array[(i * rows) + j] = (SGTYPE)$input[i][j];
            }
    }   

    $1 = shogun::SGMatrix<SGTYPE>(array, rows, cols);
}

%typemap(out) shogun::SGMatrix<SGTYPE>
{

    int32_t i, j;
    int32_t rows = $1.num_rows;
    int32_t cols = $1.num_cols;
    int32_t len = rows * cols;
    CSHARPTYPE array[rows][cols];
   
    if ((rows < 1) || (cols < 1)){
        SWIG_CSharpSetPendingException(SWIG_CSharpNullReferenceException, "null array");
        return $null;
    }
   
    if (!array) {
        SWIG_CSharpSetPendingException(SWIG_CSharpOutOfMemoryException, "array memory allocation failed");
        return $null;
    }



    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++){
            array[i][j] = (CSHARPTYPE)($1.matrix[(i * rows) + j]);
	}
    }
   
    $result = (CSHARPTYPE **)array;

//  Translation Point

}

%enddef

/*Define concrete examples of the TYPEMAP_SGMATRIX macros */

TYPEMAP_SGMATRIX(bool, bool, unsigned int)
TYPEMAP_SGMATRIX(char, char, char)
TYPEMAP_SGMATRIX(uint8_t, uint8_t, unsigned char)
TYPEMAP_SGMATRIX(int16_t, short, short)
TYPEMAP_SGMATRIX(uint16_t, ushort, unsigned short)
TYPEMAP_SGMATRIX(int32_t, int, int)
TYPEMAP_SGMATRIX(uint32_t, uint, unsigned int)
TYPEMAP_SGMATRIX(int64_t, int, long)
TYPEMAP_SGMATRIX(uint64_t, uint, unsigned long)
TYPEMAP_SGMATRIX(long long, long, long long)
TYPEMAP_SGMATRIX(float32_t, float, float)
TYPEMAP_SGMATRIX(float64_t, double, double)

#undef TYPEMAP_SGMATRIX

/////////////////////////////////  SGMATRIX Typemaps - End //////////////////////////////////////////
 
 
 /* One dimensional input arrays */
 
/*  BEGIN DEFINITION OF TYPEMAP_IN1 MACRO  "define TYPEMAP_IN1" - Comment - Daniel Korn
 *                                         (SGTYPE, JTYPE, JAVATYPE, JNITYPE) - macro input paraters
 *  EXPLINATION OF SWIG MACROS 
 *  The primary purpose of %define is to define large macros of code. Unlike normal C preprocessor macros, it is not necessary to 
 *  terminate each line with a continuation character (\)--the macro definition extends to the first occurrence of %enddef. 
 *  Furthermore, when such macros are expanded, they are reparsed through the C preprocessor. Thus, SWIG macros can contain all other 
 *  preprocessor directives except for nested %define statements.
 *
 */
//%define TYPEMAP_IN1(SGTYPE, JTYPE, JAVATYPE, JNITYPE)
%define TYPEMAP_IN1(SGTYPE, CTYPE, CSHARPTYPE) //  - JAVATYPE is removed from the CSHARP marco because it is used for the JCALLX command not supported in CSharp



//  jni typemap - JNI C types. These provide the default mapping of types from C/C++ to JNI for use in the JNI (C/C++) code.
//  x ## y Concatenates x and y together to form xy. - this is used by JNITYPE##Array
//  This is a Multi-argument typemap
//  A multi-argument typemap is a conversion rule that specifies how to convert a single object in the target language 
//  to a set of consecutive function arguments in C/C++

// OLD CODE // %typemap(jni) (SGTYPE* IN_ARRAY1, int32_t DIM1)		%{JNITYPE##Array%}
// ctype is the equivalent typemap jni
%typemap(ctype) (SGTYPE* IN_ARRAY1, int32_t DIM1) %{CSHARPTYPE *%} 



//  jtype typemap - Java intermediary types. These provide the default mapping of types from C/C++ to Java for use in the 
//  native functions in the intermediary JNI class. The type must be the equivalent Java type for the JNI C type specified in the "jni" typemap.
//  This is a multiargument typemap
//  OLD CODE //  %typemap(jtype) (SGTYPE* IN_ARRAY1, int32_t DIM1)		%{JTYPE[]%}
%typemap(imtype) (SGTYPE* IN_ARRAY1, int32_t DIM1)		%{CTYPE[]%}



//  jstype typemap - Java types. These provide the default mapping of types from C/C++ to Java for use in the Java module class, proxy 
//  classes and type wrapper classes.
//  This is a multiargument typemap
//  OLD CODE //  %typemap(jstype) (SGTYPE* IN_ARRAY1, int32_t DIM1)	%{JTYPE[]%}
%typemap(cstype) (SGTYPE* IN_ARRAY1, int32_t DIM1)	%{CTYPE[]%}

//  OLD CODE //  %typemap(in) (SGTYPE* IN_ARRAY1, int32_t DIM1) (JNITYPE *jarr) {  
%typemap(in) (SGTYPE* IN_ARRAY1, int32_t DIM1) (CSHARPTYPE* carr) {  
	
	SGTYPE *array;
	int32_t len, i;

	//  Check that the input array is not null, and the size of array items is not null
	if (!$input || (sizeof($input[0]) == 0)) {
		//SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null array");
		SWIG_CSharpSetPendingException(SWIG_CSharpNullReferenceException, "null array");
		return $null;	
	}

	len = (sizeof($1) / (sizeof($1[0])));

	array = new SGTYPE[len];

	for (i = 1; i < len; i++)
		array[i] = (SGTYPE)carr[i];

	
	//  The default C# wrapping treats arrays as pointers, so we can assign the input array directly to the SGTYPE*
	$1 = array;
	$2 = len;
	
	// if this fails, throw a memory exception
	if (!$1) {
		SWIG_CSharpSetPendingException(SWIG_CSharpOutOfMemoryException, "array memory allocation failed");
		return $null;
	}

	

}



%typemap(out) (SGTYPE* IN_ARRAY1, int32_t DIM1) {

	CSHARPTYPE arr[$2];
	int i;
	
	if (!arr)
		return NULL;
	
	for (i=0; i < $2; i++)
		arr[i] = (CSHARPTYPE)$1[i];
		
	$result = arr;
}

%typemap(freearg) (SGTYPE* IN_ARRAY1, int32_t DIM1) { 
	delete [] $1;
}

%typemap(csin) (SGTYPE* IN_ARRAY1, int32_t DIM1) "$csinput"
%typemap(csout) (SGTYPE* IN_ARRAY1, int32_t DIM1) {
	return $imcall;
}

%enddef


         //         IMTYPE == same as CTYPE  
         //(SGTYPE, CTYPE, CSTYPE)
TYPEMAP_IN1(bool, bool, unsigned int)
TYPEMAP_IN1(char, char, char)
TYPEMAP_IN1(uint8_t, uint8_t, unsigned char)
TYPEMAP_IN1(int16_t, short, short)
TYPEMAP_IN1(uint16_t, ushort, unsigned short)
TYPEMAP_IN1(int32_t, int, int)
TYPEMAP_IN1(uint32_t, uint, unsigned int)
TYPEMAP_IN1(int64_t, int, long)
TYPEMAP_IN1(uint64_t, uint, unsigned long)
TYPEMAP_IN1(long long, long, long long)
TYPEMAP_IN1(float32_t, float, float)
TYPEMAP_IN1(float64_t, double, double)

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

 /* Two dimensional input/output arrays */

 

 
%define TYPEMAP_ARRAYOUT1(SGTYPE, CTYPE, CSHARPTYPE)


%typemap(ctype) (SGTYPE** ARGOUT1, int32_t* DIM1)		%{CSHARPTYPE *%}
%typemap(imtype) (SGTYPE** ARGOUT1, int32_t* DIM1)		%{CTYPE[]%}
%typemap(cstype) (SGTYPE** ARGOUT1, int32_t* DIM1)	%{CTYPE[]%}


//  this typemap can not be correct, look into this further 0 this is allocating the pointer type, but no the data
%typemap(in) (SGTYPE** ARGOUT1, int32_t* DIM1) {
    $1 = (SGTYPE**) malloc(sizeof(SGTYPE*));
    $2 = (int32_t*) malloc(sizeof(int32_t));
}

//  This typemap can not be correct, look into this further, commented out  "$result = arr;" for now because of error
//  'jresult' was not declared in this scope
%typemap(argout) (SGTYPE** ARGOUT1, int32_t* DIM1) {
	SGTYPE* vec = *$1;
	CSHARPTYPE *arr;
	int i;
	arr = $input;
	if (!arr)
		return;
	for (i=0; i < *$2; i++)
		arr[i] = (CSHARPTYPE)vec[i];
		
	//$result = arr;
	
	//release(arr);
	//JCALL3(Release##JAVATYPE##ArrayElements, jenv, $input, arr, 0);
}


%typemap(csin) (SGTYPE** ARGOUT1, int32_t* DIM1) "$csinput"

%enddef


TYPEMAP_ARRAYOUT1(bool, bool, unsigned int)
TYPEMAP_ARRAYOUT1(char, char, char)
TYPEMAP_ARRAYOUT1(uint8_t, uint8_t, unsigned char)
TYPEMAP_ARRAYOUT1(int16_t, short, short)
TYPEMAP_ARRAYOUT1(uint16_t, ushort, unsigned short)
TYPEMAP_ARRAYOUT1(int32_t, int, int)
TYPEMAP_ARRAYOUT1(uint32_t, uint, unsigned int)
TYPEMAP_ARRAYOUT1(int64_t, int, long)
TYPEMAP_ARRAYOUT1(uint64_t, uint, unsigned long)
TYPEMAP_ARRAYOUT1(long long, long, long long)
TYPEMAP_ARRAYOUT1(float32_t, float, float)
TYPEMAP_ARRAYOUT1(float64_t, double, double)

#undef TYPEMAP_ARRAYOUT1
