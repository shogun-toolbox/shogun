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
/* counting the size of arrays */
int SWIG_table_size(lua_State* L, int index) {
	int n=0;
	lua_pushnil(L);
	while (lua_next(L, index) != 0) {
		++n;
		lua_pop(L, 1);
	}
	return n;
}
%}
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
%define TYPEMAP_IN1(sg_type)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER)
	(sg_type* IN_ARRAY1, int32_t DIM1) {
		if(!lua_istable(L, $input)) {
			$1 = 0;
		}
		else {
			$1 = 1;
			int numitems = 0;
			numitems = SWIG_table_size(L, $input);
			if(numitems < 1) {
				lua_pushstring(L, "table appears to be empty");
				lua_error(L);
				$1 = 0;
			}
		}
}
%typemap(in) (sg_type* IN_ARRAY1, int32_t DIM1) {
		sg_type *array;
		if (!lua_istable(L,$input)) {
			lua_pushstring(L,"expected a table");
			return 0;
		}
		$2 = SWIG_table_size(L,$input);
		if ($2 < 1){
			lua_pushstring(L,"table appears to be empty");
			return 0;
		}
		array = new sg_type[$2];
		for (int i = 0; i < $2; i++) {
			lua_rawgeti(L,$input,i+1);
			if (lua_isnumber(L,-1)){
				array[i] = (sg_type)lua_tonumber(L,-1);
			}
			else {
				lua_pop(L,1);
				lua_pushstring(L, "table must contain numbers");
				delete [] array;
				lua_error(L);
				return 0;
			}
			lua_pop(L,1);
		}

		$1 = array;
}

%typemap(freearg) (sg_type* IN_ARRAY1, int32_t DIM1) {
		delete[] $1;
}
%enddef

/* Define concrete examples of the TYPEMAP_IN1 macros */
TYPEMAP_IN1(uint8_t)
TYPEMAP_IN1(int32_t)
TYPEMAP_IN1(int16_t)
TYPEMAP_IN1(uint16_t)
TYPEMAP_IN1(float32_t)
TYPEMAP_IN1(float64_t)

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
%define TYPEMAP_ARGOUT1(sg_type)
%typemap(in, numinputs=0) (sg_type** ARGOUT1, int32_t* DIM1) {
    $1 = (sg_type**) malloc(sizeof(sg_type*));
    $2 = (int32_t*) malloc(sizeof(int32_t));
}

%typemap(argout) (sg_type** ARGOUT1, int32_t* DIM1) {
    sg_type* vec = *$1;
    int32_t len = *$2;

	lua_newtable(L);

    for (int32_t i=0; i < len; i++) {
		lua_pushnumber(L,(lua_Number)vec[i]);
		lua_rawseti(L,-2,i+1);
	}
    SWIG_arg++;
    free(*$1); free($1); free($2);
}
%enddef

TYPEMAP_ARGOUT1(uint8_t)
TYPEMAP_ARGOUT1(int32_t)
TYPEMAP_ARGOUT1(int16_t)
TYPEMAP_ARGOUT1(uint16_t)
TYPEMAP_ARGOUT1(float32_t)
TYPEMAP_ARGOUT1(float64_t)
#undef TYPEMAP_ARGOUT1
