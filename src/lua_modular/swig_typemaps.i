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

/* One dimensional input arrays */
%define TYPEMAP_SGVECTOR(SGTYPE)

%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) shogun::SGVector<SGTYPE> {
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

%typemap(in) shogun::SGVector<SGTYPE> {
	SGTYPE *array;
	int32_t i, len;

	if (!lua_istable(L, $input)) {
		lua_pushstring(L,"expected a table");
		return 0;
	}
	
	len = SWIG_table_size(L, $input);
	if (len < 1){
		lua_pushstring(L,"table appears to be empty");
		return 0;
	}
	
	array = new SGTYPE[len];
	for ( i = 0; i < len; i++) {
		lua_rawgeti(L, $input, i + 1);
		if (lua_isnumber(L, -1)){
			array[i] = (SGTYPE)lua_tonumber(L, -1);
		}
		else {
			lua_pop(L, 1);
			lua_pushstring(L, "table must contain numbers");
			delete [] array;
			lua_error(L);
			return 0;
		}
		lua_pop(L,1);
	}

	$1 = shogun::SGVector<SGTYPE>((SGTYPE *)array, len);
}

%typemap(out) shogun::SGVector<SGTYPE> {
	int32_t i;
	int32_t len = $1.length;
	SGTYPE* vec = $1.vector;	

	lua_newtable(L);	
	for (i = 0; i < len; i++) {
		lua_pushnumber(L, (lua_Number)vec[i]);
		lua_rawseti(L, -2, i + 1);
	}
 	SWIG_arg++;
}
%enddef

/* Define concrete examples of the TYPEMAP_SGVECTOR macros */
TYPEMAP_SGVECTOR(uint8_t)
TYPEMAP_SGVECTOR(int32_t)
TYPEMAP_SGVECTOR(int16_t)
TYPEMAP_SGVECTOR(uint16_t)
TYPEMAP_SGVECTOR(float32_t)
TYPEMAP_SGVECTOR(float64_t)

#undef TYPEMAP_SGVECTOR

/* Two dimensional input/output arrays */
%define TYPEMAP_SGMATRIX(SGTYPE)

%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) shogun::SGMatrix<SGTYPE>
{
   if(!lua_istable(L, $input)) {
		$1 = 0;
	}
	else {
		$1 = 1;
		int rows = SWIG_table_size(L, $input);
		if(rows < 1) {
			lua_pushstring(L, "table appears to be empty");
			lua_error(L);
			$1 = 0;
		}
		else {
			lua_rawgeti(L, $input, 1);
			if (!lua_istable(L, -1)) {
				lua_pushstring(L, "matrix table expected");
				lua_error(L);			
				$1 = 0;			
			}
			else {
				int cols = SWIG_table_size(L, -1);
				if (cols < 1) {
					lua_pushstring(L, "table appears to be empty");
					lua_error(L);
					$1 = 0;
				}			
			}
					
		}
	}
}

%typemap(in) shogun::SGMatrix<SGTYPE> {
	SGTYPE *array;
	int32_t i, j, rows, cols;

	if (!lua_istable(L, $input)) {
		lua_pushstring(L,"expected a table");
		return 0;
	}
	
	rows = SWIG_table_size(L, $input);
	if (rows < 1){
		lua_pushstring(L,"table appears to be empty");
		return 0;
	}
	cols = 0;

	for (i = 0; i < rows; i++) {
		lua_rawgeti(L, $input, i + 1);
		if (!lua_istable(L, -1)) {
			lua_pop(L, 1);
			lua_pushstring(L, "matrix table expected");
			lua_error(L);
			return 0;
		}
		if (cols ==0) {
			cols = SWIG_table_size(L, -1);
			array = new SGTYPE[rows * cols];
		}
		for (j = 0; j < cols; j++) {
			lua_rawgeti(L, -1, i + 1);
			if (lua_isnumber(L, -1)) {
				array[i * cols + j] = (SGTYPE)lua_tonumber(L, -1);
				lua_pop(L, 1);
			}
			else {
				lua_pop(L, 1);
				lua_pushstring(L, "table must contain numbers");
				delete [] array;
				lua_error(L);
				return 0;
			}
		}
		lua_pop(L, 1);
	}
	
	$1 = shogun::SGMatrix<SGTYPE>((SGTYPE*)array, rows, cols);
}

%typemap(out) shogun::SGMatrix<SGTYPE> {
	int32_t rows = $1.num_rows;
	int32_t cols = $1.num_cols;
	int32_t len = rows * cols;
	int32_t i, j;	
	
	lua_newtable(L);
		
	for (i = 0; i < rows; i++) {
		lua_newtable(L);
		for (j = 0; j < cols; j++) {
			lua_pushnumber(L, (lua_Number)$1.matrix[i * cols + j]);
			lua_rawseti(L, -2, j + 1);
		}
		lua_rawseti(L, -2, i + 1);
	}

	SWIG_arg++;
}

%enddef

/* Define concrete examples of the TYPEMAP_SGMATRIX macros */
TYPEMAP_SGMATRIX(uint8_t)
TYPEMAP_SGMATRIX(int32_t)
TYPEMAP_SGMATRIX(int16_t)
TYPEMAP_SGMATRIX(uint16_t)
TYPEMAP_SGMATRIX(float32_t)
TYPEMAP_SGMATRIX(float64_t)

#undef TYPEMAP_SGMATRIX
