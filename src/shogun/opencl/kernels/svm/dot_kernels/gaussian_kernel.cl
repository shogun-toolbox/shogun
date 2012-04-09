/*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 3 of the License, or
* (at your option) any later version.
*
* Written (W) 2012 Philippe Tillet
*/

__stringify(

// matrix layouts: C...row_major, A...col_major, B...col_major
__kernel void gaussian_kernel(
	  double width,
          __global double * C,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols) 
{
  for(unsigned int i = get_global_id(0) ; i < C_internal_rows*C_internal_cols ; i+=get_global_size(0))
  {
    C[i] = exp(-C[i]/width);
  }
}

)