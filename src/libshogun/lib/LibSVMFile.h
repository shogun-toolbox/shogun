/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */
#ifndef __LIBSVM_FILE_H__
#define __LIBSVM_FILE_H__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/io.h>

namespace shogun
{
	/** read sparse real valued features in svm light format
	 * e.g. -1 1:10.0 2:100.2 1000:1.3 
	 * with -1 == (optional) label
	 * and dim 1    - value  10.0
	 *     dim 2    - value 100.2
	 *     dim 1000 - value   1.3
	 *
	 * @param matrix matrix to read into
	 * @param num_feat number of features for each vector
	 * @param num_vec number of vectors in matrix
	 * @return if reading was successful
	 */
	bool read_real_valued_sparse(
		TSparse<float64_t>*& matrix, int32_t& num_feat, int32_t& num_vec);

	/** write sparse real valued features in svm light format
	 *
	 * @param matrix matrix to write
	 * @param num_feat number of features for each vector
	 * @param num_vec number of vectros in matrix
	 * @return if writing was successful
	 */
	bool write_real_valued_sparse(
		const TSparse<float64_t>* matrix, int32_t num_feat, int32_t num_vec);

	/** read dense real valued features, simple ascii format
	 * e.g. 1.0 1.1 0.2 
	 *      2.3 3.5 5
	 *
	 *  a matrix that consists of 3 vectors with each of 2d
	 *
	 * @param matrix matrix to read into
	 * @param num_feat number of features for each vector
	 * @param num_vec number of vectors in matrix
	 * @return if reading was successful
	 */
	bool read_real_valued_dense(
		float64_t*& matrix, int32_t& num_feat, int32_t& num_vec);

	/** write dense real valued features, simple ascii format
	 *
	 * @param matrix matrix to write
	 * @param num_feat number of features for each vector
	 * @param num_vec number of vectros in matrix
	 * @return if writing was successful
	 */
	bool write_real_valued_dense(
		const float64_t* matrix, int32_t num_feat, int32_t num_vec);

	/** read char string features, simple ascii format
	 * e.g. foo bar
	 *      ACGTACGTATCT
	 *
	 *  two strings
	 *
	 * @param strings strings to read into
	 * @param num_str number of strings
	 * @param max_string_len length of longest string
	 * @return if reading was successful
	 */
	bool read_char_valued_strings(TString<char>*& strings, int32_t& num_str, int32_t& max_string_len);
	/** write char string features, simple ascii format
	 *
	 * @param strings strings to write
	 * @param num_str number of strings
	 * @return if writing was successful
	 */
	bool write_char_valued_strings(const TString<char>* strings, int32_t num_str);

}
#endif //__LIBSVM_FILE_H__

