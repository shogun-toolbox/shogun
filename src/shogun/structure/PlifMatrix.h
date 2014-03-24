/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _PLIFMATRIX_H_
#define _PLIFMATRIX_H_

#include <shogun/lib/config.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/SGNDArray.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/base/SGObject.h>
#include <shogun/structure/Plif.h>
#include <shogun/structure/PlifBase.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/lib/DynamicArray.h>

namespace shogun
{
template <class T> class SGString;

/** @brief store plif arrays for all transitions in the model
 */
class CPlifMatrix: public CSGObject
{
	public:
		/** constructor
		 *
		 */
		CPlifMatrix();

		/** destructor
		 *
		 */
		~CPlifMatrix();

		/** get array of all plifs
		 *
		 * @return plif array
		 */
		inline CPlif** get_PEN() { return m_PEN; }

		/** get plif matrix
		 *
		 * @return matrix of plifs
		 */
		inline CPlifBase** get_plif_matrix() { return m_plif_matrix; }

		/** get number of states
		 *  the number of states determines the size of the plif matrix
		 *
		 * @return number of states
		 */
		inline int32_t get_num_states() { return m_num_states; }


		/** get plifs defining the mapping of signals to states
		 *
		 * @return plifs
		 */
		inline CPlifBase** get_state_signals() { return m_state_signals; }

		/** get number of plifs
		 *
		 * @return number of plifs
		 */
		inline int32_t get_num_plifs() { return m_num_plifs; }

		/** get number of support points for picewise linear transformations (PLiFs)
		 *
		 * @return number of support points
		 */
		inline int32_t get_num_limits() { return m_num_limits; }

		/** create an empty plif matrix of size num_plifs * num_limits
		 *
		 * @param num_plifs number of plifs
		 * @param num_limits number of plif limits
		 */
		void create_plifs(int32_t num_plifs, int32_t num_limits);

		/** set plif ids
		 *
		 * @param ids plif ids
		 */
		void set_plif_ids(SGVector<int32_t> ids);

		/** set array of min values for all plifs
		 *
		 * @param min_values array of min values
		 */
		void set_plif_min_values(SGVector<float64_t> min_values);

		/** set array of max values for all plifs
		 *
		 * @param max_values array of max values
		 */
		void set_plif_max_values(SGVector<float64_t> max_values);

		/** set plif use cache
		 *
		 * @param use_cache set array of bool values
		 */
		void set_plif_use_cache(SGVector<bool> use_cache);

		/** set plif use svm
		 *
		 * @param use_svm use svm
		 */
		void set_plif_use_svm(SGVector<int32_t> use_svm);

		/** set all abscissa values of the support points for the
		 *  for the pice wise linear transformations (PLiFs)
		 *
		 * @param limits array of length num_plifs*num_limits
		 */
		void set_plif_limits(SGMatrix<float64_t> limits);

		/** set all ordinate values of the support points for the
		 *  for the pice wise linear transformations (PLiFs)
		 *
		 * @param penalties plif values: array of length num_plifs*num_limits
		 */
		void set_plif_penalties(SGMatrix<float64_t> penalties);

		/** set names for the PLiFs
		 *
		 * @param names names
		 * @param num_values number of names
		 * @param maxlen maximal string len of the names
		 */
		void set_plif_names(SGString<char>* names, int32_t num_values, int32_t maxlen=0);

		/** set plif transform type; for some features the plifs live in log space
		 *  therefore the input values have to be transformed to log space before
		 *  the transformation can be applied; the transform type is string coded
		 *
		 * @param transform_type transform type (e.g. LOG(x), LOG(x+1), ...)
		 * @param num_values number of transform strings
		 * @param maxlen of transform strings
		 */
		void set_plif_transform_type(SGString<char>* transform_type, int32_t num_values, int32_t maxlen=0);

		/** return plif id for idx
		 *
		 * @param idx idx of plif
		 * @return id of plif
		 */
		inline int32_t get_plif_id(int32_t idx)
		{
			int32_t id = m_ids[idx];
			if (id>=m_num_plifs)
				SG_ERROR("plif id (%i)  exceeds array length (%i)\n",id,m_num_plifs)
			return id;
		}

		/** parse an 3D array of plif ids and compute the corresponding
		 *  2D plif matrix by subsuming the third dim into one PlifArray;
		 *  Note: the class PlifArray is derived from PlifBase. It computes
		 *        all individual plifs and sums them up.
		 *
		 * @param penalties_array 3D array of plif ids (nofstates x nofstates x nof(features for each transition))
		 * @return success
		 */
		bool compute_plif_matrix(SGNDArray<float64_t> penalties_array);

		/** parse an 3D array of plif ids and compute the corresponding
		 *  3D plif array;
		 *
		 * @param state_signals mapping of features to states
		 * @return success
		 */
		bool compute_signal_plifs(SGMatrix<int32_t> state_signals);

		/** set best path plif state signal matrix
		 *
		 * @param plif_id_matrix plif id matrix
		 * @param m dimension m of matrix
		 * @param n dimension n of matrix
		 */
		void set_plif_state_signal_matrix(int32_t *plif_id_matrix, int32_t m, int32_t n);


		/** @return object name */
		virtual const char* get_name() const { return "PlifMatrix"; }

	protected:

		/** array of plifs*/
		CPlif** m_PEN;

		/** number of plifs */
		int32_t m_num_plifs;

		/** number of supporting points per plif*/
		int32_t m_num_limits;

		/** number of states in model*/
		int32_t m_num_states;

		/** maximal number of features for a given state*/
		int m_feat_dim3;

		/** plif ids*/
		CDynamicArray<int32_t> m_ids;

		/** plif matrix */
		CPlifBase** m_plif_matrix;

		/** state signals*/
		CPlifBase** m_state_signals;
};
}
#endif /* _PLIFMATRIX_H_ */
