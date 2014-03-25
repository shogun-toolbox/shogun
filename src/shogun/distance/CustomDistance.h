/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CUSTOMDISTANCE_H___
#define _CUSTOMDISTANCE_H___

#include <shogun/lib/config.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/common.h>
#include <shogun/distance/Distance.h>
#include <shogun/features/Features.h>

namespace shogun
{
/** @brief The Custom Distance allows for custom user provided distance matrices.
 *
 * For squared training matrices it allows to store only the upper triangle of
 * the distance to save memory: Full symmetric distance matrices can be stored as
 * is or can be internally converted into (or directly given in) upper triangle
 * representation. Also note that values are stored as 32bit floats.
 *
 */
class CCustomDistance: public CDistance
{
	public:
		/** default constructor */
		CCustomDistance();

		/** constructor
		 *
		 * compute custom distance from given distance matrix
		 * @param d distance matrix
		 */
		CCustomDistance(CDistance* d);

		/** constructor
		 * @param distance_matrix distance matrix
		 */
		CCustomDistance(const SGMatrix<float64_t> distance_matrix);

		/** constructor
		 *
		 * sets full distance matrix from full distance matrix
		 * (from double precision floats)
		 *
		 * @param dm distance matrix
		 * @param rows number of rows in matrix
		 * @param cols number of cols in matrix
		 * @return if setting was successful
		 */
		CCustomDistance(
			const float64_t* dm, int32_t rows, int32_t cols);

		/** constructor
		 *
		 * sets full distance matrix from full distance matrix
		 * (from single precision floats)
		 *
		 * @param dm distance matrix
		 * @param rows number of rows in matrix
		 * @param cols number of cols in matrix
		 * @return if setting was successful
		 */
		CCustomDistance(
			const float32_t* dm, int32_t rows, int32_t cols);

		virtual ~CCustomDistance();

		/** initialize distance with dummy features
		 *
		 * Distances always need feature objects assigned. As the custom distance
		 * does not really require this it creates some magic dummy features
		 * that only know about the number of vectors
		 *
		 * @param rows features of left-hand side
		 * @param cols features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool dummy_init(int32_t rows, int32_t cols);

		/** initialize distance
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** clean up distance */
		virtual void cleanup();

		/** return what type of distance we are
		 *
		 * @return distance type CUSTOM
		 */
		virtual EDistanceType get_distance_type() { return D_CUSTOM; }

		/** return feature type the distance can deal with
		 *
		 * @return feature type ANY
		 */
		virtual EFeatureType get_feature_type() { return F_ANY; }

		/** return feature class the distance can deal with
		 *
		 * @return feature class ANY
		 */
		virtual EFeatureClass get_feature_class() { return C_ANY; }

		/** return the distance's name
		 *
		 * @return name Custom
		 */
		virtual const char* get_name() const { return "CustomDistance"; }

		/** set distance matrix (only elements from upper triangle)
		 * from elements of upper triangle (concat'd), including the
		 * main diagonal
		 *
		 * small variant for floats64's, triangle needs to have less than 2**32 elements
		 *
		 * @param dm distance matrix
		 * @param len denotes the size of the array and should match len=cols*(cols+1)/2
		 * @return if setting was successful
		 */
		bool set_triangle_distance_matrix_from_triangle(
			const float64_t* dm, int32_t len)
		{
			return set_triangle_distance_matrix_from_triangle_generic(dm, len);
		}

		/** set distance matrix (only elements from upper triangle)
		 * from elements of upper triangle (concat'd), including the
		 * main diagonal
		 *
		 * small variant for floats32's, triangle needs to have less than 2**32 elements
		 *
		 * @param dm distance matrix
		 * @param len denotes the size of the array and should match len=cols*(cols+1)/2
		 * @return if setting was successful
		 */
		bool set_triangle_distance_matrix_from_triangle(
			const float32_t* dm, int32_t len)
		{
			return set_triangle_distance_matrix_from_triangle_generic(dm, len);
		}

		/** set distance matrix (only elements from upper triangle)
		 * from elements of upper triangle (concat'd), including the
		 * main diagonal
		 *
		 * big variant, allowing the triangle to have more than 2**31-1 elements
		 *
		 * @param dm distance matrix
		 * @param len denotes the size of the array and should match len=cols*(cols+1)/2
		 * @return if setting was successful
		 */
		template <class T>
		bool set_triangle_distance_matrix_from_triangle_generic(
			const T* dm, int64_t len)
		{
			ASSERT(dm)
			ASSERT(len>0)

			int64_t cols = (int64_t) floor(-0.5 + CMath::sqrt(0.25+2*len));

			int64_t int32_max=2147483647;

			if (cols> int32_max)
				SG_ERROR("Matrix larger than %d x %d\n", int32_max)

			if (cols*(cols+1)/2 != len)
			{
				SG_ERROR("dm should be a vector containing a lower triangle matrix, with len=cols*(cols+1)/2 elements\n")
				return false;
			}

			cleanup_custom();
			SG_DEBUG("using custom distance of size %dx%d\n", cols,cols)

			dmatrix= SG_MALLOC(float32_t, len);

			upper_diagonal=true;
			num_rows=cols;
			num_cols=cols;

			for (int64_t i=0; i<len; i++)
				dmatrix[i]=dm[i];

			dummy_init(num_rows, num_cols);
			return true;
		}

		/** set distance matrix (only elements from upper triangle)
		 * from squared matrix
		 *
		 * for float64's
		 *
		 * @param dm distance matrix
		 * @param rows number of rows in matrix
		 * @param cols number of cols in matrix
		 * @return if setting was successful
		 */
		inline bool set_triangle_distance_matrix_from_full(
			const float64_t* dm, int32_t rows, int32_t cols)
		{
			return set_triangle_distance_matrix_from_full_generic(dm, rows, cols);
		}

		/** set distance matrix (only elements from upper triangle)
		 * from squared matrix
		 *
		 * for float32's
		 *
		 * @param dm distance matrix
		 * @param rows number of rows in matrix
		 * @param cols number of cols in matrix
		 * @return if setting was successful
		 */
		inline bool set_triangle_distance_matrix_from_full(
			const float32_t* dm, int32_t rows, int32_t cols)
		{
			return set_triangle_distance_matrix_from_full_generic(dm, rows, cols);
		}

		/** set distance matrix (only elements from upper triangle)
		 * from squared matrix
		 *
		 * @param dm distance matrix
		 * @param rows number of rows in matrix
		 * @param cols number of cols in matrix
		 * @return if setting was successful
		 */
		template <class T>
		bool set_triangle_distance_matrix_from_full_generic(
			const T* dm, int32_t rows, int32_t cols)
		{
			ASSERT(rows==cols)

			cleanup_custom();
			SG_DEBUG("using custom distance of size %dx%d\n", cols,cols)

			dmatrix= SG_MALLOC(float32_t, int64_t(cols)*(cols+1)/2);

			upper_diagonal=true;
			num_rows=cols;
			num_cols=cols;

			for (int64_t row=0; row<num_rows; row++)
			{
				for (int64_t col=row; col<num_cols; col++)
				{
					int64_t idx=row * num_cols - row*(row+1)/2 + col;
					dmatrix[idx]= (float32_t) dm[col*num_rows+row];
				}
			}
			dummy_init(rows, cols);
			return true;
		}

		/** set full distance matrix from full distance matrix
		 *
		 * for float64's
		 *
		 * @param dm distance matrix
		 * @param rows number of rows in matrix
		 * @param cols number of cols in matrix
		 * @return if setting was successful
		 */
		bool set_full_distance_matrix_from_full(
			const float64_t* dm, int32_t rows, int32_t cols)
		{
			return set_full_distance_matrix_from_full_generic(dm, rows, cols);
		}

		/** set full distance matrix from full distance matrix
		 *
		 * for float32's
		 *
		 * @param dm distance matrix
		 * @param rows number of rows in matrix
		 * @param cols number of cols in matrix
		 * @return if setting was successful
		 */
		bool set_full_distance_matrix_from_full(
			const float32_t* dm, int32_t rows, int32_t cols)
		{
			return set_full_distance_matrix_from_full_generic(dm, rows, cols);
		}

		/** set full distance matrix from full distance matrix
		 *
		 * @param dm distance matrix
		 * @param rows number of rows in matrix
		 * @param cols number of cols in matrix
		 * @return if setting was successful
		 */
		template <class T>
		bool set_full_distance_matrix_from_full_generic(const T* dm, int32_t rows, int32_t cols)
		{
			cleanup_custom();
			SG_DEBUG("using custom distance of size %dx%d\n", rows,cols)

			dmatrix=SG_MALLOC(float32_t, rows*cols);

			upper_diagonal=false;
			num_rows=rows;
			num_cols=cols;

			for (int32_t row=0; row<num_rows; row++)
			{
				for (int32_t col=0; col<num_cols; col++)
				{
					dmatrix[row * num_cols + col]=dm[col*num_rows+row];
				}
			}

			dummy_init(rows, cols);
			return true;
		}

		/** get number of vectors of lhs features
		 *
		 * @return number of vectors of left-hand side
		 */
		virtual int32_t get_num_vec_lhs()
		{
			return num_rows;
		}

		/** get number of vectors of rhs features
		 *
		 * @return number of vectors of right-hand side
		 */
		virtual int32_t get_num_vec_rhs()
		{
			return num_cols;
		}

		/** test whether features have been assigned to lhs and rhs
		 *
		 * @return true if features are assigned
		 */
		virtual bool has_features()
		{
			return (num_rows>0) && (num_cols>0);
		}

	protected:
		/** compute distance function
		 *
		 * @param row row
		 * @param col col
		 * @return computed distance function
		 */
		virtual float64_t compute(int32_t row, int32_t col);

	private:
		void init();

		/** only cleanup stuff specific to Custom distance */
		void cleanup_custom();

	protected:
		/** distance matrix */
		float32_t* dmatrix;
		/** number of rows */
		int32_t num_rows;
		/** number of columns */
		int32_t num_cols;
		/** upper diagonal */
		bool upper_diagonal;
};

}
#endif /* _CUSTOMKERNEL_H__ */
