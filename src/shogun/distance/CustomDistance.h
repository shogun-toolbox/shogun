/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Fernando Iglesias, Yuyu Zhang, Sergey Lisitsyn, 
 *          Evan Shelhamer
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
class CustomDistance: public Distance
{
	public:
		/** default constructor */
		CustomDistance();

		/** constructor
		 *
		 * compute custom distance from given distance matrix
		 * @param d distance matrix
		 */
		CustomDistance(const std::shared_ptr<Distance>& d);

		/** constructor
		 * @param distance_matrix distance matrix
		 */
		CustomDistance(const SGMatrix<float64_t> distance_matrix);

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
		CustomDistance(
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
		CustomDistance(
			const float32_t* dm, int32_t rows, int32_t cols);

		~CustomDistance() override;

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
		bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

		/** clean up distance */
		void cleanup() override;

		/** return what type of distance we are
		 *
		 * @return distance type CUSTOM
		 */
		EDistanceType get_distance_type() override { return D_CUSTOM; }

		/** return feature type the distance can deal with
		 *
		 * @return feature type ANY
		 */
		EFeatureType get_feature_type() override { return F_ANY; }

		/** return feature class the distance can deal with
		 *
		 * @return feature class ANY
		 */
		EFeatureClass get_feature_class() override { return C_ANY; }

		/** return the distance's name
		 *
		 * @return name Custom
		 */
		const char* get_name() const override { return "CustomDistance"; }

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

			int64_t cols = (int64_t)floor(-0.5 + std::sqrt(0.25 + 2 * len));

			int64_t int32_max=2147483647;

			if (cols> int32_max)
				error("Matrix larger than {} x {}", int32_max);

			if (cols*(cols+1)/2 != len)
			{
				error("dm should be a vector containing a lower triangle matrix, with len=cols*(cols+1)/2 elements");
				return false;
			}

			cleanup_custom();
			SG_DEBUG("using custom distance of size {}x{}", cols,cols)

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
			SG_DEBUG("using custom distance of size {}x{}", cols,cols)

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
			SG_DEBUG("using custom distance of size {}x{}", rows,cols)

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
		int32_t get_num_vec_lhs() override
		{
			return num_rows;
		}

		/** get number of vectors of rhs features
		 *
		 * @return number of vectors of right-hand side
		 */
		int32_t get_num_vec_rhs() override
		{
			return num_cols;
		}

		/** test whether features have been assigned to lhs and rhs
		 *
		 * @return true if features are assigned
		 */
		bool has_features() override
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
		float64_t compute(int32_t row, int32_t col) override;

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
