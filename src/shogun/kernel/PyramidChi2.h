/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Yuyu Zhang, Evan Shelhamer,
 *          Bjoern Esser, Sergey Lisitsyn
 */

#ifndef PYRAMIDCHI2_H_
#define PYRAMIDCHI2_H_

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/DotKernel.h>
#include <shogun/features/Features.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/RandomMixin.h>
#include <shogun/mathematics/RandomNamespace.h>

namespace shogun
{
	template <class T> class DenseFeatures;

/** @brief Pyramid Kernel over Chi2 matched histograms.
 *
 *
 *
 */
class PyramidChi2 : public RandomMixin<DotKernel>
{
public:
	/** default constructor protected to avoid its usage */
	PyramidChi2();

	/** constructor
	 *
	 * @param size cache size in MB
	 * @param num_cells2 - the number of pyramid cells
	 * @param weights_foreach_cell2 the vector of weights for each cell with which the Chi2 distance gets weighted
	 * @param width_computation_type2 - 0 use the following parameter as fixed
	 *	width, 1- use mean of inner distances
	 *	in cases 1 and 2 the value of parameter width is still important, see parameter width2
	 * @param width2 - in case of width_computation_type ==0 it is the
	 *	width, in case of width_computation_type > 0 its value determines
	 *	the how many random features are used for determining the width
	 *	in case of width_computation_type > 0 set width2 <=1 to use all
	 *	LEFT HAND SIDE features for width estimation
	 */
	PyramidChi2(int32_t size, int32_t num_cells2,
		float64_t* weights_foreach_cell2,
		int32_t width_computation_type2,
		float64_t width2);

	/** constructor
	 *
	 * @param l features lhs
	 *	convention: concatenated features along all cells, i.e. [feature for cell1, feature for cell2, ... feature for last cell] , the dimensionality of the base feature is equal to dividing the total feature length by the number ofcells
	 * @param r features rhs
	 *	the same convention as for param l applies here
	 * @param size cache size
	 * @param num_cells2 - the number of pyramid cells
	 * @param weights_foreach_cell2 the vector of weights for each cell with which the Chi2 distance gets weighted
	 * @param width_computation_type2 - 0 use the following parameter as fixed
	 *	width, 1- use mean of inner distances
	 *	in case 1 the value of parameter width is important!!!
	 * @param width2 - in case of width_computation_type ==0 it is the
	 *	width, in case of width_computation_type > 0 its value determines
	 *	the how many random features are used for determining the width
	 *	in case of width_computation_type > 0 set width2 <=1 to use all
	 *	LEFT HAND SIDE features for width estimation
	 */
	PyramidChi2(
		std::shared_ptr<DenseFeatures<float64_t>> l, std::shared_ptr<DenseFeatures<float64_t>> r,
		int32_t size, int32_t num_cells2,
		float64_t* weights_foreach_cell2,
		int32_t width_computation_type2,
		float64_t width2 );

	/** init
	 *
	 * @param l features lhs
	 * @param r reatures rhs
	 */
	virtual bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r);


	virtual ~PyramidChi2();

	/** cleanup */
	virtual void cleanup();

	/** return what type of kernel we are Linear,Polynomial, Gaussian,... */
	virtual EKernelType get_kernel_type()
	{
		return K_PYRAMIDCHI2;
	}

	/** return the name of a kernel */
	virtual const char* get_name() const { return "PyramidChi2"; }


	/** sets parameters, see also constructor
	 *
	 * @param num_cells2 - the number of pyramid cells
	 * @param weights_foreach_cell2 the vector of weights for each cell with which the Chi2 distance gets weighted
	 * @param width_computation_type2 - 0 use the following parameter as fixed
	 *	width, 1- use mean of inner distances
	 *	in cases 1 and 2 the value of parameter width is still important, see parameter width2
	 * @param width2 - in case of width_computation_type ==0 it is the
	 *	width, in case of width_computation_type > 0 its value determines
	 *	the how many random features are used for determining the width
	 *	in case of width_computation_type > 0 set width2 <=1 to use all
	 *	LEFT HAND SIDE features for width estimation
	 */
	virtual void setparams_pychi2(int32_t num_cells2,
		float64_t* weights_foreach_cell2,
		int32_t width_computation_type2,
		float64_t width2);

protected:
	/** compute kernel function for features a and b
	 *
	 * @param idx_a index of feature vector a
	 * @param idx_b index of feature vector b
	 * @return computed kernel function
	 */
	virtual float64_t compute(int32_t idx_a, int32_t idx_b);

protected:

	/** number of pyramidcells across all pyramidlevel */
	int32_t num_cells;

	/** vector of weights for each pyramid cell*/
	float64_t* weights;

	/** width_computation_type */
	int32_t width_computation_type;
		/** width */
	float64_t width;
	/** in case of adaptive width computation: how many features to use */
	int32_t num_randfeats_forwidthcomputation;




};
}
#endif /*PYRAMIDCHI2_H_*/
