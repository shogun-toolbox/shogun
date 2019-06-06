/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Sergey Lisitsyn, Wu Lin
 *          Jacob Walker, Evan Shelhamer, Giovanni De Toni, Viktor Gal,
 *          Roman Votyakov, Esben Sorig, Evgeniy Andreev, Fernando Iglesias,
 *          Saurabh Goyal, Shashwat Lal Das, Thoralf Klein, Hu Shell,
 *          Soumyajitde De, Evangelos Anagnostopoulos
 */

#include <shogun/base/progress.h>
#include <shogun/io/File.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/Time.h>
#include <shogun/lib/common.h>
#include <shogun/lib/config.h>

#include <shogun/base/Parallel.h>

#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/normalizer/IdentityKernelNormalizer.h>
#include <shogun/features/Features.h>
#include <shogun/base/Parameter.h>

#include <shogun/classifier/svm/SVM.h>

#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <shogun/mathematics/Math.h>

using namespace shogun;

CKernel::CKernel() : CSGObject()
{
	init();
	register_params();
}

CKernel::CKernel(int32_t size) : CSGObject()
{
	init();

	if (size<10)
		size=10;

	cache_size=size;
	register_params();
}


CKernel::CKernel(CFeatures* p_lhs, CFeatures* p_rhs, int32_t size) : CSGObject()
{
	init();

	if (size<10)
		size=10;

	cache_size=size;

	set_normalizer(new CIdentityKernelNormalizer());
	init(p_lhs, p_rhs);
	register_params();
}

CKernel::~CKernel()
{
	if (get_is_initialized())
		SG_ERROR("Kernel still initialized on destruction.\n")

	remove_lhs_and_rhs();
	SG_UNREF(normalizer);
}

bool CKernel::init(CFeatures* l, CFeatures* r)
{
	//make sure features were indeed supplied
	REQUIRE(l, "CKernel::init(%p, %p): Left hand side features required!\n", l, r)
	REQUIRE(r, "CKernel::init(%p, %p): Right hand side features required!\n", l, r)

	//make sure features are compatible
	if (l->support_compatible_class())
	{
		REQUIRE(l->get_feature_class_compatibility(r->get_feature_class()),
			"Right hand side of features (%s) must be compatible with left hand side features (%s)\n",
			l->get_name(), r->get_name());
	}
	else
	{
		REQUIRE(l->get_feature_class()==r->get_feature_class(),
			"Right hand side of features (%s) must be compatible with left hand side features (%s)\n",
			l->get_name(), r->get_name())
	}
	ASSERT(l->get_feature_type()==r->get_feature_type())

	//remove references to previous features
	remove_lhs_and_rhs();

	SG_REF(l);
	if (l==r)
		lhs_equals_rhs=true;
	else // l!=r
		SG_REF(r);

	lhs=l;
	rhs=r;

	ASSERT(!num_lhs || num_lhs==l->get_num_vectors())
	ASSERT(!num_rhs || num_rhs==l->get_num_vectors())

	num_lhs=l->get_num_vectors();
	num_rhs=r->get_num_vectors();

	SG_DEBUG("leaving CKernel::init(%p, %p)\n", l, r)
	return true;
}

bool CKernel::set_normalizer(CKernelNormalizer* n)
{
	SG_REF(n);
	if (lhs && rhs)
		n->init(this);

	SG_UNREF(normalizer);
	normalizer=n;

	return (normalizer!=NULL);
}

CKernelNormalizer* CKernel::get_normalizer() const
{
	SG_REF(normalizer)
	return normalizer;
}

bool CKernel::init_normalizer()
{
	return normalizer->init(this);
}

void CKernel::cleanup()
{
	remove_lhs_and_rhs();
}

void CKernel::load(CFile* loader)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
}

void CKernel::save(CFile* writer)
{
	SGMatrix<float64_t> k_matrix=get_kernel_matrix<float64_t>();
	SG_SET_LOCALE_C;
	writer->set_matrix(k_matrix.matrix, k_matrix.num_rows, k_matrix.num_cols);
	SG_RESET_LOCALE;
}

void CKernel::remove_lhs_and_rhs()
{
	if (rhs!=lhs)
		SG_UNREF(rhs);
	rhs = NULL;
	num_rhs=0;

	SG_UNREF(lhs);
	lhs = NULL;
	num_lhs=0;
	lhs_equals_rhs=false;
}

void CKernel::remove_lhs()
{
	if (rhs==lhs)
		rhs=NULL;
	SG_UNREF(lhs);
	lhs = NULL;
	num_lhs=0;
	lhs_equals_rhs=false;
}

/// takes all necessary steps if the rhs is removed from kernel
void CKernel::remove_rhs()
{
	if (rhs!=lhs)
		SG_UNREF(rhs);
	rhs = NULL;
	num_rhs=0;
	lhs_equals_rhs=false;
}

#define ENUM_CASE(n) case n: SG_INFO(#n " ") break;

void CKernel::list_kernel()
{
	SG_INFO("%p - \"%s\" weight=%1.2f OPT:%s", this, get_name(),
			get_combined_kernel_weight(),
			get_optimization_type()==FASTBUTMEMHUNGRY ? "FASTBUTMEMHUNGRY" :
			"SLOWBUTMEMEFFICIENT");

	switch (get_kernel_type())
	{
		ENUM_CASE(K_UNKNOWN)
		ENUM_CASE(K_LINEAR)
		ENUM_CASE(K_POLY)
		ENUM_CASE(K_GAUSSIAN)
		ENUM_CASE(K_GAUSSIANSHIFT)
		ENUM_CASE(K_GAUSSIANMATCH)
		ENUM_CASE(K_GAUSSIANCOMPACT)
		ENUM_CASE(K_HISTOGRAM)
		ENUM_CASE(K_SALZBERG)
		ENUM_CASE(K_LOCALITYIMPROVED)
		ENUM_CASE(K_SIMPLELOCALITYIMPROVED)
		ENUM_CASE(K_FIXEDDEGREE)
		ENUM_CASE(K_WEIGHTEDDEGREE)
		ENUM_CASE(K_WEIGHTEDDEGREEPOS)
		ENUM_CASE(K_WEIGHTEDDEGREERBF)
		ENUM_CASE(K_WEIGHTEDCOMMWORDSTRING)
		ENUM_CASE(K_POLYMATCH)
		ENUM_CASE(K_ALIGNMENT)
		ENUM_CASE(K_COMMWORDSTRING)
		ENUM_CASE(K_COMMULONGSTRING)
		ENUM_CASE(K_SPECTRUMRBF)
		ENUM_CASE(K_COMBINED)
		ENUM_CASE(K_AUC)
		ENUM_CASE(K_CUSTOM)
		ENUM_CASE(K_SIGMOID)
		ENUM_CASE(K_CHI2)
		ENUM_CASE(K_DIAG)
		ENUM_CASE(K_CONST)
		ENUM_CASE(K_DISTANCE)
		ENUM_CASE(K_LOCALALIGNMENT)
		ENUM_CASE(K_PYRAMIDCHI2)
		ENUM_CASE(K_OLIGO)
		ENUM_CASE(K_MATCHWORD)
		ENUM_CASE(K_TPPK)
		ENUM_CASE(K_REGULATORYMODULES)
		ENUM_CASE(K_SPARSESPATIALSAMPLE)
		ENUM_CASE(K_HISTOGRAMINTERSECTION)
		ENUM_CASE(K_WAVELET)
		ENUM_CASE(K_WAVE)
		ENUM_CASE(K_CAUCHY)
		ENUM_CASE(K_TSTUDENT)
		ENUM_CASE(K_MULTIQUADRIC)
		ENUM_CASE(K_EXPONENTIAL)
		ENUM_CASE(K_RATIONAL_QUADRATIC)
		ENUM_CASE(K_POWER)
		ENUM_CASE(K_SPHERICAL)
		ENUM_CASE(K_LOG)
		ENUM_CASE(K_SPLINE)
		ENUM_CASE(K_ANOVA)
		ENUM_CASE(K_CIRCULAR)
		ENUM_CASE(K_INVERSEMULTIQUADRIC)
		ENUM_CASE(K_SPECTRUMMISMATCHRBF)
		ENUM_CASE(K_DISTANTSEGMENTS)
		ENUM_CASE(K_BESSEL)
		ENUM_CASE(K_JENSENSHANNON)
		ENUM_CASE(K_DIRECTOR)
		ENUM_CASE(K_PRODUCT)
		ENUM_CASE(K_EXPONENTIALARD)
		ENUM_CASE(K_GAUSSIANARD)
		ENUM_CASE(K_GAUSSIANARDSPARSE)
		ENUM_CASE(K_STREAMING)
		ENUM_CASE(K_PERIODIC)
	}

	switch (get_feature_class())
	{
		ENUM_CASE(C_UNKNOWN)
		ENUM_CASE(C_DENSE)
		ENUM_CASE(C_SPARSE)
		ENUM_CASE(C_STRING)
		ENUM_CASE(C_STREAMING_DENSE)
		ENUM_CASE(C_STREAMING_SPARSE)
		ENUM_CASE(C_STREAMING_STRING)
		ENUM_CASE(C_STREAMING_VW)
		ENUM_CASE(C_COMBINED)
		ENUM_CASE(C_COMBINED_DOT)
		ENUM_CASE(C_WD)
		ENUM_CASE(C_SPEC)
		ENUM_CASE(C_WEIGHTEDSPEC)
		ENUM_CASE(C_POLY)
		ENUM_CASE(C_BINNED_DOT)
		ENUM_CASE(C_DIRECTOR_DOT)
		ENUM_CASE(C_LATENT)
		ENUM_CASE(C_MATRIX)
		ENUM_CASE(C_FACTOR_GRAPH)
		ENUM_CASE(C_INDEX)
		ENUM_CASE(C_SUB_SAMPLES_DENSE)
		ENUM_CASE(C_ANY)
	}

	switch (get_feature_type())
	{
		ENUM_CASE(F_UNKNOWN)
		ENUM_CASE(F_BOOL)
		ENUM_CASE(F_CHAR)
		ENUM_CASE(F_BYTE)
		ENUM_CASE(F_SHORT)
		ENUM_CASE(F_WORD)
		ENUM_CASE(F_INT)
		ENUM_CASE(F_UINT)
		ENUM_CASE(F_LONG)
		ENUM_CASE(F_ULONG)
		ENUM_CASE(F_SHORTREAL)
		ENUM_CASE(F_DREAL)
		ENUM_CASE(F_LONGREAL)
		ENUM_CASE(F_ANY)
	}
	SG_INFO("\n")
}
#undef ENUM_CASE

bool CKernel::init_optimization(
	int32_t count, int32_t *IDX, float64_t * weights)
{
   SG_ERROR("kernel does not support linadd optimization\n")
	return false ;
}

bool CKernel::delete_optimization()
{
   SG_ERROR("kernel does not support linadd optimization\n")
	return false;
}

float64_t CKernel::compute_optimized(int32_t vector_idx)
{
   SG_ERROR("kernel does not support linadd optimization\n")
	return 0;
}

void CKernel::compute_batch(
	int32_t num_vec, int32_t* vec_idx, float64_t* target, int32_t num_suppvec,
	int32_t* IDX, float64_t* weights, float64_t factor)
{
   SG_ERROR("kernel does not support batch computation\n")
}

void CKernel::add_to_normal(int32_t vector_idx, float64_t weight)
{
   SG_ERROR("kernel does not support linadd optimization, add_to_normal not implemented\n")
}

void CKernel::clear_normal()
{
   SG_ERROR("kernel does not support linadd optimization, clear_normal not implemented\n")
}

int32_t CKernel::get_num_subkernels()
{
	return 1;
}

void CKernel::compute_by_subkernel(
	int32_t vector_idx, float64_t * subkernel_contrib)
{
   SG_ERROR("kernel compute_by_subkernel not implemented\n")
}

const float64_t* CKernel::get_subkernel_weights(int32_t &num_weights)
{
	num_weights=1 ;
	return &combined_kernel_weight ;
}

SGVector<float64_t> CKernel::get_subkernel_weights()
{
	int num_weights = 1;
	const float64_t* weight = get_subkernel_weights(num_weights);
	return SGVector<float64_t>(const_cast<float64_t*>(weight),1,false);
}

void CKernel::set_subkernel_weights(const SGVector<float64_t> weights)
{
	ASSERT(weights.vector)
	if (weights.vlen!=1)
      SG_ERROR("number of subkernel weights should be one ...\n")

	combined_kernel_weight = weights.vector[0] ;
}

CKernel* CKernel::obtain_from_generic(CSGObject* kernel)
{
	if (kernel)
	{
		CKernel* casted=dynamic_cast<CKernel*>(kernel);
		REQUIRE(casted, "CKernel::obtain_from_generic(): Error, provided object"
				" of class \"%s\" is not a subclass of CKernel!\n",
				kernel->get_name());
		return casted;
	}
	else
		return NULL;
}

bool CKernel::init_optimization_svm(CSVM * svm)
{
	int32_t num_suppvec=svm->get_num_support_vectors();
	int32_t* sv_idx=SG_MALLOC(int32_t, num_suppvec);
	float64_t* sv_weight=SG_MALLOC(float64_t, num_suppvec);

	for (int32_t i=0; i<num_suppvec; i++)
	{
		sv_idx[i]    = svm->get_support_vector(i);
		sv_weight[i] = svm->get_alpha(i);
	}
	bool ret = init_optimization(num_suppvec, sv_idx, sv_weight);

	SG_FREE(sv_idx);
	SG_FREE(sv_weight);
	return ret;
}

void CKernel::load_serializable_post() noexcept(false)
{
	CSGObject::load_serializable_post();
	if (lhs_equals_rhs)
		rhs=lhs;
}

void CKernel::save_serializable_pre() noexcept(false)
{
	CSGObject::save_serializable_pre();

	if (lhs_equals_rhs)
		rhs=NULL;
}

void CKernel::save_serializable_post() noexcept(false)
{
	CSGObject::save_serializable_post();

	if (lhs_equals_rhs)
		rhs=lhs;
}

void CKernel::register_params()
{
	SG_ADD(&cache_size, "cache_size", "Cache size in MB.");
	SG_ADD(
		&lhs, "lhs", "Feature vectors to occur on left hand side.",
		ParameterProperties::READONLY);
	SG_ADD(
		&rhs, "rhs", "Feature vectors to occur on right hand side.",
		ParameterProperties::READONLY);
	SG_ADD(&lhs_equals_rhs, "lhs_equals_rhs",
		"If features on lhs are the same as on rhs.");
	SG_ADD(&num_lhs, "num_lhs", "Number of feature vectors on left hand side.");
	SG_ADD(
	    &num_rhs, "num_rhs", "Number of feature vectors on right hand side.");
	SG_ADD(
	    &combined_kernel_weight, "combined_kernel_weight",
	    "Combined kernel weight.", ParameterProperties::HYPER);
	SG_ADD(
	    &optimization_initialized, "optimization_initialized",
	    "Optimization is initialized.");
	SG_ADD(&properties, "properties", "Kernel properties.");
	SG_ADD(
	    &normalizer, "normalizer", "Normalize the kernel.",
	    ParameterProperties::HYPER);

	SG_ADD_OPTIONS(
	    (machine_int_t*)&opt_type, "opt_type", "Optimization type.",
	    ParameterProperties::NONE,
	    SG_OPTIONS(FASTBUTMEMHUNGRY, SLOWBUTMEMEFFICIENT));
}


void CKernel::init()
{
	cache_size = 10;
	lhs=NULL;
	rhs=NULL;
	num_lhs=0;
	num_rhs=0;
	lhs_equals_rhs=false;
	combined_kernel_weight=1;
	optimization_initialized=false;
	opt_type=FASTBUTMEMHUNGRY;
	properties=KP_NONE;
	normalizer=NULL;

	set_normalizer(new CIdentityKernelNormalizer());
}

namespace shogun
{
/** kernel thread parameters */
template <class T> struct K_THREAD_PARAM
{
	/** kernel */
	CKernel* kernel;
	/** start (unit row) */
	int32_t start;
	/** end (unit row) */
	int32_t end;
	/** start (unit number of elements) */
	int64_t total_start;
	/** m */
	int32_t m;
	/** n */
	int32_t n;
	/** result */
	T* result;
	/** kernel matrix k(i,j)=k(j,i) */
	bool symmetric;
	/** output progress */
	bool verbose;
	/* Progress bar*/
	PRange<int64_t>* pb;
};
}

float64_t CKernel::sum_symmetric_block(index_t block_begin, index_t block_size,
		bool no_diag)
{
	SG_DEBUG("Entering\n");

	REQUIRE(has_features(), "No features assigned to kernel\n")
	REQUIRE(lhs_equals_rhs, "The kernel matrix is not symmetric!\n")
	REQUIRE(block_begin>=0 && block_begin<num_rhs,
			"Invalid block begin index (%d, %d)!\n", block_begin, block_begin)
	REQUIRE(block_begin+block_size<=num_rhs,
			"Invalid block size (%d) at starting index (%d, %d)! "
			"Please use smaller blocks!", block_size, block_begin, block_begin)
	REQUIRE(block_size>=1, "Invalid block size (%d)!\n", block_size)

	float64_t sum=0.0;

	// since the block is symmetric with main diagonal inside, we can save half
	// the computation with using only the upper triangular part.
	// this can be done in parallel
	#pragma omp parallel for reduction(+:sum)
	for (index_t i=0; i<block_size; ++i)
	{
		// compute the kernel values on the upper triangular part of the kernel
		// matrix and compute sum on the fly
		for (index_t j=i+1; j<block_size; ++j)
		{
			float64_t k=kernel(i+block_begin, j+block_begin);
			sum+=k;
		}
	}

	// the actual sum would be twice of what we computed
	sum*=2;

	// add the diagonal elements if required - keeping this check
	// outside of the loop to save cycles
	if (!no_diag)
	{
		#pragma omp parallel for reduction(+:sum)
		for (index_t i=0; i<block_size; ++i)
		{
			float64_t diag=kernel(i+block_begin, i+block_begin);
			sum+=diag;
		}
	}

	SG_DEBUG("Leaving\n");

	return sum;
}

float64_t CKernel::sum_block(index_t block_begin_row, index_t block_begin_col,
		index_t block_size_row, index_t block_size_col, bool no_diag)
{
	SG_DEBUG("Entering\n");

	REQUIRE(has_features(), "No features assigned to kernel\n")
	REQUIRE(block_begin_row>=0 && block_begin_row<num_lhs &&
			block_begin_col>=0 && block_begin_col<num_rhs,
			"Invalid block begin index (%d, %d)!\n",
			block_begin_row, block_begin_col)
	REQUIRE(block_begin_row+block_size_row<=num_lhs &&
			block_begin_col+block_size_col<=num_rhs,
			"Invalid block size (%d, %d) at starting index (%d, %d)! "
			"Please use smaller blocks!", block_size_row, block_size_col,
			block_begin_row, block_begin_col)
	REQUIRE(block_size_row>=1 && block_size_col>=1,
			"Invalid block size (%d, %d)!\n", block_size_row, block_size_col)

	// check if removal of diagonal is required/valid
	if (no_diag && block_size_row!=block_size_col)
	{
		SG_WARNING("Not removing the main diagonal since block is not square!\n");
		no_diag=false;
	}

	float64_t sum=0.0;

	// this can be done in parallel for the rows/cols
	#pragma omp parallel for reduction(+:sum)
	for (index_t i=0; i<block_size_row; ++i)
	{
		// compute the kernel values and compute sum on the fly
		for (index_t j=0; j<block_size_col; ++j)
		{
			float64_t k=no_diag && i==j ? 0 :
				kernel(i+block_begin_row, j+block_begin_col);
			sum+=k;
		}
	}

	SG_DEBUG("Leaving\n");

	return sum;
}

SGVector<float64_t> CKernel::row_wise_sum_symmetric_block(index_t block_begin,
		index_t block_size, bool no_diag)
{
	SG_DEBUG("Entering\n");

	REQUIRE(has_features(), "No features assigned to kernel\n")
	REQUIRE(lhs_equals_rhs, "The kernel matrix is not symmetric!\n")
	REQUIRE(block_begin>=0 && block_begin<num_rhs,
			"Invalid block begin index (%d, %d)!\n", block_begin, block_begin)
	REQUIRE(block_begin+block_size<=num_rhs,
			"Invalid block size (%d) at starting index (%d, %d)! "
			"Please use smaller blocks!", block_size, block_begin, block_begin)
	REQUIRE(block_size>=1, "Invalid block size (%d)!\n", block_size)

	// initialize the vector that accumulates the row/col-wise sum on the go
	SGVector<float64_t> row_sum(block_size);
	row_sum.set_const(0.0);

	// since the block is symmetric with main diagonal inside, we can save half
	// the computation with using only the upper triangular part.
	// this can be done in parallel for the rows/cols
	#pragma omp parallel for
	for (index_t i=0; i<block_size; ++i)
	{
		// compute the kernel values on the upper triangular part of the kernel
		// matrix and compute row-wise sum on the fly
		for (index_t j=i+1; j<block_size; ++j)
		{
			float64_t k=kernel(i+block_begin, j+block_begin);
			#pragma omp critical
			{
				row_sum[i]+=k;
				row_sum[j]+=k;
			}
		}
	}

	// add the diagonal elements if required - keeping this check
	// outside of the loop to save cycles
	if (!no_diag)
	{
		#pragma omp parallel for
		for (index_t i=0; i<block_size; ++i)
		{
			float64_t diag=kernel(i+block_begin, i+block_begin);
			row_sum[i]+=diag;
		}
	}

	SG_DEBUG("Leaving\n");

	return row_sum;
}

SGMatrix<float64_t> CKernel::row_wise_sum_squared_sum_symmetric_block(index_t
		block_begin, index_t block_size, bool no_diag)
{
	SG_DEBUG("Entering\n");

	REQUIRE(has_features(), "No features assigned to kernel\n")
	REQUIRE(lhs_equals_rhs, "The kernel matrix is not symmetric!\n")
	REQUIRE(block_begin>=0 && block_begin<num_rhs,
			"Invalid block begin index (%d, %d)!\n", block_begin, block_begin)
	REQUIRE(block_begin+block_size<=num_rhs,
			"Invalid block size (%d) at starting index (%d, %d)! "
			"Please use smaller blocks!", block_size, block_begin, block_begin)
	REQUIRE(block_size>=1, "Invalid block size (%d)!\n", block_size)

	// initialize the matrix that accumulates the row/col-wise sum on the go
	// the first column stores the sum of kernel values
	// the second column stores the sum of squared kernel values
	SGMatrix<float64_t> row_sum(block_size, 2);
	row_sum.set_const(0.0);

	// since the block is symmetric with main diagonal inside, we can save half
	// the computation with using only the upper triangular part
	// this can be done in parallel for the rows/cols
#pragma omp parallel for
	for (index_t i=0; i<block_size; ++i)
	{
		// compute the kernel values on the upper triangular part of the kernel
		// matrix and compute row-wise sum and squared sum on the fly
		for (index_t j=i+1; j<block_size; ++j)
		{
			float64_t k=kernel(i+block_begin, j+block_begin);
#pragma omp critical
			{
				row_sum(i, 0)+=k;
				row_sum(j, 0)+=k;
				row_sum(i, 1)+=k*k;
				row_sum(j, 1)+=k*k;
			}
		}
	}

	// add the diagonal elements if required - keeping this check
	// outside of the loop to save cycles
	if (!no_diag)
	{
#pragma omp parallel for
		for (index_t i=0; i<block_size; ++i)
		{
			float64_t diag=kernel(i+block_begin, i+block_begin);
			row_sum(i, 0)+=diag;
			row_sum(i, 1)+=diag*diag;
		}
	}

	SG_DEBUG("Leaving\n");

	return row_sum;
}

SGVector<float64_t> CKernel::row_col_wise_sum_block(index_t block_begin_row,
		index_t block_begin_col, index_t block_size_row,
		index_t block_size_col, bool no_diag)
{
	SG_DEBUG("Entering\n");

	REQUIRE(has_features(), "No features assigned to kernel\n")
	REQUIRE(block_begin_row>=0 && block_begin_row<num_lhs &&
			block_begin_col>=0 && block_begin_col<num_rhs,
			"Invalid block begin index (%d, %d)!\n",
			block_begin_row, block_begin_col)
	REQUIRE(block_begin_row+block_size_row<=num_lhs &&
			block_begin_col+block_size_col<=num_rhs,
			"Invalid block size (%d, %d) at starting index (%d, %d)! "
			"Please use smaller blocks!", block_size_row, block_size_col,
			block_begin_row, block_begin_col)
	REQUIRE(block_size_row>=1 && block_size_col>=1,
			"Invalid block size (%d, %d)!\n", block_size_row, block_size_col)

	// check if removal of diagonal is required/valid
	if (no_diag && block_size_row!=block_size_col)
	{
		SG_WARNING("Not removing the main diagonal since block is not square!\n");
		no_diag=false;
	}

	// initialize the vector that accumulates the row/col-wise sum on the go
	// the first block_size_row entries store the row-wise sum of kernel values
	// the nextt block_size_col entries store the col-wise sum of kernel values
	SGVector<float64_t> sum(block_size_row+block_size_col);
	sum.set_const(0.0);

	// this can be done in parallel for the rows/cols
#pragma omp parallel for
	for (index_t i=0; i<block_size_row; ++i)
	{
		// compute the kernel values and compute sum on the fly
		for (index_t j=0; j<block_size_col; ++j)
		{
			float64_t k=no_diag && i==j ? 0 :
				kernel(i+block_begin_row, j+block_begin_col);
#pragma omp critical
			{
				sum[i]+=k;
				sum[j+block_size_row]+=k;
			}
		}
	}

	SG_DEBUG("Leaving\n");

	return sum;
}

template <class T> void* CKernel::get_kernel_matrix_helper(void* p)
{
	K_THREAD_PARAM<T>* params= (K_THREAD_PARAM<T>*) p;
	int32_t i_start=params->start;
	int32_t i_end=params->end;
	CKernel* k=params->kernel;
	T* result=params->result;
	bool symmetric=params->symmetric;
	int32_t n=params->n;
	int32_t m=params->m;
	bool verbose=params->verbose;
	int64_t total_start=params->total_start;
	int64_t total=total_start;
	PRange<int64_t>* pb = params->pb;

	for (int32_t i=i_start; i<i_end; i++)
	{
		int32_t j_start=0;

		if (symmetric)
			j_start=i;

		for (int32_t j=j_start; j<n; j++)
		{
			float64_t v=k->kernel(i,j);
			result[i+j*m]=v;

			if (symmetric && i!=j)
				result[j+i*m]=v;

			if (verbose)
			{
				total++;

				if (symmetric && i!=j)
					total++;

				pb->print_progress();

				// TODO: replace with the new signal
				// if (CSignal::cancel_computations())
				//	break;
			}
		}

	}

	return NULL;
}

template <class T>
SGMatrix<T> CKernel::get_kernel_matrix()
{
	T* result = NULL;

	REQUIRE(has_features(), "no features assigned to kernel\n")

	int32_t m=get_num_vec_lhs();
	int32_t n=get_num_vec_rhs();

	int64_t total_num = int64_t(m)*n;

	// if lhs == rhs and sizes match assume k(i,j)=k(j,i)
	bool symmetric= (lhs && lhs==rhs && m==n);

	SG_DEBUG("returning kernel matrix of size %dx%d\n", m, n)

	result=SG_MALLOC(T, total_num);

	int32_t num_threads=env()->get_num_threads();
	K_THREAD_PARAM<T> params;
	int64_t step = total_num/num_threads;
	index_t t = 0;
	auto pb = SG_PROGRESS(range(total_num));
#pragma omp parallel for lastprivate(t) private(params)
	for (t = 0; t < num_threads; ++t)
	{
		params.kernel = this;
		params.result = result;
		params.start = compute_row_start(t*step, n, symmetric);
		params.end = compute_row_start((t+1)*step, n, symmetric);
		params.total_start=t*step;
		params.n=n;
		params.m=m;
		params.symmetric=symmetric;
		params.verbose=false;
		params.pb = &pb;
		CKernel::get_kernel_matrix_helper<T>((void*)&params);
	}

	if (total_num % num_threads != 0)
	{
		params.kernel = this;
		params.result = result;
		params.start = compute_row_start(t*step, n, symmetric);
		params.end = m;
		params.total_start=t*step;
		params.n=n;
		params.m=m;
		params.symmetric=symmetric;
		params.verbose=false;
		params.pb = &pb;
		CKernel::get_kernel_matrix_helper<T>((void*)&params);
	}

	pb.complete();

	return SGMatrix<T>(result,m,n,true);
}


template SGMatrix<float64_t> CKernel::get_kernel_matrix<float64_t>();
template SGMatrix<float32_t> CKernel::get_kernel_matrix<float32_t>();

template void* CKernel::get_kernel_matrix_helper<float64_t>(void* p);
template void* CKernel::get_kernel_matrix_helper<float32_t>(void* p);
