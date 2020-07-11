/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Sergey Lisitsyn, Jacob Walker,
 *          Wu Lin, Evgeniy Andreev, Roman Votyakov, Bjoern Esser, Esben Sorig,
 *          Evan Shelhamer, Giovanni De Toni, Grigorii Guz, Thoralf Klein,
 *          Viktor Gal, Yuyu Zhang, Soumyajitde De
 */

#ifndef _KERNEL_H___
#define _KERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>

#include <shogun/io/SGIO.h>
#include <shogun/io/File.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/FeatureTypes.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/Features.h>

namespace shogun
{
	class File;
	class Features;
	class KernelNormalizer;

#ifdef USE_SHORTREAL_KERNELCACHE
	/** kernel cache element */
	typedef float32_t KERNELCACHE_ELEM;
#else
	/** kernel cache element */
	typedef float64_t KERNELCACHE_ELEM;
#endif

/** kernel cache index */
typedef int64_t KERNELCACHE_IDX;


/** optimization type */
enum EOptimizationType
{
	FASTBUTMEMHUNGRY,
	SLOWBUTMEMEFFICIENT
};

/** kernel type */
enum EKernelType
{
	K_UNKNOWN = 0,
	K_LINEAR = 10,
	K_POLY = 20,
	K_GAUSSIAN = 30,
	K_GAUSSIANSHIFT = 32,
	K_GAUSSIANMATCH = 33,
	K_GAUSSIANCOMPACT = 34,
	K_HISTOGRAM = 40,
	K_SALZBERG = 41,
	K_LOCALITYIMPROVED = 50,
	K_SIMPLELOCALITYIMPROVED = 60,
	K_FIXEDDEGREE = 70,
	K_WEIGHTEDDEGREE =    80,
	K_WEIGHTEDDEGREEPOS = 81,
	K_WEIGHTEDDEGREERBF = 82,
	K_WEIGHTEDCOMMWORDSTRING = 90,
	K_POLYMATCH = 100,
	K_ALIGNMENT = 110,
	K_COMMWORDSTRING = 120,
	K_COMMULONGSTRING = 121,
	K_SPECTRUMRBF = 122,
	K_SPECTRUMMISMATCHRBF = 123,
	K_COMBINED = 140,
	K_AUC = 150,
	K_CUSTOM = 160,
	K_SIGMOID = 170,
	K_CHI2 = 180,
	K_DIAG = 190,
	K_CONST = 200,
	K_DISTANCE = 220,
	K_LOCALALIGNMENT = 230,
	K_PYRAMIDCHI2 = 240,
	K_OLIGO = 250,
	K_MATCHWORD = 260,
	K_TPPK = 270,
	K_REGULATORYMODULES = 280,
	K_SPARSESPATIALSAMPLE = 290,
	K_HISTOGRAMINTERSECTION = 300,
	K_WAVELET = 310,
	K_WAVE = 320,
	K_CAUCHY = 330,
	K_TSTUDENT = 340,
	K_RATIONAL_QUADRATIC = 350,
	K_MULTIQUADRIC = 360,
	K_EXPONENTIAL = 370,
	K_SPHERICAL = 380,
	K_SPLINE = 390,
	K_ANOVA = 400,
	K_POWER = 410,
	K_LOG = 420,
	K_CIRCULAR = 430,
	K_INVERSEMULTIQUADRIC = 440,
	K_DISTANTSEGMENTS = 450,
	K_BESSEL = 460,
	K_JENSENSHANNON = 470,
	K_DIRECTOR = 480,
	K_PRODUCT = 490,
	K_EXPONENTIALARD = 500,
	K_GAUSSIANARD = 510,
	K_GAUSSIANARDSPARSE = 511,
	K_STREAMING = 520,
	K_PERIODIC = 530,
	K_MATERN = 540
};

/** kernel property */
enum EKernelProperty
{
	KP_NONE = 0,
	KP_LINADD = 1,	// Kernels that can be optimized via doing normal updates w + dw
	KP_KERNCOMBINATION = 2,	// Kernels that are infact a linear combination of subkernels K=\sum_i b_i*K_i
	KP_BATCHEVALUATION = 4  // Kernels that can on the fly generate normals in linadd and more quickly/memory efficient process batches instead of single examples
};

class SVM;

/** @brief The Kernel base class.
 *
 * Non-mathematically spoken, a kernel is a function
 * that given two input objects \f${\bf x}\f$ and \f${\bf x'}\f$ returns a
 * score describing the similarity of the vectors. The score should be larger
 * when the objects are more similar.
 *
 * It can be defined as
 *
 * \f[
 * k({\bf x},{\bf x'})= \Phi_k({\bf x})\cdot \Phi_k({\bf x'})
 * \f]
 *
 * where \f$\Phi\f$ maps the objects into some potentially high dimensional
 * feature space.
 *
 * Apart from the input features, the base kernel takes only one argument (the
 * size of the kernel cache) that is used to efficiently train kernel-machines
 * like e.g. SVMs.
 *
 * In case you would like to define your own kernel, you only have to define a
 * new compute() function (and the kernel name via get_name() and
 * the kernel type get_kernel_type()). A good example to look at is the
 * GaussianKernel.
 */
class Kernel : public SGObject
{
	friend class VarianceKernelNormalizer;
	friend class SqrtDiagKernelNormalizer;
	friend class AvgDiagKernelNormalizer;
	friend class RidgeKernelNormalizer;
	friend class FirstElementKernelNormalizer;
	friend class MultitaskKernelNormalizer;
	friend class MultitaskKernelMklNormalizer;
	friend class MultitaskKernelMaskNormalizer;
	friend class MultitaskKernelMaskPairNormalizer;
	friend class TanimotoKernelNormalizer;
	friend class DiceKernelNormalizer;
	friend class ZeroMeanCenterKernelNormalizer;
	
	public:

		/** default constructor
		 *
		 */
		Kernel();


		/** constructor
		 *
		 * @param size cache size
		 */
		Kernel(int32_t size);

		/** constructor
		 *
		 * @param l features for left-hand side
		 * @param r features for right-hand side
		 * @param size cache size
		 */
		Kernel(std::shared_ptr<Features> l, std::shared_ptr<Features> r, int32_t size);

		~Kernel() override;

		/** get kernel function for lhs feature vector a
		 * and rhs feature vector b
		 *
		 * @param idx_a index of feature vector a
		 * @param idx_b index of feature vector b
		 * @return computed kernel function
		 */
		float64_t kernel(int32_t idx_a, int32_t idx_b);

		/** get kernel matrix
		 *
		 * @return computed kernel matrix (needs to be cleaned up)
		 */
		SGMatrix<float64_t> get_kernel_matrix()
		{
			return get_kernel_matrix<float64_t>();
		}

		/** @return Vector with diagonal elements of the kernel matrix.
		 * Note that left- and right-handside features must be set and of equal
		 * size
		 *
		 * @param preallocated vector with space for results
		 */
		SGVector<float64_t> get_kernel_diagonal(SGVector<float64_t>
				preallocated=SGVector<float64_t>())
		{
			require(lhs, "Kernel::get_kernel_diagonal(): Left-handside "
					"features missing!");

			require(rhs, "Kernel::get_kernel_diagonal(): Right-handside "
						"features missing!");

			int32_t length=Math::min(lhs->get_num_vectors(),rhs->get_num_vectors());

			/* allocate space if necessary */
			if (!preallocated.vector)
				preallocated=SGVector<float64_t>(length);
			else
			{
				require(preallocated.vlen==length,
						"{}::get_kernel_diagonal(): Preallocated vector has"
						" wrong size!", get_name());
			}

			for (index_t i=0; i<preallocated.vlen; ++i)
				preallocated[i]=kernel(i, i);

			return preallocated;
		}

		/**
		 * get column j
		 *
		 * @return the jth column of the kernel matrix
		 */
		virtual SGVector<float64_t> get_kernel_col(int32_t j)
		{

			SGVector<float64_t> col = SGVector<float64_t>(num_rhs);

			for (int32_t i=0; i!=num_rhs; i++)
				col[i] = kernel(i,j);

			return col;
		}


		/**
		 * get row i
		 *
		 * @return the ith row of the kernel matrix
		 */
		virtual SGVector<float64_t> get_kernel_row(int32_t i)
		{
			SGVector<float64_t> row = SGVector<float64_t>(num_lhs);

			for (int32_t j=0; j!=num_lhs; j++)
				row[j] = kernel(i,j);

			return row;
		}

		/**
		 * Computes sum from a symmetric part of the kernel matrix that always
		 * is supposed to contain the main upper diagonal.
		 * This method is useful while computing statistical estimation of
		 * mean/variance over kernel values but the kernel matrix is too huge
		 * to be fit inside memory.
		 *
		 * @param block_begin the row and col index at which the block starts
		 * @param block_size the number of rows and cols in the block
		 *
		 * For example, block_begin 4 and block_size 5 represents the block
		 * that starts at index (4,4) in the kernel matrix and goes upto
		 * (4+5-1,4+5-1) i.e. (8,8) both inclusive
		 *
		 * @param no_diag if true (default), the diagonal elements are excluded
		 * from the sum
		 *
		 * @return sum of kernel values within the block computed as
		 * \f[
		 *	\sum_{i}\sum_{j}k(i+\text{block-begin}, j+\text{block-begin})
		 * \f]
		 * where \f$i,j\in[0,\text{block-size}-1]\f$
		 */
		virtual float64_t sum_symmetric_block(index_t block_begin,
				index_t block_size, bool no_diag=true);

		/**
		 * Computes sum of kernel values from a specified block.
		 * This method is useful while computing statistical estimation of
		 * mean/variance over kernel values but the kernel matrix is too huge
		 * to be fit inside memory.
		 *
		 * @param block_begin_row the row index at which the block starts
		 * @param block_begin_col the col index at which the block starts
		 * @param block_size_row the number of rows in the block
		 * @param block_size_col the number of cols in the block
		 *
		 * For example, block_begin_row 0, block_begin_col 4 and block_size_row
		 * 5, block_size_col 6 represents the block
		 * that starts at index (0,4) in the kernel matrix and goes upto
		 * (0+5-1,4+6-1) i.e. (4,9) both inclusive
		 *
		 * @param no_diag if true (default is false), the diagonal elements
		 * are excluded from the sum, provided that block_size_row
		 * and block_size_col are same (i.e. the block is square). Otherwise,
		 * these are always added
		 *
		 * @return sum of kernel values within the block computed as
		 * \f[
		 *	\sum_{i}\sum_{j}k(i+\text{block-begin-row}, j+\text{block-begin-col})
		 * \f]
		 * where \f$i\in[0,\text{block-size-row}-1]\f$ and
		 * \f$j\in[0,\text{block-size-col}-1]\f$
		 */
		virtual float64_t sum_block(index_t block_begin_row,
				index_t block_begin_col, index_t block_size_row,
				index_t block_size_col, bool no_diag=false);

		/**
		 * Computes row-wise/col-wise sum from a symmetric part of the kernel
		 * matrix that always is supposed to contain the main upper diagonal.
		 * This method is useful while computing statistical estimation of
		 * mean/variance over kernel values but the kernel matrix is too huge
		 * to be fit inside memory.
		 *
		 * @param block_begin the row and col index at which the block starts
		 * @param block_size the number of rows and cols in the block
		 *
		 * For Example, block_begin 4 and block_size 5 represents the block
		 * that starts at index (4,4) in the kernel matrix and goes upto
		 * (4+5-1,4+5-1) i.e. (8,8) both inclusive
		 *
		 * @param no_diag if true (default), the diagonal elements are excluded
		 * from the row/col-wise sum
		 *
		 * @return vector containing row-wise sum computed as
		 * \f[
		 *	v[i]=\sum_{j}k(i+\text{block-begin}, j+\text{block-begin})
		 * \f]
		 * where \f$i,j\in[0,\text{block-size}-1]\f$
		 */
		virtual SGVector<float64_t> row_wise_sum_symmetric_block(index_t
				block_begin, index_t block_size, bool no_diag=true);

		/**
		 * Computes row-wise/col-wise sum and squared sum of kernel values from
		 * a symmetric part of the kernel matrix that always is supposed to
		 * contain the main upper diagonal.
		 * This method is useful while computing statistical estimation of
		 * mean/variance over kernel values but the kernel matrix is too huge
		 * to be fit inside memory.
		 *
		 * @param block_begin the row and col index at which the block starts
		 * @param block_size the number of rows and cols in the block
		 *
		 * For Example, block_begin 4 and block_size 5 represents the block
		 * that starts at index (4,4) in the kernel matrix and goes upto
		 * (4+5-1,4+5-1) i.e. (8,8) both inclusive
		 *
		 * @param no_diag if true (default), the diagonal elements are excluded
		 * from the row/col-wise sum
		 *
		 * @return a matrix whose first column contains the row-wise sum of
		 * kernel values computed as
		 * \f[
		 *	v_0[i]=\sum_{j}k(i+\text{block-begin}, j+\text{block-begin})
		 * \f]
		 * and second column contains the row-wise sum of squared kernel values
		 * \f[
		 *	v_1[i]=\sum_{j}^k^2(i+\text{block-begin}, j+\text{block-begin})
		 * \f]
		 * where \f$i,j\in[0,\text{block-size}-1]\f$
		 */
		virtual SGMatrix<float64_t> row_wise_sum_squared_sum_symmetric_block(
				index_t block_begin, index_t block_size, bool no_diag=true);

		/**
		 * Computes row-wise/col-wise sum of kernel values.
		 * This method is useful while computing statistical estimation of
		 * mean/variance over kernel values but the kernel matrix is too huge
		 * to be fit inside memory.
		 *
		 * @param block_begin_row the row index at which the block starts
		 * @param block_begin_col the col index at which the block starts
		 * @param block_size_row the number of rows in the block
		 * @param block_size_col the number of cols in the block
		 *
		 * For Example, block_begin_row 0, block_begin_col 4 and block_size_row
		 * 5, block_size_col 6 represents the block
		 * that starts at index (0,4) in the kernel matrix and goes upto
		 * (0+5-1,4+6-1) i.e. (4,9) both inclusive
		 *
		 * @param no_diag if true (default is false), the diagonal elements
		 * are excluded from the row/col-wise sum, provided that block_size_row
		 * and block_size_col are same (i.e. the block is square). Otherwise,
		 * these are always added
		 *
		 * @return a vector whose first block_size_row entries contain
		 * row-wise sum of kernel values computed as
		 * \f[
		 *	v[i]=\sum_{j}k(i+\text{block-begin-row}, j+\text{block-begin-col})
		 * \f]
		 * and rest block_size_col entries col-wise sum of kernel values
		 * computed as
		 * \f[
		 *	v[\text{block-size-row}+j]=\sum_{i}k(i+\text{block-begin-row},
		 *	j+\text{block-begin-col})
		 * \f]
		 * where \f$i\in[0,\text{block-size-row}-1]\f$ and
		 * \f$j\in[0,\text{block-size-col}-1]\f$
		 */
		virtual SGVector<float64_t> row_col_wise_sum_block(
				index_t block_begin_row, index_t block_begin_col,
				index_t block_size_row, index_t block_size_col,
				bool no_diag=false);

		/** get kernel matrix (templated)
		 *
		 * @return the kernel matrix
		 */
		template <class T> SGMatrix<T> get_kernel_matrix();

		/** initialize kernel
		 *  e.g. setup lhs/rhs of kernel, precompute normalization
		 *  constants etc.
		 *  make sure to check that your kernel can deal with the
		 *  supplied features (!)
		 *
		 *  @param lhs features for left-hand side
		 *  @param rhs features for right-hand side
		 *  @return if init was successful
		 */
		virtual bool init(std::shared_ptr<Features> lhs, std::shared_ptr<Features> rhs);

		/** set the current kernel normalizer
		 *
		 * @return if successful
		 */
		virtual bool set_normalizer(std::shared_ptr<KernelNormalizer> normalizer);

		/** obtain the current kernel normalizer
		 *
		 * @return the kernel normalizer
		 */
		virtual std::shared_ptr<KernelNormalizer> get_normalizer() const;

		/** initialize the current kernel normalizer
		 *  @return if init was successful
		 */
		virtual bool init_normalizer();

		/** clean up your kernel
		 *
		 * base method only removes lhs and rhs
		 * overload to add further cleanup but make sure Kernel::cleanup() is
		 * called
		 */
		virtual void cleanup();

		/** load the kernel matrix
		 *
		 * @param loader File object via which to load data
		 */
		void load(const std::shared_ptr<File>& loader);

		/** save kernel matrix
		 *
		 * @param writer File object via which to save data
		 */
		void save(const std::shared_ptr<File>& writer);

		/** get left-hand side of features used in kernel
		 *
		 * @return features of left-hand side
		 */
		inline std::shared_ptr<Features> get_lhs() {  return lhs; }

		/** get right-hand side of features used in kernel
		 *
		 * @return features of right-hand side
		 */
		inline std::shared_ptr<Features> get_rhs() {  return rhs; }

		/** get number of vectors of lhs features
		 *
		 * @return number of vectors of left-hand side
		 */
		virtual int32_t get_num_vec_lhs()
		{
			return num_lhs;
		}

		/** get number of vectors of rhs features
		 *
		 * @return number of vectors of right-hand side
		 */
		virtual int32_t get_num_vec_rhs()
		{
			return num_rhs;
		}

		/** test whether features have been assigned to lhs and rhs
		 *
		 * @return true if features are assigned
		 */
		virtual bool has_features()
		{
			return lhs && rhs;
		}

		/** test whether features on lhs and rhs are the same
		 *
		 * @return true if features are the same
		 */
		inline bool get_lhs_equals_rhs()
		{
			return lhs_equals_rhs;
		}

		/** remove lhs and rhs from kernel */
		virtual void remove_lhs_and_rhs();

		/** remove lhs from kernel */
		virtual void remove_lhs();

		/** remove rhs from kernel */
		virtual void remove_rhs();

		/** return what type of kernel we are, e.g.
		 * Linear,Polynomial, Gaussian,...
		 *
		 * abstract base method
		 *
		 * @return kernel type
		 */
		virtual EKernelType get_kernel_type()=0 ;

		/** return feature type the kernel can deal with
		 *
		 * abstract base method
		 *
		 * @return feature type
		 */
		virtual EFeatureType get_feature_type()=0;

		/** return feature class the kernel can deal with
		 *
		 * abstract base method
		 *
		 * @return feature class
		 */
		virtual EFeatureClass get_feature_class()=0;

		/** set the size of the kernel cache
		 *
		 * @param size of kernel cache
		 */
		inline void set_cache_size(int32_t size)
		{
			cache_size = size;
#ifdef USE_SVMLIGHT
			cache_reset();
#endif //USE_SVMLIGHT
		}

		/** return the size of the kernel cache
		 *
		 * @return size of kernel cache
		 */
		inline int32_t get_cache_size() { return cache_size; }

#ifdef USE_SVMLIGHT
		/** cache reset */
		inline void cache_reset() { resize_kernel_cache(cache_size); }

		/** get maximum elements in cache
		 *
		 * @return maximum elements in cache
		 */
		inline int32_t get_max_elems_cache() { return kernel_cache.max_elems; }

		/** get activenum cache
		 *
		 * @return activecnum cache
		 */
		inline int32_t get_activenum_cache() { return kernel_cache.activenum; }

		/** get kernel row
		 *
		 * @param docnum docnum
		 * @param active2dnum active2dnum
		 * @param buffer buffer
		 * @param full_line full line
		 */
		void get_kernel_row(
			int32_t docnum, int32_t *active2dnum, float64_t *buffer,
			bool full_line=false);

		/** cache kernel row
		 *
		 * @param x x
		 */
		void cache_kernel_row(int32_t x);

		/** cache multiple kernel rows
		 *
		 * @param key key
		 * @param varnum
		 */
		void cache_multiple_kernel_rows(int32_t* key, int32_t varnum);

		/** kernel cache reset lru */
		void kernel_cache_reset_lru();

		/** kernel cache shrink
		 *
		 * @param totdoc totdoc
		 * @param num_shrink number of shrink
		 * @param after after
		 */
		void kernel_cache_shrink(
			int32_t totdoc, int32_t num_shrink, int32_t *after);

		/** resize kernel cache
		 *
		 * @param size new size
		 * @param regression_hack hack for regression
		 */
		void resize_kernel_cache(KERNELCACHE_IDX size,
			bool regression_hack=false);

		/** set the lru time
		 *
		 * @param t the time to use
		 */
		inline void set_time(int32_t t)
		{
			kernel_cache.time=t;
		}

		/** update lru time of item at given index to avoid removal from cache
		 *
		 * @param cacheidx index in cache
		 * @return if updating was successful
		 */
		inline int32_t kernel_cache_touch(int32_t cacheidx)
		{
			if(kernel_cache.index[cacheidx] != -1)
			{
				kernel_cache.lru[kernel_cache.index[cacheidx]]=kernel_cache.time;
				return(1);
			}
			return(0);
		}

		/** check if row at given index is cached
		 *
		 * @param cacheidx index in cache
		 * @return if row at given index is cached
		 */
		inline int32_t kernel_cache_check(int32_t cacheidx)
		{
			return(kernel_cache.index[cacheidx] >= 0);
		}

		/** check if there is room for one more row in kernel cache
		 *
		 * @return if there is room for one more row in kernel cache
		 */
		inline int32_t kernel_cache_space_available()
		{
			return(kernel_cache.elems < kernel_cache.max_elems);
		}

		/** initialize kernel cache
		 *
		 * @param size size to initialize to
		 * @param regression_hack if hack for regression shall be applied
		 */
		void kernel_cache_init(int32_t size, bool regression_hack=false);

		/** cleanup kernel cache */
		void kernel_cache_cleanup();

#endif //USE_SVMLIGHT

		/** list kernel */
		void list_kernel();

		/** check if kernel has given property
		 *
		 * @param p kernel property
		 * @return if kernel has given property
		 */
		inline virtual bool has_property(EKernelProperty p)
		{
			return (properties & p) != 0;
		}

		/** for optimizable kernels, i.e. kernels where the weight
		 * vector can be computed explicitly (if it fits into memory)
		 */
		virtual void clear_normal();

		/** add vector*factor to 'virtual' normal vector
		 *
		 * @param vector_idx index
		 * @param weight weight
		 */
		virtual void add_to_normal(int32_t vector_idx, float64_t weight);

		/** get optimization type
		 *
		 * @return optimization type
		 */
		inline EOptimizationType get_optimization_type() { return opt_type; }

		/** set optimization type
		 *
		 * @param t optimization type to set
		 */
		virtual void set_optimization_type(EOptimizationType t) { opt_type=t;}

		/** check if optimization is initialized
		 *
		 * @return if optimization is initialized
		 */
		inline bool get_is_initialized() { return optimization_initialized; }

		/** initialize optimization
		 *
		 * @param count count
		 * @param IDX index
		 * @param weights weights
		 * @return if initializing was successful
		 */
		virtual bool init_optimization(
			int32_t count, int32_t *IDX, float64_t *weights);

		/** delete optimization
		 *
		 * @return if deleting was successful
		 */
		virtual bool delete_optimization();

		/** initialize optimization
		 *
		 * @param svm svm model
		 * @return if initializing was successful
		 */
		bool init_optimization_svm(const std::shared_ptr<SVM>& svm) ;

		/** compute optimized
		 *
		 * @param vector_idx index to compute
		 * @return optimized value at given index
		 */
		virtual float64_t compute_optimized(int32_t vector_idx);

		/** computes output for a batch of examples in an optimized fashion
		 * (favorable if kernel supports it, i.e. has KP_BATCHEVALUATION.  to
		 * the outputvector target (of length num_vec elements) the output for
		 * the examples enumerated in vec_idx are added. therefore make sure
		 * that it is initialized with ZERO. the following num_suppvec, IDX,
		 * alphas arguments are the number of support vectors, their indices
		 * and weights
		 */
		virtual void compute_batch(
			int32_t num_vec, int32_t* vec_idx, float64_t* target,
			int32_t num_suppvec, int32_t* IDX, float64_t* alphas,
			float64_t factor=1.0);

		/** get combined kernel weight
		 *
		 * @return combined kernel weight
		 */
		inline float64_t get_combined_kernel_weight() { return combined_kernel_weight; }

		/** set combined kernel weight
		 *
		 * @param nw new combined kernel weight
		 */
		inline void set_combined_kernel_weight(float64_t nw) { combined_kernel_weight=nw; }

		/** get number of subkernels
		 *
		 * @return number of subkernels
		 */
		virtual int32_t get_num_subkernels();

		/** compute by subkernel
		 *
		 * @param vector_idx index
		 * @param subkernel_contrib subkernel contribution
		 */
		virtual void compute_by_subkernel(
			int32_t vector_idx, float64_t * subkernel_contrib);

		/** get subkernel weights
		 *
		 * @param num_weights number of weights will be stored here
		 * @return subkernel weights
		 */
		virtual const float64_t* get_subkernel_weights(int32_t& num_weights);

		/** get subkernel weights (swig compatible)
		 *
		 * @return subkernel weights
		 */
		virtual SGVector<float64_t> get_subkernel_weights();

		/** set subkernel weights
		 *
		 * @param weights new subkernel weights
		 */
		virtual void set_subkernel_weights(SGVector<float64_t> weights);


#ifndef SWIG
		/** return derivative with respect to specified parameter
		 *
		 * @param param the parameter
		 * @param index the index of the element if parameter is a vector
		 *
		 * @return gradient with respect to parameter
		 */
		virtual SGMatrix<float64_t> get_parameter_gradient(
				Parameters::const_reference param, index_t index=-1)
		{
			error("Can't compute derivative wrt {} parameter", param.first);
			return SGMatrix<float64_t>();
		}

		/** return diagonal part of derivative with respect to specified parameter
		 *
		 * @param param the parameter
		 * @param index the index of the element if parameter is a vector
		 *
		 * @return diagonal part of gradient with respect to parameter
		 */
		virtual SGVector<float64_t> get_parameter_gradient_diagonal(
				Parameters::const_reference param, index_t index=-1)
		{
			return get_parameter_gradient(param,index).get_diagonal_vector();
		}
#endif

		/** Obtains a kernel from a generic SGObject with error checking. Note
		 * that if passing NULL, result will be NULL
		 * @param kernel Object to cast to Kernel, is *not* SG_REFed
		 * @return object casted to Kernel, NULL if not possible
		 */
		static std::shared_ptr<Kernel> obtain_from_generic(const std::shared_ptr<SGObject>& kernel);
	protected:
		/** set property
		 *
		 * @param p kernel property to set
		 */
		inline void set_property(EKernelProperty p)
		{
			properties |= p;
		}

		/** unset property
		 *
		 * @param p kernel property to unset
		 */
		inline void unset_property(EKernelProperty p)
		{
			properties &= (properties | p) ^ p;
		}

		/** set is initialized
		 *
		 * @param p_init if optimization shall be set to initialized
		 */
		inline void set_is_initialized(bool p_init) { optimization_initialized=p_init; }

		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * abstract base method
		 *
		 * @param x index a
		 * @param y index b
		 * @return computed kernel function at indices a,b
		 */
		virtual float64_t compute(int32_t x, int32_t y)=0;

		/** compute row start offset for parallel kernel matrix computation
		 *
		 * @param offs offset
		 * @param n number of columns
		 * @param symmetric whether matrix is symmetric
		 */
		int32_t compute_row_start(int64_t offs, int32_t n, bool symmetric)
		{
			int32_t i_start;

			if (symmetric)
				i_start = (int32_t)Math::floor(
				    n - std::sqrt(Math::sq((float64_t)n) - offs));
			else
				i_start=(int32_t) (offs/int64_t(n));

			return i_start;
		}

		/** helper for computing the kernel matrix in a parallel way
		 *
		 * @param p thread parameters
		 */
		template <class T> static void* get_kernel_matrix_helper(void* p);

		/** Can (optionally) be overridden to post-initialize some member
		 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
		 *  first the overridden method BASE_CLASS::LOAD_SERIALIZABLE_POST
		 *  is called.
		 *
		 *  @exception ShogunException Will be thrown if an error
		 *                             occurres.
		 */
		void load_serializable_post() override;

		/** Can (optionally) be overridden to pre-initialize some member
		 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
		 *  first the overridden method BASE_CLASS::SAVE_SERIALIZABLE_PRE
		 *  is called.
		 *
		 *  @exception ShogunException Will be thrown if an error
		 *                             occurres.
		 */
		void save_serializable_pre() override;

		/** Can (optionally) be overridden to post-initialize some member
		 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
		 *  first the overridden method BASE_CLASS::SAVE_SERIALIZABLE_POST
		 *  is called.
		 *
		 *  @exception ShogunException Will be thrown if an error
		 *                             occurres.
		 */
		void save_serializable_post() override;

		/** Separate the function of parameter registration
		 *	This can be the first stage of a *general* framework for
		 *	cross-validation or other parameter-based operations
		 */
		virtual void register_params();

	private:
		/** Do basic initialisations like default settings
		 * and registering parameters */
		void init();


#ifdef USE_SVMLIGHT
#ifndef DOXYGEN_SHOULD_SKIP_THIS
		/**@ cache kernel evalutations to improve speed */
		struct KERNEL_CACHE {
			/** index */
			int32_t   *index;
			/** inverse index */
			int32_t   *invindex;
			/** active2totdoc */
			int32_t   *active2totdoc;
			/** totdoc2active */
			int32_t   *totdoc2active;
			/** least recently used */
			int32_t   *lru;
			/** occu */
			int32_t   *occu;
			/** elements */
			int32_t   elems;
			/** max elements */
			int32_t   max_elems;
			/** time */
			int32_t   time;
			/** active num */
			int32_t   activenum;

			/** buffer */
			KERNELCACHE_ELEM  *buffer;
			/** buffer size */
			KERNELCACHE_IDX   buffsize;
		};

		/** kernel thread parameters */
		struct S_KTHREAD_PARAM
		{
			/** kernel */
			Kernel* kernel;
			/** kernel cache */
			KERNEL_CACHE* kernel_cache;
			/** cache */
			KERNELCACHE_ELEM** cache;
			/** uncached rows */
			int32_t* uncached_rows;
			/** number of uncached rows */
			int32_t num_uncached;
			/** needs computation */
			uint8_t* needs_computation;
			/** start */
			int32_t start;
			/** end */
			int32_t end;
			/** of vectors */
			int32_t num_vectors;
		};
#endif // DOXYGEN_SHOULD_SKIP_THIS

		//@{
		static void* cache_multiple_kernel_row_helper(void* p);

		/// init kernel cache of size megabytes
		void   kernel_cache_free(int32_t cacheidx);
		int32_t   kernel_cache_malloc();
		int32_t   kernel_cache_free_lru();
		KERNELCACHE_ELEM *kernel_cache_clean_and_malloc(int32_t cacheidx);
#endif //USE_SVMLIGHT
		//@}

	protected:
		/// cache_size in MB
		int32_t cache_size;

#ifdef USE_SVMLIGHT
		/// kernel cache
		KERNEL_CACHE kernel_cache;
#endif //USE_SVMLIGHT

		/// this *COULD* store the whole kernel matrix
		/// usually not applicable / necessary to compute the whole matrix
		KERNELCACHE_ELEM* kernel_matrix;

		/// feature vectors to occur on left hand side
		std::shared_ptr<Features> lhs;
		/// feature vectors to occur on right hand side
		std::shared_ptr<Features> rhs;

		/// lhs
		bool lhs_equals_rhs;

		/// number of feature vectors on left hand side
		int32_t num_lhs;
		/// number of feature vectors on right hand side
		int32_t num_rhs;

		/** combined kernel weight */
		float64_t combined_kernel_weight;

		/** if optimization is initialized */
		bool optimization_initialized;
		/** optimization type (currently FASTBUTMEMHUNGRY and
		 * SLOWBUTMEMEFFICIENT)
		 */
		EOptimizationType opt_type;

		/** kernel properties */
		uint64_t  properties;

		/** normalize the kernel(i,j) function based on this normalization
		 * function */
		std::shared_ptr<KernelNormalizer> normalizer;
};

}
#endif /* _KERNEL_H__ */
