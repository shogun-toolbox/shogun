#include <shogun/base/init.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalgrefactor/linalgRefactor.h>

#include <shogun/mathematics/eigen3.h>
#include <viennacl/linalg/inner_prod.hpp>
#include <viennacl/vector.hpp>

#include <algorithm>
#include <memory>
#include <hayai/hayai.hpp>

#include <iostream>

using namespace shogun;

/**
 * Instructions :
 * 1. Install benchmarking toolkit "hayai" (https://github.com/nickbruun/hayai)
 * 2. Compile against libhayai_main, e.g.
 * 		g++ -O3 -std=c++11 linglg_refactor_benchmark.cpp -I/usr/include/eigen3 \
 *		-lshogun -lhayai_main -lOpenCL -o benchmark
 * 3. ./benchmark
 */

/** Generate data only once */
typedef float32_t T;
typedef viennacl::vector_base<T, std::size_t, std::ptrdiff_t> VCLVectorBase;

template<typename value_type>
struct Data
{
	typedef viennacl::backend::mem_handle VCLMemoryArray;

	Data()
	{
		num_rows = 100;
		init();
	}

	Data(index_t num_rows)
	{
		this->num_rows = num_rows;
		init();
	}

	~Data()
	{
		exit_shogun();
	}

	void init()
	{
		A = init_sg(1);
		B = init_sg(2);
		Av = init_v(A);
		Bv = init_v(B);
		Ac = std::unique_ptr<BaseVector<value_type>> (init_c(A));
		Bc = std::unique_ptr<BaseVector<value_type>> (init_c(B));
		Ag = std::unique_ptr<BaseVector<value_type>> (init_g(A));
		Bg = std::unique_ptr<BaseVector<value_type>> (init_g(B));

		init_shogun_with_defaults();
		GPUBackend viennaclBackend;
		sg_linalg->set_gpu_backend(&viennaclBackend);
	}

	/** SGVector **/
	SGVector<value_type> init_sg(value_type begin)
	{
		SGVector<value_type> m(num_rows);
		m.range_fill(begin);

		return m;
	}

	/** ViennaCL Vector for test **/
	VCLVectorBase init_v(SGVector<value_type> m)
	{
		VCLVectorBase mv;
		std::shared_ptr<VCLMemoryArray> vector(new VCLMemoryArray());
		viennacl::backend::memory_create(*vector, sizeof(value_type)*num_rows,
			viennacl::context());
		viennacl::backend::memory_write(*vector, 0, num_rows*sizeof(value_type),
			m.vector);
		mv = VCLVectorBase(*vector, num_rows, 0, 1);

		return mv;
	}

	/** CPUVector derived from BaseVector **/
	std::unique_ptr<BaseVector<value_type>> init_c(SGVector<value_type> m)
	{
		std::unique_ptr<CPUVector<value_type>> mc(new CPUVector<value_type>(m));
		return std::move(mc);
	}

	/** GPUVector derived from BaseVector **/
	std::unique_ptr<BaseVector<value_type>> init_g(SGVector<value_type> m)
	{
		std::unique_ptr<GPU_Vector<value_type>> mg(new GPU_Vector<value_type>(m));
		return std::move(mg);
	}

	SGVector<value_type> A;
	SGVector<value_type> B;
	VCLVectorBase Av;
	VCLVectorBase Bv;
	std::unique_ptr<BaseVector<value_type>> Ac;
	std::unique_ptr<BaseVector<value_type>> Bc;
	std::unique_ptr<BaseVector<value_type>> Ag;
	std::unique_ptr<BaseVector<value_type>> Bg;

	index_t num_rows;
};

BENCHMARK_P(CPUVector, dot_explict_eigen3, 10, 1000,
	(const SGVector<T> &A, const SGVector<T> &B))
{
	typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
	Eigen::Map<VectorXt> vec_A(A.vector, A.vlen);
	Eigen::Map<VectorXt> vec_B(B.vector, B.vlen);

    	auto C = vec_A.dot(vec_B);
}

BENCHMARK_P(CPUVector, dot_eigen3, 10, 1000,
	(BaseVector<T> *A, BaseVector<T> *B))
{
	auto C = sg_linalg->dot(A, B);
}

BENCHMARK_P(GPU_Vector, dot_explict_viennacl, 10, 1000,
	(const VCLVectorBase &A, const VCLVectorBase &B))
{
	auto C = viennacl::linalg::inner_prod(A, B);
}

BENCHMARK_P(GPU_Vector, dot_viennacl, 10, 1000,
	(BaseVector<T> *A, BaseVector<T> *B))
{
	auto C = sg_linalg->dot(A, B);
}

Data<T> data(1000000);
BENCHMARK_P_INSTANCE(CPUVector, dot_explict_eigen3, (data.A, data.B));
BENCHMARK_P_INSTANCE(CPUVector, dot_eigen3, (data.Ac.get(), data.Bc.get()));
BENCHMARK_P_INSTANCE(GPU_Vector, dot_explict_viennacl, (data.Av, data.Bv));
BENCHMARK_P_INSTANCE(GPU_Vector, dot_viennacl, (data.Ag.get(), data.Bg.get()));
