#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/UniformIntDistribution.h>

#include <benchmark/benchmark.h>
#include <stan/math/rev/scal.hpp>
#include <unsupported/Eigen/AutoDiff>

#include <random>

namespace shogun
{
	static SGMatrix<float64_t> createRandomData(const benchmark::State& state)
	{
		std::random_device rd;
		std::mt19937_64 prng(rd());
		UniformIntDistribution<index_t> uniform_int_dist(0, 1);
	 
		index_t num_dim = state.range(1);
		index_t num_vecs = state.range(0);

		SGMatrix<float64_t> mat(num_dim, num_vecs);
		for (index_t i=0; i<num_vecs; i++)
		{
			for (index_t j=0; j<num_dim; j++)
			{
				mat(j,i) = uniform_int_dist(prng) + 0.5;
			}
		}
		return mat;
	}		

	void BM_Autodiff(benchmark::State& state)
	{
	    for (auto _ : state)
		{
	        SGMatrix<float64_t> mat = createRandomData(state);

			SGMatrix<float64_t> derivative=SGMatrix<float64_t>(state.range(0), state.range(1));
	        auto feat = new CDenseFeatures(mat);
	        auto dist = CEuclideanDistance(feat, feat);
			
			stan::math::var x = 1.0;
			stan::math::var log_width = 0.0;	
			auto f = exp(-x / exp(log_width * 2.0) * 2.0);

			for (int k=0; k<state.range(1); k++)
			{
				for (int j=0; j<state.range(0); j++)
				{
					const double& tmp_const = x->val_;
					double* tmp = const_cast<double*>(&tmp_const);
					*tmp = dist.distance(j, k);
					f.grad();
					derivative(j, k) = log_width.adj();
					stan::math::set_zero_all_adjoints();
				}
			}
		}
	}

	void BM_Autodiff_eigen(benchmark::State& state)
	{
	    for (auto _ : state)
		{
			using EigenScalar = Eigen::Matrix<double, 1, 1>;
	        SGMatrix<float64_t> mat = createRandomData(state);

			SGMatrix<float64_t> derivative=SGMatrix<float64_t>(state.range(0), state.range(1));
	        auto feat = new CDenseFeatures(mat);
	        auto dist = CEuclideanDistance(feat, feat);
			
			Eigen::AutoDiffScalar<EigenScalar> eigen_log_width = 0.0;
			eigen_log_width.derivatives() = Eigen::VectorXd::Unit(1,0);

			for (int k=0; k<state.range(1); k++)
			{
				for (int j=0; j<state.range(0); j++)
				{
					auto x = dist.distance(j, k);
					Eigen::AutoDiffScalar<EigenScalar> kernel = exp(-x / exp(eigen_log_width * 2.0) * 2.0);
					derivative(j, k) = kernel.derivatives()(0);
				}
			}
		}
	}

	// struct SE {
	// 	const Eigen::Matrix<stan::math::var,Eigen::Dynamic,Eigen::Dynamic>& dist_;
	// 	SE(const Eigen::Matrix<stan::math::var,Eigen::Dynamic,Eigen::Dynamic>& dist): dist_(dist) {}
	// 	template <typename T>
	// 	T operator(const )
	// }

	// void BM_Autodiff_vectorized(benchmark::State& state)
	// {
	//     for (auto _ : state)
	// 	{
	//         SGMatrix<float64_t> mat = createRandomData(state);

	// 		SGMatrix<float64_t> derivative=SGMatrix<float64_t>(state.range(0), state.range(1));
	//         auto feat = new CDenseFeatures(mat);
	//         auto dist = CEuclideanDistance(feat, feat);
	//         Eigen::Matrix<stan::math::var,Eigen::Dynamic,Eigen::Dynamic> dist_var(state.range(0), state.range(1));
	//         for (int j=0; j<state.range(0); j++)
	// 			for (int k=0; k<state.range(1); k++)
	// 				dist_var(j, k) = dist.distance(j, k);

	// 		Eigen::Matrix<stan::math::var,Eigen::Dynamic,Eigen::Dynamic> log_width(state.range(0), state.range(1));	
	// 		auto constant_part = exp(log_width * 2.0) * 2.0;
	// 		auto gk = -dist_var / constant_part;
	// 		gk.grad();

	// 		for (int k=0; k<state.range(1); k++)
	// 		{
	// 			for (int j=0; j<state.range(0); j++)
	// 			{
	// 				derivative(j, k) = log_width(j, k).adj();
	// 			}
	// 		}
	// 	}
	// }

	void BM_Manualdiff(benchmark::State& state)
	{
	    for (auto _ : state)
		{
	        SGMatrix<float64_t> mat = createRandomData(state);
	        auto feat = new CDenseFeatures(mat);
	        auto dist = CEuclideanDistance(feat, feat);
	        auto width = std::exp(0.0 * 2.0) * 2.0;
		
			SGMatrix<float64_t> derivative=SGMatrix<float64_t>(state.range(0), state.range(1));
			
			for (int k=0; k<state.range(1); k++)
			{
	#pragma omp parallel for
				for (int j=0; j<state.range(0); j++)
				{
					auto el = dist.distance(j, k) / width;
					derivative(j, k) = std::exp(-el) * el * 2.0;
				}
			}
		}
	}
}

BENCHMARK(BM_Autodiff)->RangeMultiplier(4)->Ranges({{64, 2<<10}, {4, 16}})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Autodiff_eigen)->RangeMultiplier(4)->Ranges({{64, 2<<10}, {4, 16}})->Unit(benchmark::kMillisecond);
// BENCHMARK(BM_Autodiff_vectorized)->RangeMultiplier(4)->Ranges({{64, 2<<10}, {4, 16}})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Manualdiff)->RangeMultiplier(4)->Ranges({{64, 2<<10}, {4, 16}})->Unit(benchmark::kMillisecond);