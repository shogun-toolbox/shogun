#include <benchmark/benchmark.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/iterators/DotIterator.h>
#include <shogun/labels/BinaryLabels.h>

namespace shogun
{

	std::normal_distribution<float64_t> normal;
	std::random_device rd;
	std::mt19937 gen{rd()};

	// Mock PerceptronP
	class MockPerceptron : public SGObject
	{

	public:
		MockPerceptron() : SGObject()
		{
			max_iters = 1000;
			SG_ADD(&w, "alladin", "test_param1");
			SG_ADD(&w, "xenia", "test_param2");
			SG_ADD(&w, "ulrika", "test_param3");
			SG_ADD(&w, "ken", "test_param4");
			SG_ADD(&w, "ether", "test_param5");
			SG_ADD(&w, "w", "weights");
		}

		~MockPerceptron()
		{
		}

		bool train(const std::shared_ptr<DotFeatures>& feats, const std::shared_ptr<BinaryLabels>& labels)
		{
			int32_t num_feat = feats->get_dim_feature_space();
			weights = SGVector<float64_t>(num_feat);
			weights.set_const(1.0 / num_feat);
			float64_t bias = 0;

			SGVector<float64_t> output(feats->get_num_vectors());
			for (auto iter : range(max_iters))
			{
				auto iter_train_labels = labels->get_int_labels().begin();

				for (const auto& v : DotIterator(feats))
				{
					const auto true_label = *(iter_train_labels++);

					auto predicted_label = v.dot(weights) + bias;

					if (Math::sign<float64_t>(predicted_label) != true_label)
					{
						const auto gradient = true_label;
						bias += gradient;
						v.add(gradient, weights);
					}
				}
				m_callback();
			}
			return true;
		}

		virtual const char* get_name() const
		{
			return "MockPerceptron";
		}

		std::function<void(void)> m_callback;
		SGVector<float64_t> w;
		SGVector<float64_t> weights;
		int32_t max_iters;
	};

	class DataFixture : public benchmark::Fixture
	{
	public:
		std::shared_ptr<DotFeatures> feats;
		std::shared_ptr<BinaryLabels> labels;

		void SetUp(const ::benchmark::State& st)
		{
			createNormalFeatures(st);
			createLabels(st);
		}

		void TearDown(const ::benchmark::State&)
		{
		}

		void createNormalFeatures(const benchmark::State& state)
		{
			index_t num_dim = state.range(0);
			index_t num_vecs = state.range(1);
			SGMatrix<float64_t> mat(num_dim, num_vecs);
			for (index_t i = 0; i < num_vecs; i++)
			{
				for (index_t j = 0; j < num_dim; j++)
				{
					mat(j, i) = normal(gen);
				}
			}
			feats = std::make_shared<DenseFeatures<float64_t>>(mat);
		}

		void createLabels(const benchmark::State& state)
		{
			index_t num_dim = state.range(0);
			SGVector<float64_t> vec(num_dim);

			for (index_t i = 0; i < num_dim; i++)
			{
				vec[i] = std::rand() % 100 > 50 ? -1 : 1;
			}
			labels = std::make_shared<BinaryLabels>(vec);
		}
	};

	BENCHMARK_DEFINE_F(DataFixture, perceptron_baseline)(benchmark::State& st)
	{
		for (auto _ : st)
		{
			st.PauseTiming();
			auto perceptron = std::make_shared<MockPerceptron>();
			std::function<void()> callback = [&perceptron]() {
				perceptron->w = perceptron->weights;
			};
			perceptron->m_callback = callback;
    			st.ResumeTiming();
			perceptron->train(feats, labels);
		}
	}

	BENCHMARK_DEFINE_F(DataFixture, perceptron_with_put)(benchmark::State& st)
	{
		for (auto _ : st)
		{
			st.PauseTiming();
			auto perceptron = std::make_shared<MockPerceptron>();
			std::function<void()> callback = [&perceptron]() {
				perceptron->put<SGVector<float64_t>>("w", perceptron->weights);
			};
			perceptron->m_callback = callback;
    			st.ResumeTiming();
			perceptron->train(feats, labels);
		}
	}

	BENCHMARK_REGISTER_F(DataFixture, perceptron_baseline)
	    ->Ranges(
	        {
	            {8, 1 << 8} // range for dimensions of feature vector
	            ,
	            {8 << 5, 8 << 8} // range for number of feature vectors
	        })
	    ->Unit(benchmark::kMillisecond);

	BENCHMARK_REGISTER_F(DataFixture, perceptron_with_put)
	    ->Ranges(
	        {
	            {8, 1 << 8} // range for dimensions of feature vector
	            ,
	            {8 << 5, 8 << 8} // range for number of feature vectors
	        })
	    ->Unit(benchmark::kMillisecond);
}

