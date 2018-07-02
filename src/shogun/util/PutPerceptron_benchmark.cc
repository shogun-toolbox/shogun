#include <random>
#include <benchmark/benchmark.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/iterators/DotIterator.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/lib/parameter_observers/ParameterObserverLogger.h>
#include <shogun/lib/parameter_observers/ObservedValue.h>

namespace shogun
{

	std::normal_distribution<float64_t> normal;
	std::random_device rd;
	std::mt19937 gen{rd()};

	// Mock PerceptronP
	class CMockPerceptron : public CSGObject
	{

	public:
		CMockPerceptron() : CSGObject()
		{
			max_iters = 1000;
			SG_ADD(&w, "w", "weights", MS_AVAILABLE);
		}

		~CMockPerceptron()
		{
		}

		bool train(CDotFeatures* feats, CBinaryLabels* labels)
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

					if (CMath::sign<float64_t>(predicted_label) != true_label)
					{
						const auto gradient = true_label;
						bias += gradient;
						v.add(gradient, weights);
					}
				}
				m_callback();
			}
		}

	public:
		void observe_dispatcher(const ObservedValue& value)
		{
			CSGObject::observe(value);
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
		CDotFeatures* feats;
		CBinaryLabels* labels;

		void SetUp(const ::benchmark::State& st)
		{
			createNormalFeatures(st);
			createLabels(st);
		}

		void TearDown(const ::benchmark::State&)
		{
			SG_UNREF(feats);
			SG_UNREF(labels);
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
			feats = new CDenseFeatures<float64_t>(mat);
			SG_REF(feats);
		}

		void createLabels(const benchmark::State& state)
		{
			index_t num_dim = state.range(0);
			SGVector<float64_t> vec(num_dim);

			for (index_t i = 0; i < num_dim; i++)
			{
				vec[i] = std::rand() % 100 > 50 ? -1 : 1;
			}
			labels = new CBinaryLabels(vec);
			SG_REF(labels);
		}
	};

	BENCHMARK_DEFINE_F(DataFixture, perceptron_baseline)(benchmark::State& st)
	{
		for (auto _ : st)
		{
			auto perceptron = new CMockPerceptron();
			std::function<void()> callback = [&perceptron]() {
				perceptron->w = perceptron->weights;
			};
			perceptron->m_callback = callback;
			perceptron->train(feats, labels);
		}
	}

	BENCHMARK_DEFINE_F(DataFixture, perceptron_with_put)(benchmark::State& st)
	{
		for (auto _ : st)
		{
			auto perceptron = new CMockPerceptron();
			std::function<void()> callback = [&perceptron]() {
				perceptron->put<SGVector<float64_t>>("w", perceptron->weights);
			};
			perceptron->m_callback = callback;
			perceptron->train(feats, labels);
		}
	}

	BENCHMARK_DEFINE_F(DataFixture, perceptron_with_observe)(benchmark::State& st)
	{
		for (auto _ : st)
		{
			auto perceptron = new CMockPerceptron();
			std::function<void()> callback = [&perceptron]() {
				perceptron->w = perceptron->weights;
				auto value = make_any(perceptron->w);
				std::string name = "weights";
				ObservedValue observed_value (1, name, value, LOGGER);
				perceptron->observe_dispatcher(observed_value);
			};
			perceptron->m_callback = callback;
			perceptron->train(feats, labels);
		}
	}

	BENCHMARK_DEFINE_F(DataFixture, perceptron_with_observe_and_observer)(benchmark::State& st)
	{
		for (auto _ : st)
		{
			auto observer = new CParameterObserverLogger();
			auto perceptron = new CMockPerceptron();
			perceptron->subscribe_to_parameters(observer);
			std::function<void()> callback = [&perceptron]() {
				perceptron->w = perceptron->weights;
				auto value = make_any(perceptron->w);
				std::string name = "weights";
				ObservedValue observed_value (1, name, value, LOGGER);
				perceptron->observe_dispatcher(observed_value);
			};
			perceptron->m_callback = callback;
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

	BENCHMARK_REGISTER_F(DataFixture, perceptron_with_observe)
		->Ranges(
		{
			{8, 1 << 8}, // range for dimensions of feature vector
			{8 << 5, 8 << 8} // range for number of feature vectors
		})
		->Unit(benchmark::kMillisecond);

	BENCHMARK_REGISTER_F(DataFixture, perceptron_with_observe_and_observer)
		->Ranges(
		{
			{8, 1 << 8}, // range for dimensions of feature vector
			{8 << 5, 8 << 8} // range for number of feature vectors
		})
		->Unit(benchmark::kMillisecond);

}
