#include <benchmark/benchmark.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/iterators/DotIterator.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/mathematics/Math.h>

namespace shogun
{
	// Mock PerceptronP
	class PerceptronP : public CSGObject
	{

	public:
		PerceptronP() : CSGObject()
		{
			max_iters = 1000;
			learn_rate = 0.1;
			SG_ADD(&w, "w", "weights", MS_AVAILABLE);
		}

		~PerceptronP()
		{
		}

		bool train(CDotFeatures* feats, CBinaryLabels* labels, bool use_put)
		{
			int32_t num_feat = feats->get_dim_feature_space();
			SGVector<float64_t> weights(num_feat);
			weights.set_const(1.0 / num_feat);
			bias = 0;

			SGVector<float64_t> output(feats->get_num_vectors());
			for (auto iter : range(max_iters))
			{
				auto iter_train_labels = labels->get_int_labels().begin();
				auto iter_output = output.begin();

				for (const auto& v : DotIterator(feats))
				{
					const auto true_label = *(iter_train_labels++);
					auto& predicted_label = *(iter_output++);

					predicted_label = v.dot(weights) + bias;

					if (CMath::sign<float64_t>(predicted_label) != true_label)
					{
						const auto gradient = learn_rate * true_label;
						bias += gradient;
						v.add(gradient, weights);
					}
					if (use_put)
					{
						this->put<SGVector<float64_t>>("w", weights);
					}
					else
						w = weights;
				}
			}
		}

		virtual const char* get_name() const
		{
			return "PerceptronP";
		}

	protected:
		/** learning rate */
		float64_t bias;
		SGVector<float64_t> w;
		float64_t learn_rate;
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
			index_t num_vecs = 1000;
			SGMatrix<float64_t> mat(num_dim, num_vecs);
			for (index_t i = 0; i < num_vecs; i++)
			{
				for (index_t j = 0; j < num_dim; j++)
				{
					mat(j, i) = CMath::normal_random(0.0, 1.0);
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
				vec[i] = CMath::random(0, 1) > 0.5 ? -1 : 1;
			}
			labels = new CBinaryLabels(vec);
			SG_REF(labels);
		}
	};

	BENCHMARK_DEFINE_F(DataFixture, Put_perceptron)(benchmark::State& st)
	{
		for (auto _ : st)
		{
			auto perceptron = new PerceptronP();
			SG_REF(perceptron);
			perceptron->train(feats, labels, st.range(1));
			SG_UNREF(perceptron);
		}
	}

	BENCHMARK_REGISTER_F(DataFixture, Put_perceptron)
	    ->Ranges({{8, 8 << 10}, {0, 1}})
	    ->Unit(benchmark::kMillisecond);
}
