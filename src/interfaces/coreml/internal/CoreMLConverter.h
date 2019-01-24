#ifndef __COREML_CONVERTER_H__
#define __COREML_CONVERTER_H__

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <sstream>

#include <shogun/machine/Machine.h>

#include "../CoreMLModel.h"
#include "format/Model.pb.h"

namespace shogun
{
	namespace coreml
	{
		static constexpr int32_t SPECIFICATION_VERSION = 1;

		class ICoreMLConverter
		{
		public:
			virtual std::shared_ptr<CoreML::Specification::Model> description() const = 0;
		};

		template <class I, class O>
		class CoreMLConverter: public ICoreMLConverter
		{
		private:
			static constexpr const char* kInputName = "input";
			static constexpr const char* kPredictionName = "prediction";
			static constexpr const char* kPredictionProbabilitiesName = "prediction";

		public:
			typedef I input_type;
			typedef O output_type;
			static const std::unordered_set<std::string> supported_types;

			CoreMLConverter(const CMachine* m):
				m_spec(std::make_shared<CoreML::Specification::Model>()),
				m_machine(m)
			{
				REQUIRE(m_machine != nullptr, "No machine has been provided")
			}

			std::shared_ptr<CoreML::Specification::Model> description() const override
			{
				return m_spec;
			}

			O* convert(const I* m)
			{
				REQUIRE(m != nullptr, "No machine has been provided")
				REQUIRE(supported_types.find(m->get_name()) != supported_types.end(),
					"Exporting %s to CoreML format is not supported!", m->get_name())

				auto spec = new O();
				try
				{
					convert(m, spec);
				}
				catch(const std::runtime_error& e)
				{
					delete spec;
					throw;
				}
				return spec;
			}
		protected:
			void convert(const I*, O*);

			virtual ::CoreML::Specification::FeatureType* input_feature_type() const = 0;

			virtual void set_model_interface()
			{
				auto description = m_spec->mutable_description();

				// set input
				auto input = description->add_input();
				input->set_name(kInputName);
				input->set_allocated_type(input_feature_type());

				// set output
				auto output = description->add_output();
				output->set_name(kPredictionName);

				switch(m_machine->get_machine_problem_type())
				{
					case PT_REGRESSION:
						output->mutable_type()->mutable_doubletype();
						break;
					case PT_BINARY:
					case PT_MULTICLASS:
						output->mutable_type()->mutable_int64type();
						break;
					default:
						std::stringstream ss;
						ss << "Unsupported problem type: " << m_machine->get_machine_problem_type() << "!" << std::endl;
						throw std::runtime_error(ss.str());
				}
				description->set_predictedfeaturename(kPredictionName);

				//FIXME
				//description->set_predictedprobabilitiesname(kPredictionProbabilitiesName);
			}

			std::shared_ptr<CoreML::Specification::Model> m_spec;
			const CMachine* m_machine;
		};

		class ConverterFactory
		{
			typedef std::function<std::shared_ptr<ICoreMLConverter>(const CMachine* m)> ConverterFactoryFunction;
		public:

			auto size() const
			{
				return m_registry.size();
			}

			auto register_converter(const std::string& machine_name, ConverterFactoryFunction f)
			{
				return m_registry.emplace(std::make_pair(machine_name, f)).second;
			}

			std::shared_ptr<ICoreMLConverter> operator()(const CMachine* m)
			{
				std::string machine_name(m->get_name());
				auto f = m_registry.find(machine_name);
				if (f == m_registry.end())
					throw std::runtime_error("The provided machine cannot be converted to CoreML format!");
				return f->second(m);
			}

			static ConverterFactory* instance()
			{
				static ConverterFactory* f = new ConverterFactory();
				return f;
			}

		private:
			std::unordered_map<std::string, ConverterFactoryFunction> m_registry;
		};

#define REGISTER_COREML_CONVERTER4(factory, classname, machines, function) \
	static int register_converter##classname = []() { \
		for (auto m: machines) \
			factory->register_converter(m, function); \
		return factory->size(); \
	}();

#define REGISTER_COREML_CONVERTER3(factory, classname, machines) \
	REGISTER_COREML_CONVERTER4(factory, classname, machines, [](const CMachine* _m) { return std::make_shared<classname>(_m); })

#define REGISTER_CONVERTER(classname, ...) \
	VARARG(REGISTER_COREML_CONVERTER, ConverterFactory::instance(), classname, __VA_ARGS__)

	}
}

#endif
