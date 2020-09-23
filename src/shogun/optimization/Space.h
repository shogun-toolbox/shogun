/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 */
#include <shogun/base/SGObject.h>
#include <shogun/mathematics/RandomMixin.h>
#include <shogun/mathematics/UniformRealDistribution.h>
#include <shogun/util/traits.h>
#include <shogun/lib/SGVector.h>
#include <string>
#include <variant>
namespace shogun
{
	/**
	 * @brief: class for search space dimensions, which contains
	 * basis information for bayesian optimization.
	 */
	class Dimension : public SGObject
	{
	public:
        Dimension() {}
		Dimension(
		    std::string param_name, std::pair<float64_t, float64_t> bound,
		    std::string prior = "uniform", std::string param_type = "Real")
		    : m_param_name(param_name), m_bound(bound), m_prior(prior),
		      m_param_type(param_type)
		{
		}

		Dimension(
		    std::string param_name,
		    std::vector<std::shared_ptr<SGObject>> objects,
		    std::string prior = "uniform",
		    std::string param_type = "Categority")
		    : m_param_name(param_name), m_bound(objects), m_prior(prior),
		      m_param_type(param_type)
		{
		}

        Dimension(const Dimension& other) : m_param_name(other.m_param_name), m_bound(other.m_bound),
            m_prior(other.m_prior), m_param_type(other.m_param_type) {}
		// unnormalize the value in [0, 1] to origianl space [lower, high]
		std::variant<float64_t, std::shared_ptr<SGObject>>
		unnormolize(float64_t value)
		{
			if (std::holds_alternative<sgobjects_t>(m_bound))
			{
				//[0, v.size() - 1] for categorical type
                auto v = std::get<sgobjects_t>(m_bound);
				int idx = value * (v.size() - 1);
				return v[idx];
			}
			else
			{
                auto [lower_bound, upper_bound] = std::get<bound_pair>(m_bound);
				return value * (upper_bound - lower_bound) + lower_bound;
			}
		}

        virtual const char* get_name() const 
        {
            return "Dimension";
        }
		// this function aims to select the initial point
		std::pair<std::string, float64_t> random_samples()
		{
            std::mt19937_64 prng(0);
			if (m_prior == "uniform")
			{
				auto uniform = UniformRealDistribution<>(0.0, 1.0);
				return std::make_pair(m_param_name, uniform(prng));
			}
			// Todo log-uniform
		}

	public:
		using sgobjects_t = std::vector<std::shared_ptr<SGObject>>;
		using bound_pair = std::pair<float64_t, float64_t>;
        std::string m_param_name;
        //such as float64_t m_lower_bound, m_upper_bound;
		//{ConstKernel, GaussianKernel}
		std::variant<bound_pair, sgobjects_t> m_bound;
		std::string m_prior, m_param_type;
	};

	using sgobject_t = std::shared_ptr<SGObject>;
	struct DimensionWithSGObject : public SGObject
	{
        DimensionWithSGObject() {}
		DimensionWithSGObject(
		    std::pair<
		        std::string, std::vector<std::pair<sgobject_t, Dimension>>>
		        params)
		    : sgobject_name(params.first), m_params(params.second)
		{
		}
        virtual const char* get_name() const 
        {
            return "DimensionWithSGObject";
        }
		std::vector<std::pair<sgobject_t, Dimension>> m_params;
		std::string sgobject_name;
	};

    /**
	 * @brief: The space aim to represent a search space from given specifications.
	 */
	class Space : public SGObject
	{

	public:
        Space() {}
		Space(std::vector<std::variant<Dimension, DimensionWithSGObject>> s)
		{
			convert(s);
		}

        virtual const char* get_name() const 
        {
            return "Space";
        }

		std::pair<std::vector<std::string>, SGVector<float64_t>>
		random_samples()
		{
			std::vector<std::string> param_names;
			SGVector<float64_t> param_values(m_dimensions.size());
			int idx = 0;
			for (auto dim : m_dimensions)
			{
				auto [param_name, param_value] = dim.random_samples();
				param_names.push_back(param_name);
				param_values[idx++] = param_value;
			}
			return std::make_pair(param_names, param_values);
		}

	private:
		// this convert function aims to flatten parameters, such as we have
		// a list of parameters, and one of the parameters is kernel, which have
		// other parameters, we should flatten those parameters to a linear
		// structure, so we can do bayesian optimization.
		/*convert {("C1", (1.0, 4.0)),
		       ("C2", (1.0, 4.0)),
		       ("kernel", ("GaussianKernel", ("width" (1.0, 2.0)),
		                   ("ConstKernel", "const_value" (1.0, 2.0))))}
	   to
	   {("C1", (1.0, 4.0)),
	   ("C2", (1.0, 4.0)),
	   ("GaussianKernel::width", (1.0, 2.0)),
	   ("ConstKernel::const_value", (1.0, 2.0)),
	   ("kernel": ("GaussianKernel", "ConstKernel"))}*/
		void
		convert(std::vector<std::variant<Dimension, DimensionWithSGObject>> s)
		{
			for (auto& v : s)
			{
				std::visit(
				    overloaded{
				        [&](Dimension dim) { m_dimensions.push_back(dim); },
				        [&](DimensionWithSGObject dim) {
					        std::vector<sgobject_t> sgobjects;
					        std::string name = dim.sgobject_name;
					    std::transform(dim.m_params.begin(), dim.m_params.end(),
                            std::back_inserter(sgobjects), 
                                [](std::pair<sgobject_t , Dimension> p) {
						            return p.first;});
                        m_dimensions.emplace_back(name, sgobjects);

                        std::for_each(dim.m_params.begin(), dim.m_params.end(),
                            [&](std::pair<sgobject_t , Dimension> d) {
						        auto& p = d.second;
                                std::string name = std::string(d.first->get_name()) + "::" ;
                                std::string new_name = name + p.m_param_name;
						        p.m_param_name = std::move(new_name);
                                m_dimensions.emplace_back(p);
						    });
				        }},
				    v);
			}
		}
		std::vector<Dimension> m_dimensions;
	};
} // namespace shogun
