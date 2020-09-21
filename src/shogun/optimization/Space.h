/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 */
#include<string_view>
#include<shogun/base/SGObject.h>
#include<shogun/mathematics/UniformRealDistribution.h>

namespace shogun
{
    /**
     * @brief: class for search space dimensions, which contains 
     * basis information for bayesian optimization.
     */
    class Dimension : public SGObject {
    public:
        Dimension(float64_t lower_bound, float64_t upper_bound, std::string_view param_name,
            std::string_view prior = "uniform", std::string_view param_type = "Real") :
            m_lower_bound(lower_bound), m_upper_bound(upper_bound), m_prior(prior),
            m_param_type(param_type) {}

    private:
        float64_t m_lower_bound, m_upper_bound;
        std::string_view m_prior, m_param_name, m_param_type;

        //unnormalize the value in [0, 1] to origianl space [lower, high]
        float64_t unnormolize(float64_t value)  {
            return value * ( m_upper_bound - m_lower_bound) + m_lower_bound;
        }

        std::pair<string_view, float64_t> random_samples(){
            auto uniform = UniformRealDistribution<>(0.0, 1.0);
            return std::make_pair(m_param_name, uniform);
        }
    };

    class Space {
        
    public:
        Space(vector<_> s) {
        m_dimensions = convert(s);   
        }

        std::tuple<std::vector<std::string_view>, SGVector<float64_t>>random_samples() {
            std::vector<std::string_view> param_names;
            SGVector<float64_t> param_values(m_dimensions.size());
            int idx = 0;
            for(auto dim: m_dimensions) {
                auto [param_name, param_value] = dim.random_samples();
                param_names.push_back(param_names);
                param_values[idx++] = param_value;
            }
            return std::make_tuple(param_names, param_values);
        }
    private:
        // this covert function aims to flatten parameters, such as we have 
        // a list of parameters, and one of the parameters is kernel, which have 
        // other parameters, we should flatten those parameters to a linear structure,
        // so we can do bayesian optimization.
        vector<Dimension> convert(vector<_> ) {
        /*convert {("C1", (1.0, 4.0)),
                ("C2", (1.0, 4.0)),
                ("kernel", ("GaussianKernel", :"width" (1.0, 2.0),
                            "ConstKernel", "const_value" (1.0, 2.0)))}
        to
        {("C1", (1.0, 4.0)),
        ("C2", (1.0, 4.0)),
        ("GaussianKernel::width", (1.0, 2.0)), 
        ("ConstKernel::const_value", (1.0, 2.0)),
        ("kernel": ("GaussianKernel", "ConstKernel"))}*/
        }
        vector<Dimension> m_dimensions; //X^m,where m represents the numbers of parameters, each dimension contains the constrict on this dimension.
    };
