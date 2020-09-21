/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 */
#include<string_view>
#include<variant>
#include<shogun/mathematics/RandomMixin.h>
#include<shogun/mathematics/UniformRealDistribution.h>
#include<shogun/base/SGObject.h>
namespace shogun
{
    /**
     * @brief: class for search space dimensions, which contains 
     * basis information for bayesian optimization.
     */
    class Dimension : public RandomMixin<Dimension> {
    public:
        Dimension(std::pair<float64_t, float64_t> bound, std::string_view param_name,
            std::string_view prior = "uniform", std::string_view param_type = "Real") :
            m_bound(bound), m_prior(prior),
            m_param_type(param_type) {}

        Dimension(std::vector<std::shared_ptr<SGObject>> objects, std::string_view param_name,
            std::string_view prior = "uniform", std::string_view param_type = "Categority") :
             m_bound(bound), m_prior(prior), m_param_type(param_type) {}

         //unnormalize the value in [0, 1] to origianl space [lower, high]
        std::variant<float64_t, std::shared_ptr<SGObject>> unnormolize(float64_t value)  
        {
            if(auto v = std::get_if<sgobjects_t>(m_bound)) 
            {
                //[0, v.size() - 1] for categorical type
                int idx = value * (v.size() - 1);
                return v[idx];
            }
            else 
            {
                return value * ( m_upper_bound - m_lower_bound) + m_lower_bound;
            }  
        }
        
        //this function aims to select the initial point
        std::pair<string_view, float64_t> random_samples()
        {
            if(m_prior == "uniform") {
                auto uniform = UniformRealDistribution<>(0.0, 1.0);
                return std::make_pair(m_param_name, uniform);
            } 
            //Todo log-uniform
        }

    private:
        using sgobjects_t = std::vector<std::shared_ptr<SGObject>>;
        //float64_t m_lower_bound, m_upper_bound;
        //{ConstKernel, GaussianKernel}
        std::variant<std::pair<float64_t, float64_t>, sgobjects_t> m_bound;
        std::string_view m_prior, m_param_name, m_param_type;
    };

    using param_type1 = std::tuple<float64_t, float64_t>;
    using param_type2 = std::tuple<float64_t, float64_t, std::string_view>;
    using param_type3 = std::tuple<float64_t, float64_t, std::string_view, 
        std::string_view>;
    using param_type4 = std::tuple<float64_t, float64_t, std::string_view, 
        std::string_view, std::string_view>;
    using param_type5 = std::tuple<std::shared_ptr<SGObject>, std::string_view, 
        std::variant<param_type1, param_type2, param_type3, param_type4>>;
    using param = std::pair<std::string_view, std::variant<param_type1, 
        param_type2, param_type3, param_type4>>>;
    class Space {
        
    public:
        Space(std::vector<std::pair<string_view, >> s) {
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
        // this convert function aims to flatten parameters, such as we have 
        // a list of parameters, and one of the parameters is kernel, which have 
        // other parameters, we should flatten those parameters to a linear structure,
        // so we can do bayesian optimization.
        vector<Dimension> convert(vector<_> ) {
        /*convert {("C1", (1.0, 4.0)),
                ("C2", (1.0, 4.0)),
                ("kernel", ("GaussianKernel", "width" (1.0, 2.0),
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
