/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 */
#include<string>
#include<variant>
#include<shogun/mathematics/RandomMixin.h>
#include<shogun/mathematics/UniformRealDistribution.h>
#include<shogun/base/SGObject.h>
#include<shogun/util/traits.h>
namespace shogun
{
    /**
     * @brief: class for search space dimensions, which contains 
     * basis information for bayesian optimization.
     */
    class Dimension : public RandomMixin<Dimension> {
    public:
        Dimension(std::string param_name, std::pair<float64_t, float64_t> bound, 
            std::string prior = "uniform", std::string param_type = "Real") :
            m_bound(bound), m_prior(prior),
            m_param_type(param_type) {}

        Dimension(std::string param_name, std::vector<std::shared_ptr<SGObject>> objects, 
            std::string prior = "uniform", std::string param_type = "Categority") :
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
        std::pair<string, float64_t> random_samples()                                 
        {
            if(m_prior == "uniform") {
                auto uniform = UniformRealDistribution<>(0.0, 1.0);
                return std::make_pair(m_param_name, uniform);
            } 
            //Todo log-uniform
        }

    public:
        using sgobjects_t = std::vector<std::shared_ptr<SGObject>>;
        //float64_t m_lower_bound, m_upper_bound;
        //{ConstKernel, GaussianKernel}
        std::variant<std::pair<float64_t, float64_t>, sgobjects_t> m_bound;
        std::string m_prior, m_param_name, m_param_type;
    };

    using sgobject_t = std::shared_ptr<SGObject>;
    struct DimensionWithSGObject : public SGObject 
    {
        DimensionWithSGObject(std::pair<std::string, std::vector<std::pair<sgobject_t , Dimension>>> params)
            :sgobject_name(params.first), m_params(params.second)
            {}
        
        std::vector<std::pair<sgobject_t , Dimension>> m_params;
        std::string sgobject_name;
    }; 
    class Space {
        
    public:
        Space(std::vector<std::variant<Dimension, DimensionWithSGObject>> s) {
            m_dimensions = convert(s);   
        }

        std::tuple<std::vector<std::string>, SGVector<float64_t>>random_samples() {
            std::vector<std::string> param_names;
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
        void convert(std::vector<std::variant<Dimension, DimensionWithSGObject>> s ) {
            for(auto v : s) {
                std::visit(overloaded {
                    [&](Dimension dim) {m_dimensions.push_back(dim);}
                    [&](DimensionWithSGObject dim) {
                        std::vector<SGObject> sgobjects;
                        string name = dim.sgobject_name;
                        auto params = std::transform(dim.m_params.begin(), dim.m_params.end(),
                         std::back_inserter(sgobjects), [](std::pair<sgobject_t , Dimension> p) {
                             return p.first;
                         };
                         
                    }
                }, v)
            }
            
        }
        vector<Dimension> m_dimensions; //X^m,where m represents the numbers of parameters, each dimension contains the constrict on this dimension.
    };
