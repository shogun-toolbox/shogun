/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_GRAPH_ARRAY_H_
#define SHOGUN_GRAPH_ARRAY_H_

#include <shogun/mathematics/graph/Graph.h>
#include <shogun/mathematics/graph/Tensor.h>
#include <shogun/mathematics/graph/nodes/Add.h>
#include <shogun/mathematics/graph/nodes/Input.h>
#include <shogun/mathematics/graph/nodes/MatMul.h>

namespace shogun
{
	namespace graph
	{
		class LazyExpr
		{

		protected:
			std::unordered_map<
			    std::shared_ptr<node::Input>, std::shared_ptr<Tensor>>
			    m_inputs;
			std::shared_ptr<node::Node> m_output;
			struct Protected
			{
			};

		public:
			LazyExpr(const std::shared_ptr<Tensor>& input_tensor)
			{
				auto input = std::make_shared<node::Input>(
				    input_tensor->get_shape(),
				    input_tensor->get_type()->type());
				m_inputs.emplace(input, input_tensor);
				m_output = input;
			}

			LazyExpr(Protected)
			{
			}

			template <typename T>
			void bind_input(const T& expr)
			{
				if constexpr (std::is_same_v<T, std::unique_ptr<LazyExpr>>)
				{
					for (const auto& [input_node, input_tensor] :
					     expr->get_inputs())
					{
						if (!m_inputs.count(input_node))
						{
							m_inputs.emplace(input_node, input_tensor);
						}
					}
				}
			}

			/** bind rhs nodes to lhs (this)
			 */
			template <
			    typename OperatorType, typename... Args,
			    std::enable_if_t<std::is_base_of_v<node::Node, OperatorType>>* =
			        nullptr>
			void bind(Args&&... args)
			{
				(bind_input(std::forward<Args>(args)), ...);

				m_output = std::make_shared<OperatorType>(
				    m_output, return_node(std::forward<Args>(args))...);
			}

			const std::unordered_map<
			    std::shared_ptr<node::Input>, std::shared_ptr<Tensor>>&
			get_inputs() const
			{
				return m_inputs;
			}

			const std::shared_ptr<node::Node> get_output() const
			{
				return m_output;
			}

			std::unique_ptr<LazyExpr> copy()
			{
				auto result = std::make_unique<LazyExpr>(Protected{});
				result->m_inputs = m_inputs;
				result->m_output = m_output;
				return result;
			}

		private:
			template <typename T>
			auto return_node(const T& expr)
			{
				if constexpr (std::is_same_v<T, std::unique_ptr<LazyExpr>>)
				{
					return expr->get_output();
				}
				else
				{
					return expr;
				}
			}
		};

		/**
		 *
		 *	auto A = std::make_shared<Array>(mat);
		 *	auto B = std::make_shared<Array>(mat);
		 *
		 *	auto C = A + B;
		 *	decltype(C) == decltype(A) == decltype(B);
		 */
		class Array
		{
		protected:
			struct Protected
			{
			};
			Array(Protected)
			{
			}

			std::unique_ptr<LazyExpr> m_lazy_expr;
			std::shared_ptr<Tensor> m_output_tensor;

		public:
			template <typename T>
			Array(const T& input) : m_output_tensor(std::make_shared<Tensor>(input))
			{
				m_lazy_expr = std::make_unique<LazyExpr>(m_output_tensor);
			}

			template <typename T, std::enable_if_t<std::is_rvalue_reference_v<T&&>>* = nullptr>
			Array(T&& input)
			    : m_output_tensor(std::make_shared<Tensor>(std::move(input)))
			{
				m_lazy_expr = std::make_unique<LazyExpr>(m_output_tensor);
			}

			template <typename T>
			static std::shared_ptr<Array> create_view(const T& input)
			{
				auto result = std::make_shared<Array>(Protected{});
				result->m_output_tensor = Tensor::create_view(input);
				result->m_lazy_expr =
				    std::make_unique<LazyExpr>(result->m_output_tensor);
				return result;
			}

			Array(std::unique_ptr<LazyExpr>&& expr)
			    : m_lazy_expr(std::move(expr))
			{
			}

			const Shape& get_shape() const
			{
				return m_lazy_expr->get_output()->get_shapes()[0];
			}

			const std::shared_ptr<NumberType>& get_type() const
			{
				return m_lazy_expr->get_output()->get_types()[0];
			}

			const std::unique_ptr<LazyExpr>& get_lazy_expression() const
			{
				return m_lazy_expr;
			}

			std::shared_ptr<Tensor> evaluate()
			{
				if (!m_output_tensor)
					force_evaluate();
				return m_output_tensor;
			}

		private:
			void force_evaluate()
			{
				const auto& inputs = m_lazy_expr->get_inputs();
				const auto& output_node = m_lazy_expr->get_output();

				std::vector<std::shared_ptr<Tensor>> input_tensors;
				std::vector<std::shared_ptr<node::Input>> input_nodes;

				for (const auto& [node, tensor] : inputs)
				{
					input_nodes.push_back(node);
					input_tensors.push_back(tensor);
				}

				// evaluate the graph
				auto graph = std::make_shared<Graph>(
				    input_nodes, std::vector{output_node});
				// build with default backend
				graph->build();

				auto result = graph->evaluate(input_tensors);
				// there should be only one output here
				require(
				    result.size() == 1,
				    "Expected only one output Tensor in Array evaluation.");

				m_output_tensor = result[0];
			}
		};

		std::shared_ptr<Array> operator+(
		    const std::shared_ptr<Array>& lhs,
		    const std::shared_ptr<Array>& rhs)
		{
			auto expr = lhs->get_lazy_expression()->copy();

			expr->bind<node::Add>(rhs->get_lazy_expression());
			return std::make_shared<Array>(std::move(expr));
		}

		std::shared_ptr<Array> matmul(
		    const std::shared_ptr<Array>& lhs,
		    const std::shared_ptr<Array>& rhs, const bool transpose_a,
		    const bool transpose_b)
		{
			auto expr = lhs->get_lazy_expression()->copy();
			expr->bind<node::MatMul>(
			    rhs->get_lazy_expression(), transpose_a, transpose_b);
			return std::make_shared<Array>(std::move(expr));
		}
	} // namespace graph
} // namespace shogun

#endif