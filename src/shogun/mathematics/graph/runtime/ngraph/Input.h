/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_INPUT_NGRAPH_H_
#define SHOGUN_INPUT_NGRAPH_H_

#include <shogun/mathematics/graph/nodes/Input.h>
#include <shogun/mathematics/graph/runtime/RuntimeNode.h>

#include <ngraph/op/parameter.hpp>

namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace ngraph
			{
				inline std::shared_ptr<shogun::graph::NumberType>
				get_enum_from_ngraph(::ngraph::element::Type_t type)
				{
					switch (type)
					{
					case ::ngraph::element::Type_t::f32:
						return number_type(element_type::FLOAT32);
					case ::ngraph::element::Type_t::f64:
						return number_type(element_type::FLOAT64);
					case ::ngraph::element::Type_t::boolean:
						return number_type(element_type::BOOLEAN);
					case ::ngraph::element::Type_t::i8:
						return number_type(element_type::INT8);
					case ::ngraph::element::Type_t::i16:
						return number_type(element_type::INT16);
					case ::ngraph::element::Type_t::i32:
						return number_type(element_type::INT32);
					case ::ngraph::element::Type_t::i64:
						return number_type(element_type::INT64);
					case ::ngraph::element::Type_t::u8:
						return number_type(element_type::UINT8);
					case ::ngraph::element::Type_t::u16:
						return number_type(element_type::UINT16);
					case ::ngraph::element::Type_t::u32:
						return number_type(element_type::UINT32);
					case ::ngraph::element::Type_t::u64:
						return number_type(element_type::UINT64);
					case ::ngraph::element::Type_t::undefined:
					case ::ngraph::element::Type_t::dynamic:
					case ::ngraph::element::Type_t::bf16:
					case ::ngraph::element::Type_t::f16:
					case ::ngraph::element::Type_t::u1:
						error("Cannot handle type {)", type);
					}
				}

				inline ::ngraph::element::Type_t get_ngraph_type_from_enum(
				    const std::shared_ptr<shogun::graph::NumberType>& type)
				{
					switch (type->type())
					{
					case element_type::FLOAT32:
						return ::ngraph::element::Type_t::f32;
					case element_type::FLOAT64:
						return ::ngraph::element::Type_t::f64;
					case element_type::BOOLEAN:
						return ::ngraph::element::Type_t::boolean;
					case element_type::INT8:
						return ::ngraph::element::Type_t::i8;
					case element_type::INT16:
						return ::ngraph::element::Type_t::i16;
					case element_type::INT32:
						return ::ngraph::element::Type_t::i32;
					case element_type::INT64:
						return ::ngraph::element::Type_t::i64;
					case element_type::UINT8:
						return ::ngraph::element::Type_t::u8;
					case element_type::UINT16:
						return ::ngraph::element::Type_t::u16;
					case element_type::UINT32:
						return ::ngraph::element::Type_t::u32;
					case element_type::UINT64:
						return ::ngraph::element::Type_t::u64;
					}
				}

				inline ::ngraph::Shape to_ngraph_shape(const Shape& shape)
				{
					return ::ngraph::Shape(shape.begin(), shape.end());
				}

				inline ::ngraph::PartialShape
				to_ngraph_partial_shape(const Shape& shape)
				{
					std::vector<::ngraph::Dimension> result;
					for (const auto& el : shape)
					{
						if (el == Shape::Dynamic)
							result.push_back(::ngraph::Dimension::dynamic());
						else
							result.emplace_back(el);
					}
					return ::ngraph::PartialShape(result);
				}

				inline Shape from_ngraph_shape(const ::ngraph::Shape& shape)
				{
					std::vector<Shape::shape_type> out;
					for (const auto& el : shape)
						out.push_back(static_cast<Shape::shape_type>(el));

					return Shape(out);
				}

				IGNORE_IN_CLASSLIST class InputNGraph
				    : public RuntimeNodeTemplate<node::Input, ::ngraph::Node>
				{
				public:
					InputNGraph() : RuntimeNodeTemplate()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "Input";
					}

					[[nodiscard]] std::shared_ptr<::ngraph::Node>
					build_input(const std::shared_ptr<node::Node>& node) const {
						const auto& shape = node->get_shapes()[0];
						const auto& type = node->get_types()[0];

						// dynamic shape
						if (std::find(
						        shape.begin(), shape.end(), Shape::Dynamic) !=
						    shape.end())
						{
							return std::make_shared<::ngraph::op::Parameter>(
							    get_ngraph_type_from_enum(type),
							    to_ngraph_partial_shape(shape));
						}
						else
						{
							return std::make_shared<::ngraph::op::Parameter>(
							    get_ngraph_type_from_enum(type),
							    to_ngraph_shape(shape));
						}
					}

					    [[nodiscard]] std::
					        shared_ptr<::ngraph::Node> build_implementation(
					            const std::shared_ptr<node::Node>& node)
					            const final
					{
						error("Input nodes use Input::build_input(node) "
						      "instead.");
						return nullptr;
					}
				};
			} // namespace ngraph
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif
