#include <shogun/base/DynArray.h>
#include <shogun/base/SGObject.h>
#include <shogun/base/mixins/ParameterHandler.h>
#include <shogun/base/range.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/config.h>
#include <shogun/lib/memory.h>

#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/lib/observers/ParameterObserver.h>

#include <rxcpp/operators/rx-filter.hpp>
#include <rxcpp/rx-lite.hpp>
#include <rxcpp/rx-subscription.hpp>

#include <algorithm>
#include <memory>
#include <shogun/evaluation/EvaluationResult.h>
#include <unordered_map>
#include <utility>

#include <shogun/distance/Distance.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/Features.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/labels/Labels.h>
#include <shogun/machine/Machine.h>
#include <shogun/multiclass/MulticlassStrategy.h>
#include <shogun/multiclass/ecoc/ECOCDecoder.h>
#include <shogun/multiclass/ecoc/ECOCEncoder.h>

using namespace shogun;

namespace shogun
{
	typedef std::unordered_map<std::string, std::string> ObsParamsList;
	typedef std::map<BaseTag, AnyParameter> ParametersMap;

	template <typename Derived>
	class ParameterHandler<Derived>::Self
	{
	public:
		void create(const BaseTag& tag, const AnyParameter& parameter)
		{
			if (has(tag))
			{
				SG_SERROR("Can not register %s twice", tag.name().c_str())
			}
			map[tag] = parameter;
		}

		void update(const BaseTag& tag, const Any& value)
		{
			if (!has(tag))
			{
				SG_SERROR(
				    "Can not update unregistered parameter %s",
				    tag.name().c_str())
			}
			map.at(tag).set_value(value);
		}

		AnyParameter get(const BaseTag& tag) const
		{
			if (!has(tag))
				return AnyParameter();
			return map.at(tag);
		}

		bool has(const BaseTag& tag) const
		{
			return map.find(tag) != map.end();
		}

		ParametersMap filter(ParameterProperties pprop) const
		{
			ParametersMap result;
			std::copy_if(
			    map.cbegin(), map.cend(), std::inserter(result, result.end()),
			    [&pprop](const std::pair<BaseTag, AnyParameter>& each) {
				    return each.second.get_properties().has_property(pprop);
			    });
			return result;
		}

		ParametersMap map;
	}; // namespace shogun
} // namespace shogun

template <typename Derived>
ParameterHandler<Derived>::ParameterHandler(
    const ParameterHandler<Derived>& orig)
    : ParameterHandler()
{
}

template <typename Derived>
ParameterHandler<Derived>::ParameterHandler()
    : self(), house_keeper((HouseKeeper<Derived>&)(*(Derived*)this)), // FIXME
      io(house_keeper.io)
{
}

template <typename Derived>
ParameterHandler<Derived>::~ParameterHandler()
{
}

template <typename Derived>
Derived* ParameterHandler<Derived>::clone() const
{
	SG_DEBUG("Starting to clone %s at %p.\n", house_keeper.get_name(), this);
	SG_DEBUG(
	    "Constructing an empty instance of %s.\n", house_keeper.get_name());
	Derived* clone = house_keeper.create_empty();
	SG_DEBUG(
	    "Empty instance of %s created at %p.\n", house_keeper.get_name(),
	    clone);

	REQUIRE(
	    clone,
	    "Could not create empty instance of %s. The reason for "
	    "this usually is that get_name() of the class returns something "
	    "wrong, that a class has a wrongly set generic type, or that it "
	    "lies outside the main source tree and does not have "
	    "CSGObject::create_empty() overridden.\n",
	    house_keeper.get_name());

	for (const auto& it : self->map)
	{
		const BaseTag& tag = it.first;
		const Any& own = it.second.get_value();

		if (!own.cloneable())
		{
			SG_SDEBUG(
			    "Skipping clone of %s::%s of type %s.\n",
			    this->house_keeper.get_name(), tag.name().c_str(),
			    own.type().c_str());
			continue;
		}

		SG_SDEBUG(
		    "Cloning parameter %s::%s of type %s.\n",
		    this->house_keeper.get_name(), tag.name().c_str(),
		    own.type().c_str());

		clone->get_parameter(tag).get_value().clone_from(own);
	}

	SG_DEBUG(
	    "Done cloning %s at %p, new object at %p.\n", house_keeper.get_name(),
	    this, clone);
	return clone;
}

template <typename Derived>
bool ParameterHandler<Derived>::has(const std::string& name) const
{
	return has_parameter(BaseTag(name));
}

template <typename Derived>
void ParameterHandler<Derived>::create_parameter(
    const BaseTag& _tag, const AnyParameter& parameter)
{
	self->create(_tag, parameter);
}

template <typename Derived>
void ParameterHandler<Derived>::update_parameter(
    const BaseTag& _tag, const Any& value)
{
	if (!self->map[_tag].get_properties().has_property(
	        ParameterProperties::READONLY))
		self->update(_tag, value);
	else
	{
		SG_ERROR(
		    "%s::%s is marked as read-only and cannot be modified",
		    house_keeper.get_name(), _tag.name().c_str());
	}
	self->map[_tag].get_properties().remove_property(ParameterProperties::AUTO);
}

template <typename Derived>
AnyParameter ParameterHandler<Derived>::get_parameter(const BaseTag& _tag) const
{
	const auto& parameter = self->get(_tag);
	if (parameter.get_value().empty())
	{
		SG_ERROR(
		    "There is no parameter called \"%s\" in %s\n", _tag.name().c_str(),
		    house_keeper.get_name());
	}
	return parameter;
}

template <typename Derived>
bool ParameterHandler<Derived>::has_parameter(const BaseTag& _tag) const
{
	return self->has(_tag);
}

template <typename Derived>
std::map<std::string, std::shared_ptr<const AnyParameter>>
ParameterHandler<Derived>::get_params() const
{
	std::map<std::string, std::shared_ptr<const AnyParameter>> result;
	for (auto const& each : self->map)
	{
		result.emplace(
		    each.first.name(),
		    std::make_shared<const AnyParameter>(each.second));
	}
	return result;
}

template <typename Derived>
void ParameterHandler<Derived>::init_auto_params()
{
	auto params = self->filter(ParameterProperties::AUTO);
	for (const auto& param : params)
	{
		update_parameter(
		    param.first, param.second.get_init_function()->operator()());
	}
}

template <typename Derived>
std::string ParameterHandler<Derived>::string_enum_reverse_lookup(
    const std::string& param, machine_int_t value) const
{
	auto param_enum_map = m_string_to_enum_map.at(param);
	auto enum_value = value;
	auto enum_map_it = std::find_if(
	    param_enum_map.begin(), param_enum_map.end(),
	    [&enum_value](const std::pair<std::string, machine_int_t>& p) {
		    return p.second == enum_value;
	    });
	return enum_map_it->first;
}

template <typename Derived>
CSGObject*
ParameterHandler<Derived>::get(const std::string& name, index_t index) const
{
	auto* result = sgo_details::get_by_tag(
	    (Derived*)this, name, std::move(sgo_details::GetByNameIndex(index)));
	if (!result && has(name))
	{
		SG_ERROR(
		    "Cannot get array parameter %s::%s[%d] of type %s as object.\n",
		    house_keeper.get_name(), name.c_str(), index,
		    self->map[BaseTag(name)].get_value().type().c_str());
	}
	return result;
}

template <typename Derived>
CSGObject*
ParameterHandler<Derived>::get(const std::string& name, std::nothrow_t) const
    noexcept
{
	return sgo_details::get_by_tag(
	    (Derived*)this, name, std::move(sgo_details::GetByName()));
}

template <typename Derived>
CSGObject* ParameterHandler<Derived>::get(const std::string& name) const
    noexcept(false)
{
	if (!has(name))
	{
		SG_ERROR(
		    "Parameter %s::%s does not exist.\n", house_keeper.get_name(),
		    name.c_str())
	}
	if (auto* result = get(name, std::nothrow))
	{
		return result;
	}
	SG_ERROR(
	    "Cannot get parameter %s::%s of type %s as object.\n",
	    house_keeper.get_name(), name.c_str(),
	    self->map[BaseTag(name)].get_value().type().c_str());
	return nullptr;
}

class ToStringVisitor : public AnyVisitor
{
public:
	ToStringVisitor(std::stringstream* ss) : AnyVisitor(), m_stream(ss)
	{
	}

	virtual void on(bool* v)
	{
		stream() << (*v ? "true" : "false");
	}
	virtual void on(int32_t* v)
	{
		stream() << *v;
	}
	virtual void on(int64_t* v)
	{
		stream() << *v;
	}
	virtual void on(float* v)
	{
		stream() << *v;
	}
	virtual void on(double* v)
	{
		stream() << *v;
	}
	virtual void on(long double* v)
	{
		stream() << *v;
	}

	virtual void on(CSGObject** v)
	{
		if (*v)
		{
			stream() << (*v)->get_name() << "(...)";
		}
		else
		{
			stream() << "null";
		}
	}

	virtual void on(SGVector<int>* v)
	{
		to_string(v);
	}
	virtual void on(SGVector<float>* v)
	{
		to_string(v);
	}
	virtual void on(SGVector<double>* v)
	{
		to_string(v);
	}
	virtual void on(SGMatrix<int>* mat)
	{
		to_string(mat);
	}
	virtual void on(SGMatrix<float>* mat)
	{
		to_string(mat);
	}
	virtual void on(SGMatrix<double>* mat)
	{
		to_string(mat);
	}

private:
	std::stringstream& stream()
	{
		return *m_stream;
	}

	template <class T>
	void to_string(SGMatrix<T>* m)
	{
		if (m)
		{
			stream() << "Matrix<" << demangled_type<T>() << ">(" << m->num_rows
			         << "," << m->num_cols << "): [";
			for (auto col : range(m->num_cols))
			{
				stream() << "[";
				for (auto row : range(m->num_rows))
				{
					stream() << (*m)(row, col);
					if (row < m->num_rows - 1)
						stream() << ",";
				}
				stream() << "]";
				if (col < m->num_cols)
					stream() << ",";
			}
			stream() << "]";
		}
	}

	template <class T>
	void to_string(SGVector<T>* v)
	{
		if (v)
		{
			stream() << "Vector<" << demangled_type<T>() << ">(" << v->vlen
			         << "): [";
			for (auto i : range(v->vlen))
			{
				stream() << (*v)[i];
				if (i < v->vlen - 1)
					stream() << ",";
			}
			stream() << "]";
		}
	}

private:
	std::stringstream* m_stream;
};

template <typename Derived>
std::string ParameterHandler<Derived>::to_string() const
{
	std::stringstream ss;
	std::unique_ptr<AnyVisitor> visitor(new ToStringVisitor(&ss));
	ss << house_keeper.get_name();
	ss << "(";
	for (auto it = self->map.begin(); it != self->map.end(); ++it)
	{
		ss << it->first.name() << "=";
		auto value = it->second.get_value();
		if (m_string_to_enum_map.find(it->first.name()) !=
		    m_string_to_enum_map.end())
		{
			ss << string_enum_reverse_lookup(
			    it->first.name(), any_cast<machine_int_t>(value));
		}
		else if (value.visitable())
		{
			value.visit(visitor.get());
		}
		else
		{
			ss << "{function}";
		}
		if (std::next(it) != (self->map.end()))
		{
			ss << ",";
		}
	}
	ss << ")";
	return ss.str();
}

template <typename Derived>
bool ParameterHandler<Derived>::equals(const Derived* other) const
{
	if (other == this)
		return true;

	if (other == nullptr)
	{
		SG_DEBUG("No object to compare to provided.\n");
		return false;
	}

	/* Assumption: can use SGObject::get_name to distinguish types */
	if (strcmp(this->house_keeper.get_name(), other->get_name()))
	{
		SG_DEBUG(
		    "Own type %s differs from provided %s.\n", house_keeper.get_name(),
		    other->get_name());
		return false;
	}

	/* Assumption: objects of same type have same set of tags. */
	for (const auto& it : self->map)
	{
		const BaseTag& tag = it.first;
		const Any& own = it.second.get_value();

		if (!own.visitable())
		{
			SG_SDEBUG(
			    "Skipping comparison of %s::%s of type %s as it is "
			    "non-visitable.\n",
			    this->house_keeper.get_name(), tag.name().c_str(),
			    own.type().c_str());
			continue;
		}

		const Any& given = other->get_parameter(tag).get_value();

		SG_SDEBUG(
		    "Comparing parameter %s::%s of type %s.\n",
		    this->house_keeper.get_name(), tag.name().c_str(),
		    own.type().c_str());
		if (own != given)
		{
			if (house_keeper.io->get_loglevel() <= MSG_DEBUG)
			{
				std::stringstream ss;
				std::unique_ptr<AnyVisitor> visitor(new ToStringVisitor(&ss));

				ss << "Own parameter " << this->house_keeper.get_name()
				   << "::" << tag.name() << "=";
				own.visit(visitor.get());

				ss << " different from provided " << other->get_name()
				   << "::" << tag.name() << "=";
				given.visit(visitor.get());

				SG_SDEBUG("%s\n", ss.str().c_str());
			}

			return false;
		}
	}

	SG_SDEBUG("All parameters of %s equal.\n", this->house_keeper.get_name());
	return true;
}

namespace shogun
{
	template class ParameterHandler<CSGObject>;
	template class ParameterHandler<ObservedValue>;
} // namespace shogun