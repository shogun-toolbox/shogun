/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann
 */

#ifndef MAPPED_LABELS_H
#define MAPPED_LABELS_H

#ifndef SWIG
#include <shogun/lib/config.h>

#include <shogun/lib/common.h>

#include <shogun/labels/DenseLabels.h>
#include <shogun/base/class_list.h>
#include <shogun/base/range.h>


namespace shogun
{

typedef std::map<float64_t, float64_t> LabelMap;

template <typename T,
	      std::enable_if_t<
	          std::is_base_of<
	              CDenseLabels, typename std::remove_pointer<T>::type>::value,
	          T>* = nullptr>
class MappedLabels : public T
{
public:
	MappedLabels() : T() { init(); }
	MappedLabels(CLabels* orig) : T()
	{
		init();

		auto dense = orig->as<CDenseLabels>();
		ASSERT(dense);
		m_orig = dense;
		SG_REF(m_orig);
	}

	virtual ~MappedLabels()
	{
		SG_UNREF(m_orig);
	}

	virtual SGVector<float64_t> get_labels() const
	{
		if (m_to_internal.empty())
			return m_orig->get_labels();

		auto num_labels = m_orig->get_num_labels();
		SGVector<float64_t> converted(num_labels);
		for (auto i : range(num_labels))
			converted[i] = m_to_internal.at(m_orig->get_label(i));

		return converted;
	}

	virtual float64_t get_label(index_t i) const
	{
		auto orig = m_orig->get_label(i);
		return m_to_internal.size() ? m_to_internal.at(orig) : orig;
	}

	virtual int32_t get_num_labels() const { return m_orig->get_num_labels(); }

	CDenseLabels* invert(const CLabels* labels) const
	{
		auto dense = labels->as<CDenseLabels>();
		ASSERT(dense); // internal error

		auto orig = dense->get_labels();
		auto inverted = SGVector<float64_t>(orig.size());
		std::transform(orig.begin(), orig.end(), inverted.begin(),
				[this](float64_t l) {
					return this->m_from_internal.at(l);
				});

		auto result = create_object<CDenseLabels>(m_orig->get_name());
		SG_REF(result);
		result->set_labels(inverted);
		return result;
	}
	template<class MAPPED_CLASS, std::enable_if_t<std::is_base_of<T, MAPPED_CLASS>::value, MAPPED_CLASS>* = nullptr>
	static T* wrap_if_necessary(CLabels* orig)
	{
		auto casted = dynamic_cast<T*>(orig);
		if (casted && casted->is_valid())
		{
			SG_SDEBUG("Not necessary to map %s as %s, using original.\n", orig->get_name(), demangled_type<MAPPED_CLASS>().c_str());
			return casted;

		}

		SG_SDEBUG("Mapping %s as %s.\n", orig->get_name(), demangled_type<MAPPED_CLASS>().c_str());
		MAPPED_CLASS* result = new MAPPED_CLASS(orig);

		// cast is safe since this is a mixin
		auto mapping = std::move((static_cast<MappedLabels<T>*>(result))->create_mapping(orig));

		result->m_to_internal = std::move(mapping.first);
		result->m_from_internal = std::move(mapping.second);

		SG_SDEBUG("to internal mapping:\n");
		for (const auto& it : result->m_to_internal)
			SG_SDEBUG("\t%f->%f\n", it.first, it.second)

		SG_SDEBUG("from internal mapping:\n");
		for (const auto& it : result->m_from_internal)
			SG_SDEBUG("\t%f->%f\n", it.first, it.second)

		return result;
	}

private:
	void init()
	{
		m_orig = nullptr;
		SG_ADD(&m_orig, "orig", "Original labels that are mapped via this object");
		this->watch_param("to_internal", &m_to_internal, AnyParameterProperties("Map from provided into internal representation"));
		this->watch_param("from_internal", &m_from_internal, AnyParameterProperties("Map from internal back into originally provided representation"));
	}

protected:

	virtual std::pair<LabelMap, LabelMap> create_mapping(const CLabels* orig) const = 0;

	CDenseLabels* m_orig;
	LabelMap m_to_internal;
	LabelMap m_from_internal;
};

template <class MAPPED_CLASS>
CLabels* invert_labels_if_possible(CLabels* inverter, CLabels* result)
{
	CLabels* potentially_inverted = result;
	if (auto casted = dynamic_cast<MAPPED_CLASS*>(inverter))
	{
		potentially_inverted = casted->invert(result);
		SG_UNREF(result);
	}
	return potentially_inverted;
}

} // namespace shogun
#endif // SWIG
#endif // MAPPED_LABELS_H
