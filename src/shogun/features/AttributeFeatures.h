/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Vladislav Horbatiuk, Evan Shelhamer, Yuyu Zhang
 */
#ifndef _CATTRIBUTE_FEATURES__H__
#define _CATTRIBUTE_FEATURES__H__

#include <shogun/lib/config.h>

#include <shogun/features/Features.h>

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/** Attribute Struct */
struct T_ATTRIBUTE
{
	/// attribute name
	char* attr_name;
	/// attribute object
	std::shared_ptr<Features> attr_obj;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

/** @brief Implements attributed features, that is in the simplest case a number of
 * (attribute, value) pairs.
 *
 * For example
 *
 * x[0...].attr1 = <value(s)>
 * x[0...].attr2 = <value(s)>.
 *
 * A more complex
 * example would be nested structures x[0...].attr1[0...].subattr1 = ..
 *
 * This might be used to represent
 * (attr, value) pairs, simple structures, trees ...
 */
class AttributeFeatures : public Features
{

public:
	/** default constructor */
	AttributeFeatures();

	/** destructor */
	~AttributeFeatures() override;

	/** return the feature object matching attribute name
	 *
	 * @param attr_name attribute name
	 * @return feature object
	 */
	std::shared_ptr<Features> get_attribute(char* attr_name);

	/** return the feature object at index
	 *
	 * @param idx index of attribute
	 * @param attr_name attribute name (returned by reference)
	 * @param attr_obj attribute object (returned by reference)
	 */
	void get_attribute_by_index(int idx, const char* &attr_name, std::shared_ptr<Features>& attr_obj);

	/** set the feature object for attribute name
	 *
	 * @param attr_name attribute name
	 * @param attr_obj feature object to set
	 * @return true on success
	 */
	bool set_attribute(char* attr_name, std::shared_ptr<Features> attr_obj);

	/** delete the attribute matching attribute name
	 *
	 * @param attr_name attribute name
	 * @return true on success
	 */
	bool del_attribute(char* attr_name);

	/** get number of attributes
	 *
	 * @return number of attributes
	 */
	int32_t get_num_attributes();

	/** @return object name */
	const char* get_name() const override { return "AttributeFeatures"; }

	/** duplicate feature object
	 *
	 * abstract base method
	 *
	 * @return feature object
	 */
	std::shared_ptr<Features> duplicate() const override =0;

	/** get feature type
	 *
	 * abstract base method
	 *
	 * @return templated feature type
	 */
	EFeatureType get_feature_type() const override =0;

	/** get feature class
	 *
	 * abstract base method
	 *
	 * @return feature class like STRING, SIMPLE, SPARSE...
	 */
	EFeatureClass get_feature_class() const override =0;

	/** get number of examples/vectors
	 *
	 * abstract base method
	 *
	 * @return number of examples/vectors
	 */
	int32_t get_num_vectors() const override =0;

protected:
	/** find the index of the attribute matching attribute name
	 *
	 * @param attr_name attribute name
	 * @return index (if found), otherwise -1
	 */
	inline int32_t find_attr_index(char* attr_name);

protected:
	///list of attributes (sorted)
	std::vector<T_ATTRIBUTE> features;
};
}
#endif
