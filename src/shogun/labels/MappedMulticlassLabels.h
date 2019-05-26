/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann
 */

#ifndef MAPPED_MULTICLASS_LABELS_H
#define MAPPED_MULTICLASS_LABELS_H

#include <shogun/labels/MappedLabels.h>
#include <shogun/labels/MulticlassLabels.h>

namespace shogun
{

class CMappedMulticlassLabels : public MappedLabels<CMulticlassLabels>
{
public:
	CMappedMulticlassLabels() : MappedLabels() {} // for class_list.h
	CMappedMulticlassLabels(CLabels* l) : MappedLabels(l) {}
	virtual int32_t get_num_classes();
	virtual CLabels* shallow_subset_copy();
	virtual const char* get_name() const { return "MappedMulticlassLabels"; }
protected:
	virtual std::pair<LabelMap, LabelMap> create_mapping(const CLabels* orig) const;
};

} // namespace shogun

#endif // MAPPED_MULTICLASS_LABELS_H
