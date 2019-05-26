/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann
 */

#ifndef MAPPED_BINARY_LABELS_H
#define MAPPED_BINARY_LABELS_H

#include <shogun/labels/MappedLabels.h>
#include <shogun/labels/BinaryLabels.h>

namespace shogun
{

class CMappedBinaryLabels : public MappedLabels<CBinaryLabels>
{
public:
	CMappedBinaryLabels() {} // for class_list.h
	CMappedBinaryLabels(CLabels* l) : MappedLabels(l) {}
	virtual const char* get_name() const { return "MappedBinaryLabels"; }
protected:
	virtual std::pair<LabelMap, LabelMap> create_mapping(const CLabels* orig) const;
};

} // namespace shogun

#endif // MAPPED_BINARY_LABELS_H
