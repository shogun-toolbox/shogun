/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Evan Shelhamer, Yuyu Zhang
 */

#ifndef CONVERTER_H_
#define CONVERTER_H_

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/features/Features.h>
#include <shogun/lib/common.h>
#include <shogun/transformer/Transformer.h>

namespace shogun
{

/** @brief class Converter used to convert data
 *
 */
class Converter : public Transformer
{
public:
	/** constructor */
	Converter() : Transformer(){};

	/** destructor */
	virtual ~Converter() {};

	/** get name */
	virtual const char* get_name() const { return "Converter"; }

	virtual std::shared_ptr<Features> transform(std::shared_ptr<Features> features, bool inplace = true) = 0;
};
}
#endif /* CONVERTER_H_ */

