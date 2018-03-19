#ifndef __ARROW_FACTORY_H__
#define __ARROW_FACTORY_H__

#include <memory>
#include <arrow/table.h>

#include <shogun/features/Features.h>

namespace shogun
{
	std::shared_ptr<Features> features(const std::shared_ptr<arrow::Table>& table);
}


#endif /* __ARROW_FACTORY_H__ */
