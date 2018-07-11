#include <shogun/base/Parameter.h>
#include <shogun/lib/exception/NotFittedException.h>
#include <shogun/transformer/Transformer.h>

namespace shogun
{

	CTransformer::CTransformer() : CSGObject()
	{
		SG_ADD(
		    &m_fitted, "fitted", "Whether the transformer has been fitted.",
		    MS_NOT_AVAILABLE);
	}

	void CTransformer::check_fitted() const
	{
		REQUIRE_E(
		    m_fitted, NotFittedException, "Transformer has not been fitted.\n")
	}
}
