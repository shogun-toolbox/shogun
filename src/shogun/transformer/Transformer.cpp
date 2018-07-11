#include <shogun/base/Parameter.h>
#include <shogun/lib/exception/NotFittedException.h>
#include <shogun/transformer/Transformer.h>

namespace shogun
{

	CTransformer::CTransformer() : CSGObject()
	{
		m_fitted = false;

		SG_ADD(
		    &m_fitted, "is_fitted", "Whether the transformer has been fitted.",
		    MS_NOT_AVAILABLE);
	}

	void CTransformer::assert_fitted() const
	{
		REQUIRE_E(
		    m_fitted, NotFittedException, "Transformer has not been fitted.\n")
	}
}
