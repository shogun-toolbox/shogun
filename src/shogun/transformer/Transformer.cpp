#include <shogun/lib/exception/NotFittedException.h>
#include <shogun/transformer/Transformer.h>

namespace shogun
{

	Transformer::Transformer() : SGObject()
	{
		SG_ADD(
		    &m_fitted, "is_fitted", "Whether the transformer has been fitted.",
			ParameterProperties::READONLY);
	}

	void Transformer::assert_fitted() const
	{
		require<NotFittedException>(
		    m_fitted, "Transformer has not been fitted.");
	}
}
