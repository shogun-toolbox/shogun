#include <shogun/lib/Tokenizer.h>
#include <shogun/base/Parameter.h>

namespace shogun
{

CTokenizer::CTokenizer() : CSGObject()
{
	init();
}

void CTokenizer::set_text(SGVector<char> txt)
{
	text = txt;
}

void CTokenizer::init()
{
	SG_ADD(&text, "text", "The text", MS_NOT_AVAILABLE)
}
}
