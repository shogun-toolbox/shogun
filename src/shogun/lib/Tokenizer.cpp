#include <shogun/lib/Tokenizer.h>
#include <shogun/base/Parameter.h>

namespace shogun
{

CTokenizer::CTokenizer() : CSGObject()
{
	init();
}

CTokenizer::CTokenizer(const CTokenizer& orig) : CSGObject(orig)
{
	text = orig.text;
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
