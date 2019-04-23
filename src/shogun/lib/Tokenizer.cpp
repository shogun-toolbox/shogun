#include <shogun/lib/Tokenizer.h>
#include <shogun/base/Parameter.h>

namespace shogun
{

Tokenizer::Tokenizer() : SGObject()
{
	init();
}

Tokenizer::Tokenizer(const Tokenizer& orig) : SGObject(orig)
{
	text = orig.text;
}

void Tokenizer::set_text(SGVector<char> txt)
{
	text = txt;
}

void Tokenizer::init()
{
	SG_ADD(&text, "text", "The text");
}
}
