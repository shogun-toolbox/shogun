#include <shogun/base/init.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/Parameter.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/Map.h>


using namespace shogun;

void print_message(FILE* target, const char* str)
{
    fprintf(target, "%s", str);
}


int main(int argc, char **argv)
{
    init_shogun(&print_message, &print_message, &print_message);
    
    float64_t dummy = 0.0;
    
    CMap<SGVector<float64_t>, float64_t> gradient(128, 128);
    SGVector<float64_t> vec(3);
	vec.zero();
    gradient.add(vec, dummy);

    exit_shogun();

    return 0;
}
