#ifndef __SG_UNIQUE_UNITTEST_H__
#define __SG_UNIQUE_UNITTEST_H__

#include <shogun/base/unique.h>

class SomeTestingClassWithUnique
{
public:
    void set(int value);
    int get();
private:
    class Self;
    shogun::Unique<Self> self;
};

#endif
