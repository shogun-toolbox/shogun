#include "Base.h"

template <typename T>
class Derived: public Base<Derived<T>> {};

