## Tag based parameter framework

### Table of Contents

- [Motivation](#motivation)
- [What are Tags?](#what-are-tags)
- [API](#api)
    - [sets() / gets()](#sets--gets)
    - [has()](#has)
    - [list_params()](#list_params)
- [Python API](#python-api)
- [For Developers](#for-developers)
    - [BaseTag](#basetag)
    - [Any](#any)
    - [register_param() / register_member()](#register_param--register_member)
    - [SWIG interface](#swig-interface)
    - [Tests](#tests)

### Motivation

With the new framework, we aimed for the following:
- A consistent API to set or get the value of a parameter in any shogun class, and allow the removal of innumerable setters and getters.
- Cleanup the Shogun base classes and modularize things for easier development.
- Work towards allowing plugins for Shogun.
- Faster SWIG compilation with less symbols being exported.

### What are Tags?

`Tag` is a Shogun class which stores name and type information of a parameter. This allows a `Tag` object to be used as an identifier for a parameter in this new framework. Eg:

```cpp
auto width_tag = shogun::Tag<float64_t>("log_width");
// width_tag can be used as an identifier for a parameter of
// type float64_t and name "log_width"
```

### API

#### sets() / gets()

Let's start with an example.

```cpp
// We instantiate a Gaussian Kernel with width = 5
float64_t width = 5;
auto gkernel = shogun::GaussianKernel(width);
// Update the width to 6
gkernel.sets(width_tag, 6);
// Get the value of width
auto w = gkernel.gets(width_tag);
```

The above code shows the usage of `sets()` and `gets()`. There are a few things to note here:
- `sets()` and `gets()` can be used in any Shogun class to modify parameters as long as the class supports the new parameter framework.
- This makes the API syntax easy as we don't have setters and getters like `set_width()` or `get_width()`. One only needs to know the name and type of the parameter.
- Can't modify or query arbitrary parameters. Use `list_params()` to view the names of available parameters in an object.
```cpp
// Tag object for an float.
auto foo_tag = shogun::Tag<float64_t>("foo");
gkernel.sets(foo_tag, 6);
auto foo = gkernel.gets(foo_tag);
// Above two lines will fail.
```

Parameters can be updated and viewed without the use of `Tag` objects also, by using `sets<T>()` and `gets<T>()`:

```cpp
// Another way to update the width without using a Tag object.
gkernel.sets<float64_t>("log_width", 7);
// Another way to get the value of width parameter without using a Tag object.
auto w = gkernel.gets<float64_t>("log_width");
```

The above method is syntactically easier but if it is required to update or query a parameter multiple times then using a `Tag` object is more efficient. Eg:

```cpp
// More efficient
auto width_tag = shogun::Tag<float64_t>("log_width");
for(int i=1; i<10000; ++i)
    gkernel.sets(width_tag, i);

// Less efficient
for(int i=1; i<10000; ++i)
    gkernel.sets<float64_t>("log_width", i);
```

#### has()

`has()` can be used on a Shogun object to check if a parameter corresponding to a particular name exists or not.

```cpp
gkernel.has("foo");         // returns false
gkernel.has("log_width");   // returns true
```

Similarly, `has<T>()` can be used to check if a type exists for a parameter with a given name.

```cpp
gkernel.has<int32_t>("foo");            // returns false
gkernel.has<int32_t>("log_width");      // returns false
gkernel.has<float32_t>("log_width");    // returns true
```

#### list_params()

Coming soon.

**For developers:** The above set of functions can be found in [`SGObject.h`](https://github.com/shogun-toolbox/shogun/blob/develop/src/shogun/base/SGObject.cpp).

We discussed about how to use the new parameter framework in C++. Now let's look at one of the high level languages that Shogun supports like Python.

### Python API

Let's look at another `GaussianKernel` example.

```python
# import shogun
import modshogun as sg
# Instantiate a Gaussian Kernel object
gkernel = sg.GaussianKernel()
# set width = 5
gkernel.sets("log_width", 5.0)
# if we want to use a Tag object
float_width_tag = TagFloat("log_width")
gkernel.sets(float_width_tag, 6.0)
# get value of width
w = gkernel.getsFloat("log_width")
w = gkernel.getsInt("log_width")    # throws exception
# or by using a Tag object
w = gkernel.gets(float_width_tag)
# to check if a parameter (corresponding to name) exists
gkernel.has("log_width")    # returns true
gkernel.has("foo")          # returns false
# to check if a parameter (corresponding to name and type) exists
gkernel.hasInt("log_width")     # returns false
gkernel.hasFloat("log_width")   # returns true
gkernel.hasFloat("foo")         # returns false
```

The semantics remain the same with minute syntactic differences. Important things to note:
- `sets()` can be used to set a parameter using its name or a `Tag<Type>` (like `TagFloat`) object.
- `gets()` can be used to query a parameter's value using a `Tag<Type>` (like `TagFloat`) object **but** `gets<Type>()` (like `getsFloat()`) is used to query the value by the parameter's name.
- `has()` can be used to check the availability of a parameter by its name. `has<Type>()` (like `hasFloat()`) is used to check the availability of a parameter by it's name and type.

This nomenclature of functions `sets`/`gets`/`has` is same in all the high level languages supported by Shogun i.e. Python, Ruby, R, Lua, Java, C# and Octave.

*A pro tip:* Use `gets<Tab>`/`has<Tab>`/`Tag<Tab>` auto-completion to list all the types available.

### For developers

All this is accomplished by having a `std::unordered_map<BaseTag, Any>` (or `std::map<BaseTag, Any>` depending on the availability of C++11) in [`SGObject.cpp`](https://github.com/shogun-toolbox/shogun/blob/develop/src/shogun/base/SGObject.cpp), where the map stores the values of parameters available in a Shogun class derived from `CSGObject`. From here on, I'll refer to the `CSGObject`'s `std::unordered_map`/`std::map` as `map`.

#### BaseTag

`BaseTag` is the parent class of `Tag`. For a particular parameter, `BaseTag` stores the name, while `Tag` (being a template class) stores the type information. Name and type information are stored separately because `Tag` (being a template class) can't be used as the key in `map`, so `BaseTag` is used as the key. `BaseTag` also stores a hash generated from the parameter's name which results in fast look-ups.

See examples of `BaseTag` in [`SGObject.h`](https://github.com/shogun-toolbox/shogun/blob/develop/src/shogun/base/SGObject.h).

#### Any

`Any` allows to store objects of arbitrary types in a type agnostic way. This makes it possible to store a variety of types in the `map`.

```cpp
int32_t integer = 5;                // integer
GaussianKernel gkernel(5);          // gaussian kernel object
auto any_integer = Any(integer);    // any object for integer
auto any_kernel = Any(gkernel);     // any object of gaussian kernel
```

`Any` class has `BaseAnyPolicy` which is an interface for a policy to store a value. The value can be any data like primitive data-types, shogun objects, etc. and the policy defines how to handle this data. It works with a provided memory region and is able to set value, clear it and return the type-name as string. There are two derived classes of `BaseAnyPolicy` class:
- `NonOwningValueAnyPolicy`: This uses external pointer in non-owning fashion (the pointer is never deleted by `Any` object) and new values are stored directly by the provided pointer.
- `PointerValueAnyPolicy`: Unlike the above policy, this policy doesn't use external pointer to store values. But uses void pointers in owning fashion (the pointer is deleted by `Any` destructor).

Now we look at an example using the two above discussed policies:
```cpp
// By default Any object uses PointerValueAnyPolicy
int32_t value = 5;
auto owning_any = Any(value);
value = 6;
auto owning_any_val = recall_type<int32_t>(owning_any);
// owning_any_val=5 is not equal to value=6

// Now if we use NonOwningValueAnyPolicy
auto non_owning_any = Any::non_owning(&value);
value = 7;
auto non_owning_any_val = recall_type<int32_t>(non_owning_any);
// non_owning_any_val=7 is equal to value=7
```

`Tag`, `BaseTag` and `Any` can be found in [`src/shogun/lib`](https://github.com/shogun-toolbox/shogun/tree/develop/src/shogun/lib).

#### register_param() / register_member()

While defining a new Shogun class, `register_param()` or `register_member()` should be used in the constructor to register parameters and member variables respectively. This would allow the parameters or class member variables' values to be queried or modified using `gets()` / `sets()`. Registering members and parameters are required to prevent Shogun users to modify / query only the registered variables and not any arbitrary variables. The new Shogun class should also inherit `CSGObject` as these two functions are `protected` in `CSGObject`. Let's look at an example to make this more concrete.

```cpp
class CMockObject : public CSGObject
{
public:
    CMockObject() : CSGObject(), m_float(), m_vector(), m_kernel()
    {
        // registering non-member variables
        int32_t int_param = 1;
        register_param("integer", int_param);

        // registering member variables
        register_member("float", &m_float);
        register_member("vector", &m_vector);
        register_member("kernel", &m_kernel);
    }

private:
    float64_t m_float;
    SGVector<float64_t> m_vector;
    CKernel* m_kernel;
};
```
`register_param()` uses `PointerValueAnyPolicy` while `register_member()` uses `NonOwningValueAnyPolicy`. `SG_ADD`, a macro in `CSGObject` also uses `register_member()`.

#### SWIG interface

We support high level languages in Shogun via [SWIG](http://www.swig.org/). For this parameter framework, we need `Tag<Type>`, `gets<Type>()` and `has<Type>()` for all the base classes in Shogun, like,
- `TagKernel`, `getsKernel()`, `hasKernel()`
- `TagFeature`, `getsFeature()`, `hasFeature()`
- `TagRealVector`, `getsRealVector()`, `hasRealVector()`
- and so on

SWIG interface file (`shogun-base.i`) is generated by using a Jinja template `shogun-base.i.jinja2`, a Python script `shogun-base.i.py` and a list of Shogun base-classes `shogun-base-list.txt`. These files can be found in [`src/interfaces/modular`](https://github.com/shogun-toolbox/shogun/tree/develop/src/interfaces/modular). The generated file instantiates template functions with all the types that are supported, by using SWIG's `%template`. The generated file looks like this,

```
%template(TagKernel) Tag<CKernel*>;
%template(sets) CSGObject::sets<CKernel*>;
%template(sets) CSGObject::sets<CKernel*, void>;
%template(gets) CSGObject::gets<CKernel*>;
%template(getsKernel) CSGObject::gets<CKernel*, void>;
%template(has) CSGObject::has<CKernel*>;
%template(hasKernel) CSGObject::has<CKernel*, void>;

# and same for other types
```

#### Tests

This framework is tested by using unit-tests and integration tests.

[`MockObject.h`](https://github.com/shogun-toolbox/shogun/blob/develop/tests/unit/base/MockObject.h) is used in [`SGObject_unittest.cc`](https://github.com/shogun-toolbox/shogun/blob/develop/tests/unit/base/SGObject_unittest.cc) to test tag-parameters. `Any_unittest.cc` tests `Any`.

Python integration test `tags_params_modular.py` also uses [`MockObject.h`](https://github.com/shogun-toolbox/shogun/blob/develop/tests/unit/base/MockObject.h). The integration tests will be soon replaced by the new meta-example tests.
