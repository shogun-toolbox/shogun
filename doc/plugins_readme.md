## Shogun's Plugin Architecture

### Table of Contents

- [Motivation](#motivation)
- [Writing a New Plugin](#writing-a-new-plugin)
- [Ingredients of Plugin Architecture](#ingredients-of-plugin-architecture)
    - [Library](#library)
    - [MetaClass](#metaclass)
    - [Manifest](#manifest)
    - [Example](#example)
- [Tests](#tests)

### Motivation

Advantages of using a plugin architecture are:
- No need to recompile entire Shogun code but only the plugin that is being developed.
- Each plugin will have its own dependencies, so base-shogun will end up having minimal dependencies.
- We can have 'frozen' implementation of base shogun that actually could be installed and never touched again. Plugins could change arbitrarily.
- Scientists could use Shogun again as the building process would be much easier.

### Writing a New Plugin

Coming soon.

### Ingredients of Plugin Architecture

#### Library

At the core of plugin architecture is `Library` class. It handles loading, calling and closing of plugins from shared object (`.so`) files by using [`dlopen()`](http://linux.die.net/man/3/dlopen). Loaded plugins are available as objects of `Library` class.

```cpp
// Load a plugin named "libplugin.so" as a Library class instance.
Library library = load_library("./libplugin.so");

// Load a plugin using Library constructor
auto dup_library = Library("./libplugin.so");

library == dup_library; // True
```

#### MetaClass

`MetaClass<T>` is a template class (with `typename T`) which provides an API to return a shared-pointer-like object (`Some<T>`) of `T`. `Any` object of `std::function<Some<T>()` is passed as an argument to `MetaClass<T>` constructor as shown below:

```cpp
// erase_type() returns an Any object
MetaClass<CSGObject> meta_class(erase_type(
    std::function<Some<CSGObject>()>(
        []() -> Some<CSGObject>
        {
            return Some<CSGObject>(new MockBaseClass);
        }
    )));
```

In context of a plugin, `MetaClass<T>` objects are stored in [`Manifest<T>`](#manifest) and are the only way for users to make objects of classes available in the plugin. See the [example](#example) below to make this more concrete.

#### Manifest

`Manifest` stores meta-data of `Library`. Each `Manifest` object has a description and a set meta-classes (`MetaClass<T>`) which are responsible for creating instances of exported classes. The set of meta-classes are passed in the `Manifest` constructor in form of `std::initializer_list<std::pair<std::string,Any>>`.

It is mandatory for a plugin to declare a function called `shogunManifest()` which returns an instance of `Manifest`. We provide three macros to make this easier:
- `BEGIN_MANIFEST`: Starts manifest declaration with its description. Always immediately followed by `EXPORT`.
- `EXPORT`: Declares class to be exported. Always use this macro between `BEGIN_MANIFEST` and `END_MANIFEST`.
- `END_MANIFEST`: Ends manifest declaration.

```cpp
// Example of a plugin with description "Mock library".
// The plugin exports two classes - MockClass and AnotherMockClass.
// MockBaseClass is the parent class for the two exported classes.
// "mock_class" and "another_mock_class" are identifiers which are used as keys
// for the std::unordered_map used by this Manifest object under the hood.
BEGIN_MANIFEST("Mock library")
EXPORT(MockClass, MockBaseClass, "mock_class")
EXPORT(AnotherMockClass, MockBaseClass, "another_mock_class")
END_MANIFEST()

// The above macros are translated to:

// BEGIN_MANIFEST("Mock library")
extern "C" Manifest shogunManifest() {
    static Manifest manifest("Mock library", {
        // EXPORT(MockClass, MockBaseClass, "mock_class")
        std::make_pair("mock_class", erase_type(
            MetaClass<MockBaseClass>(erase_type(
                std::function<Some<MockBaseClass>()>(
                    []() -> Some<MockBaseClass>
                    {
                        return Some<MockBaseClass>(new MockClass);
                    }
                ))))),
        // EXPORT(AnotherMockClass, MockBaseClass, "another_mock_class")
        std::make_pair("another_mock_class", erase_type(
            MetaClass<MockBaseClass>(erase_type(
                std::function<Some<MockBaseClass>()>(
                    []() -> Some<MockBaseClass>
                    {
                        return Some<MockBaseClass>(new AnotherMockClass);
                    }
                ))))),
    // END_MANIFEST()
    });
    return manifest;
}
```
#### Example

This example illustrates how to load and use a Shogun plugin.

```cpp
// Load a plugin named "libplugin.so" as a Library class instance.
Library library = load_library("./libplugin.so");

// Get the manifest, every plugin should have a manifest compatible with Shogun's requirements.
Manifest manifest = library.manifest();

// Get MetaClass objects of MockClass and AnotherMockClass, both of which are derived from MockBaseClass.
MetaClass<MockBaseClass> mock_class = manifest.class_by_name<MockBaseClass>("mock_class");
MetaClass<MockBaseClass> another_mock_class = manifest.class_by_name<MockBaseClass>("another_mock_class");

// MetaClass objects return shared-pointer-like objects (Some<MockBaseClass>) for classes available in libplugin.so.
Some<MockBaseClass> mock_class_obj = mock_class.instance();
Some<MockBaseClass> another_mock_class_obj = another_mock_class.instance();

// Now the objects can be used as desired.
std::string mock_class_name = mock_class_obj->get_name();
std::string another_mock_class_name = another_mock_class_obj->get_name();
mock_class_obj->mock_method();
another_mock_class_obj->mock_method;
```

### Tests

Unit-tests for this plugin architecture can found in `tests/unit/base/Plugin_unittest.cc`. The unit-tests use `tests/unit/base/MockBaseClass.h` and a dummy plugin `tests/unit/base/MockLibrary.cpp`.
