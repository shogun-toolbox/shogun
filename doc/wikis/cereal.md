## SHOGUN `Cereal` serialization framework

#### Table of Contents

- [Motivation](#motivation)
- [For SHOGUN developers](#For SHOGUN developers)
  - [Examples] (#Examples)
- [For serialization framework developers] (#For serialization framework developers)
  - [Serialization interface] (#Serialization interface)
  - [Serialization methods in `SGObject`] (#Serialization methods in `SGObject`)
  - [Serialization methods in `Any`] (#Serialization methods in `Any`)
  - [Serialization methods in `SGVector`, `SGMatirx` and `SGReferencedData`] (#Serialization methods in `SGVector`, `SGMatirx` and `SGReferencedData`)


### Motivation

[`Cereal`](http://uscilab.github.io/cereal/) is a header-only C++11 serialization library that is fast, light-weight, and easy to extend. 

The `Cereal` serialization framework in SHOGUN uses the new tag parameter framework, which allows the easy and readable archive of `SGObject` data.

### For SHOGUN developers

- `Cereal` serialization library is required for SHOGUN compilation. If no `Cereal` is found, SHOGUN will automatically download the library to `third_party/`.

- SHOGUN now supports the serialization of data into 3 formats: binary, XML, and JSON archives. The 3 pairs of save/load methods can be called by:
```
save_binary(filename);
save_json(filename);
save_xml(filename);
load_binary(filename);
load_json(filename);
load_xml(filename);
```

- All parameters saved in tag parameter list for one `SGObject` can be saved and load by:

```
SGObject obj_save;
obj_save.save_json(filename);

SGObject obj_load;
obj_load.load_json(filename);
```

- Customized archives can be added as shown [here](http://uscilab.github.io/cereal/serialization_archives.html)

#### Examples

 `CCerealObject` class defined in `tests/unit/io/CerealObject.h` is a `SGObject`-based class used for `Cereal` serialization unit tests. 
 We also use `CCerealObject` here to show how to serialize `SGObject` in SHOGUN.
 
 In `CCerealObject`, we initialized a member `SGVector<float64_t> m_vector` and regisitered it to the parameter list in constructors:
 
 ```
 #include <shogun/base/SGObject.h>
 #include <shogun/lib/SGVector.h>
 
 using namespace shogun;
 
 class CCerealObject : public CSGObject
 {
     public:
        // Construct CCerealObject from input SGVector
        CCerealObject(SGVector<float64_t> vec) : CSGObject()
        {
            m_vector = vec;
            init_params();
        }
        
        // Default constructor
        CCerealObject() : CSGObject()
        {
            m_vector = SGVector<float64_t>(5);
            m_vector.set_const(0);
            init_params();
        }
    
        const char* get_name() const { return "CerealObject"; }
        
     protected:
        // Register m_vector to parameter list with name(tag) "test_vector"
        void init_params()
        {

            register_param("test_vector", m_vector);
        }

        SGVector<float64_t> m_vector;
 }
```
 
 `m_vector` will be archived if we call serialization methods on `CCerealObject` instance.
  
 ```
 #include "CerealObject.h"
 #include <shogun/lib/SGVector.h>
 
 using namespace shogun;
 
 // Create a CCerealObject instance with assigned SGVector values
 SGVector<float64_t> vec;
 vec.range_fill(1.0);
 CCerealObject obj_save(vec);
 
 // Serialization
 obj_save.save_json("serialization_test_json.cereal");
 
 // Create another CCerealObject instance for data loading
 CCerealObject obj_load();
 obj_load.load_json("serialization_test_json.cereal");
 
 // We can extract the loaded parameter:
 SGVector<float64_t> vec_load;
 vec_load = obj_load.get<SGVector<float64_t>>("test_vector");
 ```
 
 The JSON file `serialization_test_json.cereal` will be:
 ```
{
    "CerealObject": {                           // Class name
        "test_vector": {                        // The tag of the parameter to be saved
            "value0": 2,                        // Container type for internal use
            "value1": 12,                       // Primitive type for internal use
            "value2": {                         // Data to archive
                "ReferencedData": {             // Reference Data
                    "ref_counting": true,           
                    "refcount number": 3        
                },
                "length": 5,                    // Length of the vector
                "value1": 0,                    // values of the vector
                "value2": 1,                    
                "value3": 2,
                "value4": 3,
                "value5": 4
            }
        }
    }
}
 ```
 
 
### For serialization framework developers

The serialization framework has two components:

- serialization interfaces implemented in `SGObejct`, and

- serialization (load/save) methods implemented in `SGObject` and non-`SGObject` based data structrues.

#### Serialization interface

- The `save_binary()` method in `SGObject.h` generates an `cereal::BinaryOutputArchive` object and saves `SGObject` to binary file by calling `cereal_save()` method in `SGObject`. `load_binary()` method generates an `cereal::BinaryInputArchive` object and loads the parameters from binary file back to `SGObject` by calling `cereal_load()` method in `SGObject`. The ideas are the same for JSON and XML archives.

#### Serialization methods in `SGObject`

- `cereal_save()` method iterates through the parameter list of `SGObject` registered as `self::map`, archives the [`name value pair`](https://uscilab.github.io/cereal/assets/doxygen/classcereal_1_1NameValuePair.html), with name as `basetag.name()` and value by calling `any.cereal_save()`.

- `cereal_load()` method iterates through the parameter list and reset the parameter by calling `any.cereal_load()`

#### Serialization methods in `Any`

- Namespace `serial` and object `serial::DataType m_datatype` in `Any.h` save and convert the data type of the value of parameters in `Any` constructors into `Enum`.

```
    enum EnumContainerType
    {
        CT_UNDEFINED,
        CT_PRIMITIVE,
        CT_SGVECTOR,
        CT_SGMATRIX
    };

    enum EnumPrimitiveType
    {
        PT_UNDEFINED,
        PT_BOOL_TYPE,
        PT_CHAR_TYPE,
        PT_INT_8,
        PT_UINT_8,
        PT_INT_16,
        PT_UINT_16,
        PT_INT_32,
        PT_UINT_32,
        PT_INT_64,
        PT_UINT_64,
        PT_FLOAT_32,
        PT_FLOAT_64,
        PT_FLOAT_MAX,
        PT_COMPLEX_128,
    };
```

- `Cereal_save()` together with `cereal_save_helper()` methods cast the object `storage` to its input type and archives the value.

- `Cereal_load()` together with `cereal_load_helper()` methods read the saved value back to `storage` and reset the `policy` based on the data type.

#### Serialization methods in `SGVector`, `SGMatirx` and `SGReferencedData`

Both `SGVector` and `SGMatirx` are derived from `SGReferencedData` class.

- `SGReferencedData` archives whether `ref_counting` is on by saving `true`/`false`, and the `ref_counting` value if `m_refcount != NULL`, i.e. `ref_counting` is on.

- `SGVector` and `SGMatrix` archive `ref_counting` value by calling base class load/save methods: `cereal::base_class<SGReferencedData>(this)` ([Introduction](http://uscilab.github.io/cereal/inheritance.html)).
For `SGVector`, length and vector values are archived, while for `SGMatrix`, row number, column number, and matrix values in `T* matrix` are archived. Data of `complex128_t` type is casted to `float64_t` type before archiving.
