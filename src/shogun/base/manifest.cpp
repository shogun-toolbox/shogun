/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Sanuj Sharma
 */

#include <shogun/base/manifest.h>
#include <shogun/io/SGIO.h>

#include <unordered_map>

namespace shogun
{

    class Manifest::Self
    {
        public:
            std::string description;
            std::unordered_map<std::string, Any> classes;
    };

    Manifest::Manifest(const std::string& description,
            const std::initializer_list<std::pair<std::string,Any>> classes) :
        self()
    {
        self->description = description;
        for (const auto& m : classes)
            add_class(m.first, m.second);
    }

    Manifest::Manifest(const Manifest& other) :
        self()
    {
        self->description = other.self->description;
        self->classes = other.self->classes;
    }

    Manifest& Manifest::operator=(const Manifest& other)
    {
        self->description = other.self->description;
        self->classes = other.self->classes;
        return *this;
    }

    bool operator==(const Manifest& first, const Manifest& second)
    {
        return (first.self->description == second.self->description)
            and (first.self->classes == second.self->classes);
    }

    bool operator!=(const Manifest& first, const Manifest& second)
    {
        return !(first == second);
    }

    Manifest::~Manifest()
    {
    }

    std::string Manifest::description() const
    {
        return self->description;
    }

    void Manifest::add_class(const std::string& name, Any clazz)
    {
        self->classes[name] = clazz;
    }

    Any Manifest::find_class(const std::string& name) const
    {
        if (!self->classes.count(name))
            SG_SERROR("MetaClass corresponding to the name '%s' couldn't be found.\n", name.c_str());
        return self->classes.at(name);
    }

}
