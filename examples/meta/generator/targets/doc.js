/** Target files specify rules for translation from meta-language to a target
 *  programming language. Each rule specifies the syntax of a specific language
 *  construct in the target programming language. The rules are given in 
 *  Python's template string syntax. Each keyword, prefixed with a "$", is
 *  substituted by the associated translation which is obtained by recursive
 *  applications of the translation rules. Some rules are optional.
 *
 *  This document lists all rules and the keywords available for each rule.
 */
{
    /** "Root" translation rule
     * Keywords:
     * $program: String of all translated statements and comments
     * $dependencies: Dependency string as specified by the rules in "Dependencies"
     * $programName: Name of the program
     */
    "Program": "$programName $dependencies $program",

    /** Translation rules to speficy how dependencies are translated. Only used
     * if the $dependencies keyword is used in the "Program" rule.
     */
    "Dependencies": {
        /** Flags to specify what types of dependencies to include
         * IncludeAllClasses: Include all classes ever mentioned in the example
         * IncludeInterfacedClasses: Include classes where class/static methods
         *                           have been called. This includes the class
         *                           constructor. If "IncludeAllClasses" is
         *                           true, this value has no effect.
         * IncludeEnums: Include all enum types used in the example
         */
        "IncludeAllClasses": false, 
        "IncludeInterfacedClasses": true,
        "IncludeEnums": true,

        /** Default element translation rule. Applies to both classes and enums
            if the specific rules are not given.
         * Keywords:
         * $typeName: The class/enum name of the dependency
         * $includePath: path to header file where this dependency is defined
         *               e.g. shogun/clustering/KMeansMiniBatch.h
         */
        "DependencyListElement": "$includePath $typeName",

        /** Optional. Element translation rule for class dependencies (i.e. not enums)
         * $typeName: The class name
         * $includePath: path to header file where this dependency is defined
         *               e.g. shogun/clustering/KMeansMiniBatch.h
         */
        "DependencyListElementClass": "$typeName $includePath",

        /** Optional. Element translation rule for enum dependencies (i.e. not classes)
         * $typeName: The enum type name
         & $value: the enum value name
         * $includePath: path to header file where this dependency is defined
         *               e.g. shogun/clustering/KMeansMiniBatch.h
         */
        "DependencyListElementEnum": "$typeName $value $includePath",

        /** String to insert between each translated dependency element
         */
        "DependencyListSeparator": "\n",
    },

    /** Translation rule for statements. 
     * Keywords:
     * $statement: the translated statement
     */
    "Statement": "$statement;\n",

    /** Translation rule for comments. 
     * Keywords:
     * $comment: the comment string
     */
    "Comment": "//$comment\n",

    /** Translation rule for variable initialisation (either by constructing 
     * objects or by assigning an expression to the variable.
     */
    "Init": {
        /** Keywords:
         * $name: The name of the variable
         * $typeName: The name of the variable's type
         * $arguments: Arguments passed to the class constructor
         */
        "Construct": "auto $name = some<C$typeName>($arguments)",

        /** Keywords:
         * $name: The name of the variable
         * $typeName: The name of the variable's type
         * $expr: translated expression to assign
         */
        "Copy": "auto $name = $expr",

        /** SGVector and SGMatrix construction rule. Allows to use native types
         *  in the target languages.
         * $name: The name of the variable
         * $typeName: The name of the variable's type (e.g. RealMatrix)
         * $arguments: Arguments passed to the class constructor
         */
        "BoolVector": "$name = np.zeros( ($arguments), dtype='bool')",
        "CharVector": "$name = np.zeros( ($arguments), dtype='|S1')",
        "ByteVector": "$name = np.zeros( ($arguments), dtype='uint8')",
        "WordVector": "$name = np.zeros( ($arguments), dtype='uint16')",
        "IntVector": "$name = np.zeros( ($arguments), dtype='int32')",
        "LongIntVector": "$name = np.zeros( ($arguments), dtype='int64')",
        "ULongIntVector": "$name = np.zeros( ($arguments), dtype='uint64')",
        "ShortRealVector": "$name = np.zeros( ($arguments), dtype='float32')",
        "RealVector": "$name = np.zeros( ($arguments), dtype='float64')",
        "ComplexVector": "$name = np.zeros( ($arguments), dtype='complex128)'",
        "BoolMatrix": "$name = np.zeros( ($arguments), dtype='bool')",
        "CharMatrix": "$name = np.zeros( ($arguments), dtype='|S1')",
        "ByteMatrix": "$name = np.zeros( ($arguments), dtype='uint8')",
        "WordMatrix": "$name = np.zeros( ($arguments), dtype='uint16')",
        "IntMatrix": "$name = np.zeros( ($arguments), dtype='int32')",
        "LongIntMatrix": "$name = np.zeros( ($arguments), dtype='int64')",
        "ULongIntMatrix": "$name = np.zeros( ($arguments), dtype='uint64')",
        "ShortRealMatrix": "$name = np.zeros( ($arguments), dtype='float32')",
        "RealMatrix": "$name = np.zeros( ($arguments), dtype='float64')",
        "ComplexMatrix": "$name = np.zeros( ($arguments), dtype='complex128')"
    },

    /** Translation rule for assignment
     * Keywords:
     * $lhs: a variable name or a vector/matrix element access translation as 
     *       specified by the rules in "ElementAccess"
     * $expr: translated expression
     */
    "Assign": "$lhs = $expr",

    /** Translation rules for types
     * All rules here have the same keyword:
     * $typeName: name of the type
     */
    "Type": {
        // Default rule
        "Default": "$typeName",        

        // Basic type rules
        "bool": "bool",
        "string": "char*",
        "int": "int32_t",
        "float": "float32_t",
        "real": "float64_t",

        // Custom type maps are specified like so:
        "RealFeatures": "DenseFeatures<float64_t>",
        "RealSubsetFeatures": "DenseSubsetFeatures<float64_t>",
        "StringCharFeatures": "CStringFeatures<char>",

        // Here the SGVector and SGMatrix types are mapped to their correct
        // types in the target language
        "BoolVector": "SGVector<bool>",
        "CharVector": "SGVector<char>",
        "ByteVector": "SGVector<uint8_t>",
        "WordVector": "SGVector<uint16_t>",
        "ShortVector": "SGVector<int16_t>",
        "IntVector": "SGVector<int32_t>",
        "LongIntVector": "SGVector<int64_t>",
        "ULongIntVector": "SGVector<uint64_t>",
        "ShortRealVector": "SGVector<float32_t>",
        "RealVector": "SGVector<float64_t>",
        "LongRealVector": "SGVector<floatmax_t>",
        "ComplexVector": "SGVector<complex128_t>",
        "BoolMatrix": "SGMatrix<bool>",
        "CharMatrix": "SGMatrix<char>",
        "ByteMatrix": "SGMatrix<uint8_t>",
        "WordMatrix": "SGMatrix<uint16_t>",
        "ShortMatrix": "SGMatrix<int16_t>",
        "IntMatrix": "SGMatrix<int32_t>",
        "LongIntMatrix": "SGMatrix<int64_t>",
        "ULongIntMatrix": "SGMatrix<uint64_t>",
        "ShortRealMatrix": "SGMatrix<float32_t>",
        "RealMatrix": "SGMatrix<float64_t>",
        "LongRealMatrix": "SGMatrix<floatmax_t>",
        "ComplexMatrix": "SGMatrix<complex128_t>",
        "RealDistance": "RealDistance<float64_t>",
        "RealDenseDistance": "CDenseDistance<float64_t>"
    },

    /** Translation rules for expressions
     */
    "Expr": {
        // Keywords: $literal
        "StringLiteral": "\"$literal\"",
        
        "CharLiteral": "'$literal'",
        
        "BoolLiteral": {
            "True": "true",
            "False": "false"
        },

        // Keywords: $number
        "IntLiteral": "$number",

        /** 64bit float. Keywords:
         * $number
         */
        "RealLiteral": "$number",

        /** 32bit float. Keywords:
         * $number
         */
        "FloatLiteral": "${number}f",

        /** Keywords:
         * $object: name of the object
         * $method: name of the method
         * $arguments: argument list passed to the method
         */
        "MethodCall": "$object->$method($arguments)",

        /** Keywords:
         * $typeName: name of class to call the static method on
         * $method: name of the static method
         * $arguments: argument list passed to the method
         */
        "StaticCall": "C$typeName::$method($arguments)",

        // Keywords: $identifier
        "Identifier": "$identifier",

        /** Keywords:
         * $typeName: name of enum type
         * $value: name of enum value
         */
        "Enum":"$value"
    },

    // Translation rules for access and assignment to SGVector/SGMatrix elements
    "Element": {
        "Access": {
            /** Keywords:
             * $identifier: name of vector/matrix object.
             * $indices: single index or pair of indices
             */
            "Vector": "$identifier.get($indices)",
            "Matrix": "$identifier.get($indices)",

            // Rules for specific vector/matrix types can also be specified
            "BoolVector": "$identifier.bool_getter($indices)"
        },
        "Assign": {
            /** Keywords:
             * $identifier: name of vector/matrix object.
             * $indices: single index or pair of indices
             * $expr: expression to assign to matrix/vector element
             */
            "Vector": "$identifier.put($indices, $expr)",
            "Matrix": "$identifier.put($indices, $expr)",

            // Rules for specific vector/matrix types can also be specified
            "BoolVector": "$identifier.bool_setter($indices, $expr)"
        },

        /** Is the target language zero-indexed?
         * Note that the meta-language is zero-indexed. If this value is set to
         * false, all indices are therefore incremented by one in the translations.
         */
        "ZeroIndexed": true
    },

    // Keywords: $expr
    "Print": "SG_SPRINT($expr)",

    "OutputDirectoryName": "cpp",
    "FileExtension": ".cpp"
}
