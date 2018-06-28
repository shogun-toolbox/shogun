import json
import sys
from string import Template
import os.path
import argparse
import re
from copy import copy
from functools import reduce
# import set in a python 2->3 safe way
try:
    set
except NameError:
    from sets import Set as set
import warnings


def find(key, dictionary):
    """ Recursively search a dictionary for a key """
    for k, v in dictionary.items():
        if k == key:
            yield v
        elif isinstance(v, dict):
            for result in find(key, v):
                yield result
        elif isinstance(v, list):
            for d in v:
                for result in find(key, d):
                    yield result


def getDependencies(program):
    """ Traverses the program AST and extracts all dependencies """
    allClasses = set()
    interfaceClasses = set()
    enums = set()
    globalFunctions = set()

    # All classes used
    for objectType in find("ObjectType", program):
        allClasses.add(objectType)

    for shogunSGType in find("ShogunSGType", program):
        allClasses.add(shogunSGType)

    # All classes where the constructor is called
    for initialisation in find("Init", program):
        # Constructor interface is used if an argument list is passed
        if list(initialisation[2].keys())[0] == "ArgumentList":
            typeDict = initialisation[0]
            objectKey = list(typeDict.keys())[0]
            interfaceClasses.add(typeDict[objectKey])

    # All classes on which a static method is called
    for staticCall in find("StaticCall", program):
        typeDict = staticCall[0]
        objectKey = list(typeDict.keys())[0]
        interfaceClasses.add(typeDict[objectKey])

    # All global function calls
    for globalCall in find("GlobalCall", program):
        identifier = list(globalCall[0].values())[0]
        globalFunctions.add(identifier)

    # All enums used
    for enum in find("Enum", program):
        enumType = enum[0]["Identifier"]
        enumValue = enum[1]["Identifier"]
        enums.add((enumType, enumValue))

    return allClasses, interfaceClasses, enums, globalFunctions


def getBasicTypesToStore():
    """ Returns all basic types which will be serialized """
    return ("real", "float", "int")

def getSGTypesToStore():
    """ Returns all SG* types which will be serialized """
    return ("RealVector","RealMatrix","FloatVector","FloatMatrix")

def getSGTypeToStoreMethodName(sgType):
    """ Translates given SG* type into meta language type """
    assert sgType in getSGTypesToStore()

    if sgType=="RealVector":
        return "real_vector"
    elif sgType=="FloatVector":
        return "float_vector"
    elif sgType=="RealMatrix":
        return "real_matrix"
    elif sgType=="FloatMatrix":
        return "float_matrix"

    else:
        raise RuntimeError("Given Shogun type \"%s\" cannot be translated to meta type", sgType)


def getVarsToStore(program):
    """ Extracts all variables in program that should be stored """
    varsToStore = []

    # store only real valued matrices, vectors and scalars
    for init in find("Init", program):
        typeObject = init[0]
        nameString = init[1]["Identifier"]

        appendStore = False
        if "ShogunSGType" in init[0]:
            if init[0]["ShogunSGType"] in getSGTypesToStore():
                appendStore = True

        elif "BasicType" in init[0]:
            if init[0]["BasicType"] in getBasicTypesToStore():
                appendStore = True

        if appendStore:
            varsToStore.append((typeObject, nameString))

    return varsToStore

def findPath(key, dictionary, path=[]):
    """ Recursively search a dictionary for a key """
    for k, v in dictionary.items():
        if k == key:
            yield v, path + [k]
        if isinstance(v, dict):
            for result, p in findPath(key, v, path + [k]):
                yield result, p
        if isinstance(v, list):
            for idx, d in enumerate(v):
                for result, p in findPath(key, d, path + [k, idx]):
                    yield result, p

def validateProgram(program, filePath=None):
    """ Checks that a program is valid. Throws exceptions if not. """

    for d, path in findPath('KeywordArgument', program):
        # Check that any keyword argument appears either in a constructor call
        # or in the first global call of a init by copy statement
        try:
            initByGlobalCall = path[-8:-2] == ['Init', 2, 'Expr', 'GlobalCall', 1, 'ArgumentList']
            initByConstructor = path[-5:-2] == ['Init', 2, 'ArgumentList']
        except:
            initByGlobalCall = False
            initByConstructor = False

        if not (initByGlobalCall or initByConstructor):
            raise TranslationFailure('Keyword arguments must only appear in initialisation of variables')

    for d, path in findPath('ArgumentList', program):
        # Check that normal arguments always come before kwargs
        if not isinstance(d, list):
            raise TranslationFailure('ArgumentList should be a list')

        kwargObserved = False
        for arg in d:
            if 'KeywordArgument' in arg:
                kwargObserved = True
            else:
                if kwargObserved:
                    raise TranslationFailure('Keyword argument must come after normal arguments')

    # Warn against misspelled boolean literals
    for identifier, path in findPath('Identifier', program):
        # Retrieve the line number of the identifier
        lineNo = None
        if "Statement" in path:
            lineNoDict = program
            for p in path:
                if p == "Statement":
                    break
                lineNoDict = lineNoDict[p]
            lineNo = lineNoDict.get("__PARSER_INFO_LINE_NO")

        if identifier.lower() == 'true' or identifier.lower() == 'false':
            boolLiteral = identifier[0].upper() + identifier[1:].lower()
            warn_text = 'Detected identifier "{}". Did you mean the boolean literal "{}"?'.format(
                identifier, boolLiteral
            )
            warnings.showwarning(warn_text, UserWarning, filePath, lineNo)


class TranslationFailure(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Translator:
    def __init__(self, targetDict, tags={}):
        self.targetDict = targetDict
        self.tags = tags
        self.variableTypes = {}

    def translateProgram(self, program, programName=None, storeVars=False):
        """ Translate program AST
        Args:
            program: object like {"Program": [statementAST,
                                              statementAST,
                                              statementAST, ..]}
        """

        try:
            filePath = program["FilePath"]
        except KeyError:
            filePath = None

        if storeVars:
            varsToStore = getVarsToStore(program)

            # Shallow copy of the statement list, so we don't alter the original
            program = copy(program)
            program["Program"] = copy(program["Program"])

            self.injectVarsStoring(program["Program"],
                                   programName,
                                   varsToStore)

        targetProgram = ""
        for line in program["Program"]:
            try:
                if "Statement" in line:
                    validateProgram(line, filePath=filePath)
                    targetProgram += self.translateStatement(line["Statement"])
                elif "Comment" in line:
                    targetProgram += self.translateComment(line["Comment"])
            except:
                try:
                    print("File {}: Translation failed on line {}".format(
                        filePath, line["__PARSER_INFO_LINE_NO"])
                    )
                except:
                    print("Translation failed on line\n%s\n" % line)
                    pass
                raise

        allClasses, interfacedClasses, enums, globalFunctions = getDependencies(program)
        try:
            dependenciesString = self.dependenciesString(allClasses,
                                                         interfacedClasses,
                                                         enums,
                                                         globalFunctions)
        except Exception as e:
            print("Translation of dependencies failed!")
            raise

        programTemplate = Template(self.targetDict["Program"])
        return programTemplate.substitute(program=targetProgram,
                                          dependencies=dependenciesString,
                                          programName=programName)

    def injectVarsStoring(self, statementList, programName, varsToStore):
        """ Injects statements at the end of the program that perform variable
            storing
        """
        storage = "sg_storage"
        storageFile = "sg_storage_file"

        # TODO: handle directories
        storageFilename = {
            "Expr": {"StringLiteral": "{}.dat".format(programName)}
        }
        # 'w'
        storageFilemode = {"Expr": {"CharLiteral": 'w'}}
        storageComment = {"Comment": " Serialize output for integration testing (automatically generated)"}
        storageInit = {"Init": [{"ObjectType": "DynamicObjectArray"},
                                {"Identifier": storage},
                                {"ArgumentList": []}]}
        storageFileInit = {
            "Init": [{"ObjectType": "SerializableAsciiFile"},
                     {"Identifier": storageFile},
                     {"ArgumentList": [storageFilename, storageFilemode]}]
        }

        statementList.append({"Statement": "\n"})
        statementList.append(storageComment)
        statementList.append({"Statement": storageInit})
        statementList.append({"Statement": storageFileInit})

        for vartypeAST, varname in varsToStore:
            # avoid storing itself
            if varname in (storage, storageFile):
                continue

            varnameExpr = {"Expr": {"StringLiteral": varname}}
            varnameIdentifierExpr = {"Expr": {"Identifier": varname}}

            # extract SWIG type name, we know that vartypeAST is a dict of the form {"BasicType": "real"}
            # i.e. one key
            # python2/3 compatible key accessing
            sgType = vartypeAST[list(vartypeAST.keys())[0]]
            assert sgType in getBasicTypesToStore() or sgType in getSGTypesToStore()

            appendElementIdentifier = "append_element"
            if not ("cpp" in self.targetDict["FileExtension"]):
                suffix = "_"
                if sgType in getBasicTypesToStore():
                    suffix += sgType
                elif sgType in getSGTypesToStore():
                    suffix += getSGTypeToStoreMethodName(sgType)
                appendElementIdentifier += suffix

            methodCall = {
                "MethodCall": [{"Identifier": storage},
                               {"Identifier": appendElementIdentifier},
                               {"ArgumentList": [varnameIdentifierExpr,
                                                 varnameExpr]}]
            }
            expression = {"Expr": methodCall}
            statementList.append({"Statement": expression})

        storageSerialize = {
            "Expr": {"MethodCall": [
                {"Identifier": storage},
                {"Identifier": "save_serializable"},
                {"ArgumentList": [{"Expr": {"Identifier": storageFile}}]}
            ]}
        }
        statementList.append({"Statement": storageSerialize})

    def dependenciesString(self, allClasses, interfacedClasses, enums,
                           globalFunctions):
        """ Returns dependency import string
            e.g. for python: "from shogun import RealFeatures\n\n"
        """

        if "Dependencies" not in self.targetDict:
            # Dependency strings are optional so we just return empty string
            return ""

        dependencies = set()

        if self.targetDict["Dependencies"].get("IncludeAllClasses"):
            dependencies = dependencies.union(allClasses)
        if self.targetDict["Dependencies"].get("IncludeInterfacedClasses"):
            dependencies = dependencies.union(interfacedClasses)
        if self.targetDict["Dependencies"].get("IncludeEnums"):
            dependencies = dependencies.union(enums)
        if self.targetDict["Dependencies"].get("IncludeGlobalFunctions"):
            dependencies = dependencies.union(globalFunctions)

        dependencies = list(dependencies)

        translations = list(set(map(self.translateDependencyElement, dependencies)))
        translations.sort()

        separator = self.targetDict["Dependencies"]["DependencyListSeparator"]
        result = reduce(lambda l, r: r if l == "" else l+separator+r,
                      translations,
                      "")
        return result

    def translateDependencyElement(self, dependencyElement):
        """ Translates a dependency element
        Args:
            dependencyElement: object like "RealFeatures"
                               or ("LIBLINEAR_SOLVER_TYPE", "L2R_L2LOSS_SVC")
        """
        dependencyRules = self.targetDict["Dependencies"]
        elementTemplate = Template(dependencyRules["DependencyListElement"])

        typeName = dependencyElement
        value = ""
        includePath = ""

        # If enum
        if isinstance(dependencyElement, tuple):
            typeName = dependencyElement[0]
            value = dependencyElement[1]

            if "DependencyListElementEnum" in dependencyRules:
                elementTemplate = Template(dependencyRules["DependencyListElementEnum"])

        elif "DependencyListElementClass" in dependencyRules:
            elementTemplate = Template(dependencyRules["DependencyListElementClass"])

        if "$includePath" in elementTemplate.template:
            includePath = self.getIncludePathForClass(typeName)

        return elementTemplate.substitute(typeName=typeName,
                                          value=value,
                                          includePath=includePath)

    def getIncludePathForClass(self, type_):
        translatedType = self.translateType({"ObjectType": type_})
        templateParameterMatcher = '\<[0-9a-zA-Z_]*\>'
        variants = [
            'C' + translatedType,
            translatedType,
            'C' + re.sub(templateParameterMatcher, '', translatedType),
            re.sub(templateParameterMatcher, '', translatedType)
        ]

        candidates = []
        for variant in variants:
            if variant in self.tags:
                candidates.append(self.tags[variant])

        uniqueCandidates = [path for i, path in enumerate(candidates)
                                      if candidates.index(path) == i]

        if len(uniqueCandidates) == 1:
            return uniqueCandidates[0]

        elif len(uniqueCandidates) > 1:
            msg = "Several possible include paths for type {}.\n"\
                  "Candidate paths: {}\nChosen: {}"
            warnings.warn(msg.format(type_,
                                     uniqueCandidates,
                                     uniqueCandidates[0]))
            return uniqueCandidates[0]

        raise TranslationFailure('Failed to obtain include path for %s' %
                                 (' or '.join(variants)))

    def translateStatement(self, statement):
        """ Translate statement AST
        Args:
            statement: object like {"Init": initAST}, {"Assign": assignAST},
                       {"Expr": exprAST}, "\n"
        """
        # Newline handling
        if statement == "\n":
            return "\n"

        # python2/3 safe dictionary keys
        type = list(statement.keys())[0]

        translation = None
        if type == "Init":
            translation = self.translateInit(statement["Init"])
        elif type == "Assign":
            translation = self.translateAssign(statement["Assign"])
        elif type == "Expr":
            translation = self.translateExpr(statement["Expr"])
        elif type == "Print":
            translation = self.translatePrint(statement["Print"])

        if translation is None:
            raise TranslationFailure("Unknown statement type: " + type)

        template = Template(self.targetDict["Statement"])
        return template.substitute(statement=translation)

    def translateInit(self, init):
        """ Translate initialisation statement AST
        Args:
            init: object like [typeAST, identifierAST, exprAST],
                  [typeAST, identifierAST, argumentListAST], etc.
        """
        typeDict = init[0]
        typeString = self.translateType(typeDict)
        nameString = init[1]["Identifier"]
        initialisation = init[2]
        typeKey = list(typeDict.keys())[0]

        self.variableTypes[nameString] = typeDict

        # python2/3 safe dictionary keys
        if list(initialisation.keys())[0] == "Expr":
            template = Template(self.targetDict["Init"]["Copy"])
            result = self.translateExpr(
                initialisation["Expr"], returnKwargs=True
            )
            exprString = result
            kwargs = []
            if isinstance(result, tuple):
                exprString, kwargs = result

            kwargsString = self.translateKwargs(
                kwargs, nameString, typeString
            )
            return template.substitute(name=nameString,
                                       typeName=typeString,
                                       expr=exprString,
                                       kwargs=kwargsString)
        elif list(initialisation.keys())[0] == "ArgumentList":
            template = Template(self.targetDict["Init"]["Construct"])

            # Optional custom SGType construction
            if typeKey == "ShogunSGType"\
               and init[0][typeKey] in self.targetDict["Init"]:
                template = Template(self.targetDict["Init"][init[0][typeKey]])

            argsString, kwargs = self.translateArgumentList(
                initialisation, returnKwargs=True
            )

            normalArgs = [
                arg for arg in initialisation["ArgumentList"]
                    if not "KeywordArgument" in arg
            ]
            kwargsString = self.translateKwargs(
                kwargs, nameString, typeString,
                argsGtZero=len(normalArgs) > 0
            )
            return template.substitute(name=nameString,
                                       typeName=typeString,
                                       arguments=argsString,
                                       kwargs=kwargsString)

    def translateAssign(self, assign):
        """ Translatie assignment AST
        Args:
            assign: object like [ElementAccessAST, expr] and [identifier, expr]
        """
        firstElementKey = list(assign[0].keys())[0]
        expr = assign[1]["Expr"]

        if firstElementKey == "Identifier":
            identifier = assign[0][firstElementKey]

            assert identifier in self.variableTypes, \
                "Variable {} not initialised".format(identifier)

            template = Template(self.targetDict["Assign"])
            exprString = self.translateExpr(expr)
            return template.substitute(identifier=identifier, expr=exprString)

        elif firstElementKey == "ElementAccess":
            return self.translateElementAccess(assign[0][firstElementKey],
                                               expr)

        else:
            raise TranslationFailure("Uknown assignment structure: " +
                                     str(assign))

    def translateExpr(self, expr, returnKwargs=False):
        """ Translate expression AST
        Args:
            expr: objects like
                {"MethodCall": [identifierAST, identifierAST, argumentListAST]}
                {"BoolLiteral": "False"}
                {"StringLiteral": "train.dat"}
                {"Char": 'w'}
                {"IntLiteral": 4}
                {"Identifier": "feats_test"}
                etc.
        """
        # python2/3 safe dictionary keys
        key = list(expr.keys())[0]

        if key == "MethodCall":
            return self.translateMethodCall(expr[key])

        elif key == "StaticCall":
            return self.translateStaticCall(expr[key])

        elif key == "GlobalCall":
            return self.translateGlobalCall(expr[key], returnKwargs=returnKwargs)

        elif key == "ElementAccess":
            return self.translateElementAccess(expr[key])

        elif key == "BoolLiteral":
            return self.targetDict["Expr"]["BoolLiteral"][expr[key]]

        elif key == "StringLiteral":
            template = Template(self.targetDict["Expr"]["StringLiteral"])
            return template.substitute(literal=expr[key])

        elif key == "CharLiteral":
            template = Template(self.targetDict["Expr"]["CharLiteral"])
            return template.substitute(literal=expr[key])

        elif key == "IntLiteral":
            template = Template(self.targetDict["Expr"]["IntLiteral"])
            return template.substitute(number=expr[key])

        elif key == "RealLiteral":
            template = Template(self.targetDict["Expr"]["RealLiteral"])
            return template.substitute(number=expr[key])

        elif key == "FloatLiteral":
            template = Template(self.targetDict["Expr"]["FloatLiteral"])
            return template.substitute(number=expr[key])

        elif key == "Identifier":
            template = Template(self.targetDict["Expr"]["Identifier"])
            return template.substitute(identifier=expr[key])

        elif key == "Enum":
            template = Template(self.targetDict["Expr"]["Enum"])
            return template.substitute(typeName=expr[key][0]["Identifier"],
                                       value=expr[key][1]["Identifier"])

        raise TranslationFailure("Unknown expression type: " + key)

    def translateMethodCall(self, methodCall):
        """ Translates a method call expression
        Args:
            methodCall: object like [identifierAST, identifierAST, argumentListAST]
        """

        object = methodCall[0]["Identifier"]
        method = methodCall[1]["Identifier"]
        argsList = None
        try:
            argsList = methodCall[2]
        except IndexError:
            pass
        translatedArgsList = self.translateArgumentList(argsList)

        if object not in self.variableTypes:
            raise TranslationFailure("Variable {} not initialised".format(object))

        template = Template(self.targetDict["Expr"]["MethodCall"]["Default"])
        if method in self.targetDict["Expr"]["MethodCall"]:
            template = Template(self.targetDict["Expr"]["MethodCall"][method])

        return template.substitute(object=object,
                                   method=method,
                                   arguments=translatedArgsList)

    def translateStaticCall(self, staticCall):
        """ Translates a method call expression
        Args:
            staticCall: object like [identifierAST, identifierAST, argumentListAST]
        """
        template = Template(self.targetDict["Expr"]["StaticCall"])
        type_ = self.translateType(staticCall[0])
        method = staticCall[1]["Identifier"]
        argsList = None
        try:
            argsList = staticCall[2]
        except IndexError:
            pass
        translatedArgsList = self.translateArgumentList(argsList)

        return template.substitute(typeName=type_,
                                   method=method,
                                   arguments=translatedArgsList)

    def translateGlobalCall(self, globalCall, returnKwargs):
        """ Translates a method call expression
        Args:
            staticCall: object like [identifierAST, argumentListAST]
        """
        template = Template(self.targetDict["Expr"]["GlobalCall"])
        method = globalCall[0]["Identifier"]
        argsList = None
        try:
            argsList = globalCall[1]
        except IndexError:
            pass

        translatedArgsList, kwargs = self.translateArgumentList(
            argsList, returnKwargs=True
        )

        normalArgs = [
            arg for arg in argsList["ArgumentList"]
                if not "KeywordArgument" in arg
        ]
        kwargsString = self.translateKwargs(
            kwargs, argsGtZero=len(normalArgs) > 0
        )

        translation = template.substitute(
            typeName=type,method=method, arguments=translatedArgsList,
            kwargs=kwargsString
        )

        if returnKwargs:
            return translation, kwargs

        return translation

    def translatePrint(self, printStmt):
        template = Template(self.targetDict["Print"])
        return template.substitute(expr=self.translateExpr(printStmt["Expr"]))

    def translateComment(self, commentString):
        template = Template(self.targetDict["Comment"])
        return template.substitute(comment=commentString)

    def translateType(self, type):
        """ Translate type AST
        Args:
            type: object like {"ObjectType": "IntMatrix"},
                              {"BasicType": "float"}, etc.
        """
        template = ""
        # python2/3 safe dictionary keys
        typeKey = list(type.keys())[0]

        if type[typeKey] in self.targetDict["Type"]:
            template = Template(self.targetDict["Type"][type[typeKey]])
        else:
            template = Template(self.targetDict["Type"]["Default"])

        return template.substitute(typeName=type[typeKey])

    def translateArgumentList(self, argumentList, returnKwargs=False):
        """ Translate argument list AST
        Args:
            argumentList: object like None, {"Expr": exprAST},
                          [{"Expr": exprAST}, {"Expr": exprAST}],
                          [{"KeywordArgument":[identifier, expr]], etc.
        """
        kwargs = []
        translation = ""
        initialArg = True
        for el in argumentList["ArgumentList"]:
            if "Expr" in el:
                if not initialArg:
                    translation += ", "
                translation += self.translateExpr(el["Expr"])
                initialArg = False
            elif "KeywordArgument" in el:
                kwargs.append(el)

        if returnKwargs:
            return translation, kwargs

        return translation

    def translateKwargs(self, kwargs, name="", typeName="", argsGtZero=False):
        if len(kwargs) == 0:
            return ""

        kwarglist = Template(self.targetDict["Init"]["KeywordArguments"]["List"])
        elem = Template(self.targetDict["Init"]["KeywordArguments"]["Element"])
        sep = self.targetDict["Init"]["KeywordArguments"]["Separator"]
        initialSep = self.targetDict["Init"]["KeywordArguments"]["InitialSeperatorWhenArgs>0"]

        elements = []
        for kwarg in kwargs:
            elements.append(
                elem.substitute(
                    name=name,
                    typeName=typeName,
                    keyword=self.translateExpr(kwarg["KeywordArgument"][0]),
                    expr=self.translateExpr(kwarg["KeywordArgument"][1]["Expr"])
                )
            )

        separatedElements = sep.join(elements)
        if initialSep and argsGtZero and len(elements) > 0:
            separatedElements = sep + separatedElements

        return kwarglist.substitute(elements=separatedElements)


    def translateElementAccess(self, elementAccess, expr=None):
        """ Translate element access AST
        Args:
            elementAccess: object like [identifierAST, argumentListAST]
            expr: exprAST - if given this method will use the element
                  assignment rule.
        """
        identifier = elementAccess[0]["Identifier"]
        indexList = elementAccess[1]["IndexList"]

        assert identifier in self.variableTypes,\
            "Variable {} not initialised".format(identifier)
        assert list(self.variableTypes[identifier].keys())[0] == "ShogunSGType",\
            "Variable {} is not a vector or matrix type". format(identifier)

        type = self.variableTypes[identifier]["ShogunSGType"]

        exprString = None
        targetDict = self.targetDict["Element"]["Access"]
        if expr is not None:
            targetDict = self.targetDict["Element"]["Assign"]
            exprString = self.translateExpr(expr)

        indexOffsetRequired=True
        if type in targetDict:
            template = Template(targetDict[type])
            indexOffsetRequired=False
        elif len(indexList) == 1:
            template = Template(targetDict["Vector"])
        elif len(indexList) == 2:
            template = Template(targetDict["Matrix"])
        else:
            raise TranslationFailure("Element access takes either 1 index "
                                     "(vector) or 2 indices (matrix). Given "
                                     " " + str(len(indexList)) + " indices")

        # only potentially offset index lists if native Vector or Matrix assignment is used
        # otherwise, it is not possible to call native shogun vector/matrix methods
        # (which are zero indexed)
        indexListTranslation = self.translateIndexList(indexList, indexOffsetRequired)

        return template.substitute(identifier=identifier,
                                   indices=indexListTranslation,
                                   expr=exprString)

    def translateIndexList(self, indexList, indexOffsetRequired):
        """ Translate index list AST
        Args:
            indexList: object like [IntLiteralAST, IntLiteralAST, ..]
        """
        addOne = not self.targetDict["Element"]["ZeroIndexed"] and indexOffsetRequired
        translation = ""
        for idx, intLiteral in enumerate(indexList):
            index = int(intLiteral["IntLiteral"])

            if addOne:
                index += 1

            translation += str(index)

            if idx < len(indexList) - 1:
                translation += ", "

        return translation


def translate(ast, targetDict, tags, storeVars):
    translator = Translator(targetDict, tags)
    try:
        programName = os.path.basename(ast["FilePath"]).split(".")[0]
    except KeyError:
        programName = "Unknown"

    return translator.translateProgram(ast,
                                       programName,
                                       storeVars)


def loadTargetDict(targetJsonPath):
    try:
        with open(targetJsonPath, "r") as jsonDictFile:
            return json.load(jsonDictFile)
    except IOError as err:
        if err.errno == 2:
            print("Target \"" + targetJsonPath + "\" does not exist")
        else:
            print(err)
        raise Exception("Could not load target dictionary")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t",
                        "--target",
                        nargs='?',
                        help="Translation target. Possible values: cpp, python, java, r, octave, csharp, ruby, lua. (default: python)")
    parser.add_argument("path",
                        nargs='?',
                        help="Path to input file. If not specified input is read from stdin")
    parser.add_argument("-g", "--ctags", help="path to ctags file")
    parser.add_argument("--store-vars",
                        help="whether to store all variables for testing",
                        action='store_true')
    args = parser.parse_args()

    # Load target dictionary
    target = "python"
    if args.target:
        target = args.target

    targetsDirPath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'targets'
    )
    targetDict = loadTargetDict(os.path.join(targetsDirPath, target + ".json"))

    # Load ctags file
    tags = {}
    if args.ctags:
        from generate import parseCtags
        tags = parseCtags(args.ctags)

    storeVars = True if args.store_vars else False

    # Read from input file (stdin or given path)
    programObject = None
    if args.path:
        with open(args.path, "r") as inputFile:
            programObject = json.load(inputFile)
    else:
        programObject = json.load(sys.stdin)

    print(translate(programObject,
                    targetDict,
                    tags=tags,
                    storeVars=storeVars))
