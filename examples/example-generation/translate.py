import json
import sys
from string import Template
from sets import Set

class Translator:
    def __init__(self, targetDictPath):
        self.dependencies = Set()
        with open(targetDictPath, "r") as targetDict:
            self.targetDict = json.loads(targetDict.read())

    def translateProgram(self, program):
        """ Translate program AST
        Args:
            program: object like [statementAST, statementAST, statementAST, ...]
        """
        self.dependencies = Set() # reset dependencies
        targetProgram = ""
        for statement in program:
            targetProgram += self.translateStatement(statement["Statement"])
        return self.dependencyImportString() + targetProgram

    def dependencyImportString(self):
        """ Returns dependency import string 
            e.g. for python: "from modshogun import RealFeatures\n\n" 
        """
        dependencyList = list(self.dependencies)
        importTemplate = Template(self.targetDict["ImportDependencies"])
        csdependencies = "" # comma separated dependencies
        for i, x in enumerate(dependencyList):
            csdependencies += x
            if i < len(dependencyList)-1:
                csdependencies += ", "
        return importTemplate.substitute(dependencies=csdependencies)

    def translateStatement(self, statement):
        """ Translate statement AST
        Args:
            statement: object like {"Init": initAST}, {"Assign": assignAST},
                       {"Expr": exprAST}, "\n"
        """
        if statement == "\n": # Newline handling
            return "\n"

        type = statement.keys()[0]
        translation = None
        if type == "Init":
            translation = self.translateInit(statement["Init"])
        elif type == "Assign":
            template = Template(self.targetDict["Assign"])
            name = statement["Assign"][0]["Identifier"]
            expr = self.translateExpr(statement["Assign"][1]["Expr"])
            translation = template.substitute(name=name, expr=expr)
        elif type == "Expr":
            translation = self.translateExpr(statement["Expr"])

        if translation == None:
            raise Exception("Unknown statment type: " + type)

        template = Template(self.targetDict["Statement"])
        return template.substitute(statement=translation)

    def translateInit(self, init):
        """ Translate initialisation statement AST
        Args:
            init: object like [typeAST, identifierAST, exprAST],
                  [typeAST, identifierAST, argumentListAST], etc.
        """
        typeString = self.translateType(init[0])
        nameString = init[1]["Identifier"]
        initialisation = init[2]

        if initialisation.keys()[0] == "Expr":
            template = Template(self.targetDict["Init"]["Copy"])
            exprString = self.translateExpr(initialisation["Expr"])
            return template.substitute(name=nameString, type=typeString, expr=exprString)

        elif initialisation.keys()[0] == "ArgumentList":
            self.dependencies.add(typeString)
            template = Template(self.targetDict["Init"]["Construct"])
            argsString = self.translateArgumentList(initialisation["ArgumentList"])
            return template.substitute(name=nameString, type=typeString, arguments=argsString)

    def translateExpr(self, expr):
        """ Translate expression AST
        Args:
            expr: object like {"MethodCall": [identifierAST, identifierAST, argumentListAST]},
                  {"BoolLiteral": "False"}, {"StringLiteral": "train.dat"}, {"NumberLiteral": 4},
                  {"Identifier": "feats_test"}, etc.
        """
        key = expr.keys()[0]
        if key == "MethodCall":
            template = Template(self.targetDict["Expr"]["MethodCall"])
            object = expr[key][0]["Identifier"]
            method = expr[key][1]["Identifier"]
            argsList = None
            try:
                argsList = expr[key][2]
            except IndexError:
                pass
            translatedArgsList = self.translateArgumentList(argsList)

            return template.substitute(object=object, method=method, arguments=translatedArgsList)

        elif key == "BoolLiteral":
            return self.targetDict["Expr"]["BoolLiteral"][expr[key]]

        elif key == "StringLiteral":
            template = Template(self.targetDict["Expr"]["StringLiteral"])
            return template.substitute(literal=expr[key])

        elif key == "NumberLiteral":
            template = Template(self.targetDict["Expr"]["NumberLiteral"])
            return template.substitute(number=expr[key])

        elif key == "Identifier":
            template = Template(self.targetDict["Expr"]["Identifier"])
            return template.substitute(identifier=expr[key])

        raise Exception("Unknown expression type: " + key)

    def translateType(self, type):
        """ Translate type AST
        Args:
            type: object like {"ObjectType": "IntMatrix"}, {"BasicType": "float"}, etc.
        """
        template = ""
        typeKey = type.keys()[0]
        if typeKey == "ObjectType":
            template = Template(self.targetDict["Type"]["ObjectType"])
        elif typeKey == "BasicType":
            template = Template(self.targetDict["Type"]["BasicType"])
        else:
            raise Exception("Invalid type key: " + type.keys()[0])

        return template.substitute(type=type[typeKey])

    def translateArgumentList(self, argumentList):
        """ Translate argument list AST
        Args:
            argumentList: object like None, {"Expr": exprAST},
                          [{"Expr": exprAST}, {"Expr": exprAST}], etc.
        """
        if argumentList == None:
            return ""
        if isinstance(argumentList, list):
            head = argumentList[0]
            tail = argumentList[1:]

            translation = self.translateArgumentList(head)
            if len(tail) > 0:
                translation += ", " + self.translateArgumentList(tail)

            return translation
        else:
            if "Expr" in argumentList:
                return self.translateExpr(argumentList["Expr"])
            elif "ArgumentList" in argumentList:
                return self.translateArgumentList(argumentList["ArgumentList"])

if __name__ == "__main__":
    translator = Translator("targets/python.json")

    programString = sys.stdin.read()
    programObject = json.loads(programString)
    print translator.translateProgram(programObject["Program"])