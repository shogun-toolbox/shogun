import json

class BaseNode(object):
    def __init__(self, string, location, tokens):
        self.tokens = tokens
        self.string = string
        self.location = location

class Program(BaseNode): pass
class Statement(BaseNode): pass
class Init(BaseNode): pass
class Assign(BaseNode): pass
class Identifier(BaseNode): pass
class ArgumentList(BaseNode): pass
class Expr(BaseNode): pass
class MethodCall(Expr): pass
class Type(BaseNode): pass # Abstract. Don't use directly
class BasicType(Type): pass
class ObjectType(Type): pass
class Literal(Expr): pass # Abstract. Don't use directly
class BoolLiteral(Literal): pass
class NumberLiteral(Literal): pass
class StringLiteral(Literal): pass

class JSONEncoder(json.JSONEncoder):
    def default(self, node):
        if issubclass(node.__class__, BaseNode):
            if len(node.tokens) > 1:
                # If tokens is a list we only consider instances of BaseClass subclasses
                objectList = [self.default(element) for element in node.tokens if issubclass(element.__class__, BaseNode)]
                if len(objectList) == 1:
                    objectList = objectList[0]

                return {node.__class__.__name__: objectList}
            else:
                return {node.__class__.__name__:self.default(node.tokens[0])}
        return node