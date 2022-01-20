import javalang
import re

from contextlib import contextmanager

from .token import TokenTypes, TokenizationState


class VarScope:

    def __init__(self):
        self.scopes = [set()]
    
    def __contains__(self, key):
        return any(key in scope for scope in self.scopes)

    def add(self, key):
        self.scopes[-1].add(key)
    
    def new_scope(self):
        self.scopes.append(set())
    
    def pop_scope(self):
        self.scopes.pop(-1)



class NodeTokenizer:

    def __init__(self, state):
        self.state = state
        self.scope = VarScope()

    def __call__(self, node): return self.tokenize(node)

    def tokenize(self, node):
        target_func = getattr(self, f"tokenize_{type(node).__name__}", None)
        if target_func is not None:
            target_func(node)
        else:
            raise ValueError("Cannot handle node type: %s" % type(node).__name__)

    # Utils ------------------------------------

    def tokenize_List(self, L, divider=","):

        lastelem = len(L) - 1
        for i, e in enumerate(L):
            self.tokenize(e)
            if i != lastelem: self.state.add(divider) 


    @contextmanager
    def brackets(self, lpar = "(", rpar = ")"):
        self.state.add(lpar)
        yield
        self.state.add(rpar)


    @contextmanager
    def pre_post(self, node):

        if node.prefix_operators is not None:
            for op in node.prefix_operators:
                self.state.add(op, TokenTypes.UOP)

        yield

        if hasattr(node, "selectors") and node.selectors is not None:
            self.tokenize_selectors(node)

        if node.postfix_operators is not None:
            for op in node.postfix_operators:
                self.state.add(op, TokenTypes.UOP)

    @contextmanager
    def var_scope(self):
        self.scope.new_scope()

        yield

        self.scope.pop_scope()


    def _identify_name(self, name):

        if name in self.scope: return TokenTypes.USE_VAR
        if name[0].isupper(): return TokenTypes.USE_TYPE

        if self.state.context == "USE":
            return TokenTypes.ATTR
        else:
            return TokenTypes.NAME


    def tokenize_qualifier(self, node, base_type):

        access = node.split(".")

        lastaccess = len(access) -1
        for i, qual in enumerate(access):
            if i == 0:
                self.state.add(qual, base_type(qual) if callable(base_type) else base_type)
            else:
                self.state.add(qual, TokenTypes.ATTR)
            if i != lastaccess: self.state.add(".")

    def tokenize_selectors(self, node):
        if node.selectors is not None:
            for selector in node.selectors:
                
                if type(selector).__name__ in ["MemberReference", "MethodInvocation"]:
                    self.state.add(".")

                self.tokenize(selector)

    def tokenize_label(self, node):
        if node.label is not None:
            self.tokenize(node.label)
            self.state.add(":")


    def tokenize_body(self, body):

        if not isinstance(body, list): return self.tokenize(body)

        with self.brackets("{", "}"):
            with self.var_scope():
                for statement in body:
                    if isinstance(statement, list):
                        self.tokenize_body(statement)
                    else:
                        self.tokenize(statement)


    def tokenize_dimensions(self, node):
        if node.dimensions is not None:
            for d in node.dimensions:
                if d is None:
                    self.state.add("[]")
                else:
                    with self.brackets("[", "]"):
                        self.tokenize(d)

    
    def tokenize_operand(self, node):
        name = type(node).__name__

        if name in ["BinaryOperation", "Assignment", "TernaryExpression"]:
            with self.brackets(): self.tokenize(node)
        else:
            self.tokenize(node)


    def tokenize_NoneType(self, node):
        return

    def tokenize_str(self, node):
        self.state.add(node, TokenTypes.NAME)

    # Elements --------------------------------

    def tokenize_Annotation(self, node):
        self.state.add("@")
        self.state.add(node.name, TokenTypes.NAME)

        if node.element is not None:
            with self.brackets():
                if isinstance(node.element, list):
                    self.tokenize_List(node.element)
                else:
                    self.tokenize(node.element)

    def tokenize_ElementValuePair(self, node):
        self.state.add(node.name, TokenTypes.NAME)
        self.state.add("=")
        self.tokenize_body(node.value)

    def tokenize_ElementArrayValue(self, node):
        with self.brackets("{", "}"):
            self.tokenize_List(node.values)

    # Types ------------------------------------

    def tokenize_BasicType(self, node):
        self.state.add(node.name, TokenTypes.USE_TYPE)
        self.tokenize_dimensions(node)

    def tokenize_ReferenceType(self, node):
        self.state.add(node.name, TokenTypes.USE_TYPE)
        
        if node.arguments is not None:
            with self.brackets("<", ">"):
                self.tokenize_List(node.arguments)

        self.tokenize_dimensions(node)

        if node.sub_type is not None:
            self.state.add(".")
            self.tokenize(node.sub_type)


    def tokenize_TypeParameter(self, node):
        self.state.add(node.name, TokenTypes.NAME)

        if node.extends is not None:
            self.state.add("extends", TokenTypes.KEYWORDS)
            self.tokenize_List(node.extends, "&")


    # Parameter --------------------------------

    def tokenize_FormalParameter(self, node):
        
        for annotation in node.annotations:
            self.tokenize(annotation)
        
        for modifier in node.modifiers:
            self.state.add(modifier, TokenTypes.KEYWORDS)
        
        self.tokenize(node.type)
        self.state.add(node.name, TokenTypes.DEF_VAR)
        self.scope.add(node.name)

        if node.varargs: self.state.add("*")

    def tokenize_InferredFormalParameter(self, node):
        self.state.add(node.name, TokenTypes.DEF_VAR)
        self.scope.add(node.name)

    def tokenize_LocalVariableDeclaration(self, node):
        for ann in node.annotations:
            self.tokenize(ann)
        
        for modifier in node.modifiers:
            self.state.add(modifier, TokenTypes.KEYWORDS)

        self.tokenize(node.type)
        self.tokenize_List(node.declarators)
        self.state.add(";")

    def tokenize_VariableDeclaration(self, node):
        for annotation in node.annotations:
            self.tokenize(annotation)
        
        for modifier in node.modifiers:
            self.state.add(modifier, TokenTypes.KEYWORDS)
        
        self.tokenize(node.type)
        self.tokenize_List(node.declarators)


    def tokenize_VariableDeclarator(self, node):
        self.state.add(node.name, TokenTypes.DEF_VAR)
        self.scope.add(node.name)
        self.tokenize_dimensions(node)

        if node.initializer is not None:
            self.state.add("=")
            self.tokenize(node.initializer)


    # Statements --------------------------------

    def tokenize_IfStatement(self, node):
        self.tokenize_label(node)

        self.state.add("if", TokenTypes.KEYWORDS)
        
        with self.brackets():
            self.tokenize(node.condition)

        self.tokenize(node.then_statement)

        if node.else_statement is not None:
            self.state.add("else", TokenTypes.KEYWORDS)
            self.tokenize(node.else_statement)

    def tokenize_WhileStatement(self, node):
        self.tokenize_label(node)
        self.state.add("while", TokenTypes.KEYWORDS)

        with self.brackets():
            self.tokenize(node.condition)
        
        self.tokenize_body(node.body)


    def tokenize_DoStatement(self, node):
        self.tokenize_label(node)
        self.state.add("do", TokenTypes.KEYWORDS)
        
        self.tokenize_body(node.body)

        self.state.add("while")
        with self.brackets():
            self.tokenize(node.condition)
        
        self.state.add(";")


    def tokenize_ForStatement(self, node):
        self.tokenize_label(node)

        self.state.add("for", TokenTypes.KEYWORDS)

        with self.var_scope():
            with self.brackets():
                self.tokenize(node.control)
            self.tokenize_body(node.body)

    def tokenize_AssertStatement(self, node):
        self.tokenize_label(node)
        self.state.add("assert", TokenTypes.KEYWORDS)
        self.tokenize(node.condition)

        if node.value is not None:
            self.state.add(",")
            self.tokenize(node.value)
        
        self.state.add(";")


    def tokenize_BreakStatement(self, node):
        self.tokenize_label(node)
        self.state.add("break", TokenTypes.KEYWORDS)

        if node.goto is not None:
            self.state.add(node.goto, TokenTypes.NAME)

        self.state.add(";")

    def tokenize_ContinueStatement(self, node):
        self.tokenize_label(node)
        self.state.add("continue", TokenTypes.KEYWORDS)

        if node.goto is not None:
            self.state.add(node.goto, TokenTypes.NAME)

        self.state.add(";")

    def tokenize_ReturnStatement(self, node):
        self.tokenize_label(node)
        self.state.add("return", TokenTypes.KEYWORDS)

        if node.expression is not None:
            self.tokenize(node.expression)

        self.state.add(";")

    def tokenize_ThrowStatement(self, node):
        self.tokenize_label(node)
        self.state.add("throw", TokenTypes.KEYWORDS)
        self.tokenize(node.expression)
        self.state.add(";")

    def tokenize_SynchronizedStatement(self, node):
        self.tokenize_label(node)

        self.state.add("synchronized", TokenTypes.KEYWORDS)

        with self.brackets():
            self.tokenize(node.lock)

        with self.brackets("{", "}"):
            for statement in node.block:
                self.tokenize(statement)


    def tokenize_TryStatement(self, node):
        self.tokenize_label(node)
        self.state.add("try", TokenTypes.KEYWORDS)

        if node.resources is not None:
            with self.brackets():
                self.tokenize_List(node.resources)

        with self.brackets("{", "}"):
            for statement in node.block:
                self.tokenize(statement)
        
        if node.catches is not None:
            for catch in node.catches:
                self.tokenize(catch)

        if node.finally_block is not None:
            self.state.add("finally", TokenTypes.KEYWORDS)
            with self.brackets("{", "}"):
                for statement in node.finally_block:
                    self.tokenize(statement)

    def tokenize_SwitchStatement(self, node):
        self.tokenize_label(node)
        self.state.add("switch", TokenTypes.KEYWORDS)

        with self.brackets():
            self.tokenize(node.expression)
        
        with self.brackets("{", "}"):
            for case in node.cases:
                self.tokenize(case)


    def tokenize_BlockStatement(self, node):
        self.tokenize_label(node)

        with self.var_scope():
            with self.brackets("{", "}"):
                for statement in node.statements:
                    self.tokenize(statement)


    def tokenize_StatementExpression(self, node):
        self.tokenize_label(node)
        with self.state.new_ctx("USE"):
            self.tokenize(node.expression)
        self.state.add(";")

    def tokenize_Statement(self, node):
        self.tokenize_label(node)
        self.state.add(";")

    # Try-Catch ----------------------------------

    def tokenize_TryResource(self, node):
        for annotation in node.annotations:
            self.tokenize(annotation)

        for modifier in node.modifiers:
            self.state.add(modifier, TokenTypes.KEYWORDS)

        self.tokenize(node.type)
        self.state.add(node.name, TokenTypes.DEF_VAR)

        if node.value is not None:
            self.state.add("=")

            with self.state.new_ctx("USE"):
                self.tokenize(node.value)


    def tokenize_CatchClause(self, node):
        self.tokenize_label(node)
        self.state.add("catch", TokenTypes.KEYWORDS)

        with self.var_scope():
            with self.brackets():
                self.tokenize(node.parameter)
            
            with self.brackets("{", "}"):
                for statement in node.block:
                    self.tokenize(statement)


    def tokenize_CatchClauseParameter(self, node):
        if node.annotations is not None:
            for ann in node.annotations:
                self.tokenize(ann)
        
        if node.modifiers is not None:
            for modifier in node.modifiers:
                self.state.add(modifier, TokenTypes.KEYWORDS)
        
        last_type = len(node.types) - 1
        for i, type in enumerate(node.types):
            self.state.add(type, TokenTypes.USE_TYPE)
            if i != last_type: self.state.add("|")

        self.state.add(node.name, TokenTypes.DEF_VAR)
        self.scope.add(node.name)

    # Control ------------------------------------

    def tokenize_SwitchStatementCase(self, node):
        self.state.add("case", TokenTypes.KEYWORDS)
        self.tokenize_List(node.case)
        self.state.add(":")

        for statement in node.statements:
            self.tokenize(statement)


    def tokenize_ForControl(self, node):
        
        if isinstance(node.init, list):
            self.tokenize_List(node.init)
        else:
            self.tokenize(node.init)

        self.state.add(";")

        if node.condition is not None:
            self.tokenize(node.condition)

        self.state.add(";")

        if node.update is not None:
            self.tokenize_List(node.update)

    def tokenize_EnhancedForControl(self, node):
        self.tokenize(node.var)
        self.state.add(":")
        self.tokenize(node.iterable)

    # Expression ---------------------------------

    def tokenize_MethodInvocation(self, node):


        with self.pre_post(node):

            if node.qualifier is not None and len(node.qualifier) > 0:
                self.tokenize_qualifier(node.qualifier, self._identify_name) 
                self.state.add(".")

            if node.type_arguments is not None:
                with self.brackets("<", ">"):
                    self.tokenize_List(node.type_arguments)

            self.state.add(node.member, TokenTypes.CALL_FUNC)

            with self.brackets():
                self.tokenize_List(node.arguments)

        

    def tokenize_SuperMethodInvocation(self, node):
        with self.pre_post(node):

            self.state.add("super", TokenTypes.KEYWORDS)
            self.state.add(".")

            if node.qualifier is not None:
                self.tokenize_qualifier(node.qualifier, self._identify_name)
                self.state.add(".")
                
            self.state.add(node.member, TokenTypes.CALL_FUNC)

            with self.brackets():
                self.tokenize_List(node.arguments)


    def tokenize_SuperConstructorInvocation(self, node):
        with self.pre_post(node):
            self.state.add("super", TokenTypes.KEYWORDS)

            with self.brackets():
                self.tokenize_List(node.arguments)
            


    def tokenize_Assignment(self, node):
        
        with self.state.new_ctx("DEF"):
            self.tokenize(node.expressionl)

        self.state.add(node.type)

        with self.state.new_ctx("USE"):
            self.tokenize(node.value)

    def tokenize_TernaryExpression(self, node):
        with self.state.new_ctx("USE"):
            self.tokenize_operand(node.condition)
            self.state.add("?")
            self.tokenize_operand(node.if_true)
            self.state.add(":")
            self.tokenize_operand(node.if_false)


    def tokenize_BinaryOperation(self, node):
        with self.state.new_ctx("USE"):
            self.tokenize_operand(node.operandl)
            self.state.add(node.operator, TokenTypes.BOP)
            self.tokenize_operand(node.operandr)


    def tokenize_Cast(self, node):
        with self.brackets(): 
            self.tokenize(node.type)
        self.tokenize_operand(node.expression)

    def tokenize_MethodReference(self, node):
        self.tokenize(node.expression)
        self.state.add("::")
        with self.state.new_ctx("NAME"):
            self.tokenize(node.method)

    def tokenize_LambdaExpression(self, node):

        with self.var_scope():
            with self.brackets():
                last_param = len(node.parameters) - 1
                for i, param in enumerate(node.parameters):
                    self.tokenize(param)
                    if i != last_param: self.state.add(",")

            self.state.add("->")

            self.tokenize_body(node.body)

    # Basics ------------------------------------
    
    def tokenize_Literal(self, node):
        
        with self.pre_post(node):
            
            if "\"" in node.value or "\'" in node.value:
                lit_id = TokenTypes.LIT_STRING
            else:
                lit_id = TokenTypes.LIT_NUMBER
            
            self.state.add(node.value, lit_id)


    def tokenize_This(self, node):
        with self.pre_post(node):
            if node.qualifier is not None:
                self.tokenize_qualifier(node. qualifier, self._identify_name)
                self.state.add(".")

            self.state.add("this", TokenTypes.KEYWORDS)


    def tokenize_MemberReference(self, node):
        
        with self.pre_post(node):
            member_name = node.member
            if node.qualifier is not None and len(node.qualifier) > 0: 
                member_name = node.qualifier + "." + member_name
            self.tokenize_qualifier(member_name, self._identify_name)


    def tokenize_SuperMemberReference(self, node):
        with self.pre_post(node):

            self.state.add("super", TokenTypes.KEYWORDS)
            self.state.add(".")
            self.state.add(node.member, TokenTypes.ATTR)

    def tokenize_ArraySelector(self, node):
        with self.brackets("[", "]"):
            with self.state.new_ctx("USE"):
                self.tokenize(node.index)

    def tokenize_ClassReference(self, node):
        with self.pre_post(node):
            
            if node.qualifier is not None and len(node.qualifier) > 0:
                self.tokenize_qualifier(node.qualifier, self._identify_name)
                self.state.add(".")

            self.tokenize(node.type)
            self.state.add(".class", TokenTypes.KEYWORDS)


    def tokenize_VoidClassReference(self, node):
        with self.pre_post(node):
            self.state.add("void.class", TokenTypes.KEYWORDS)


    def tokenize_TypeArgument(self, node):

        if node.pattern_type is not None:

            if node.pattern_type == "?":
                self.state.add("?", TokenTypes.KEYWORDS)
            else:
                self.state.add("?", TokenTypes.KEYWORDS)
                self.state.add(node.pattern_type, TokenTypes.KEYWORDS)
        
        if node.type is not None:
            self.tokenize(node.type)


    # Creator -----------------------------------

    def tokenize_ArrayCreator(self, node):
        with self.pre_post(node):
            self.state.add("new", TokenTypes.KEYWORDS)
            self.tokenize(node.type)
            for d in node.dimensions:
                if d is None:
                    self.state.add("[]")
                else:
                    self.state.add("[")
                    self.tokenize(d)
                    self.state.add("]")

            self.tokenize(node.initializer)

    def tokenize_ArrayInitializer(self, node):
        with self.brackets("{", "}"):
            self.tokenize_List(node.initializers)

    def tokenize_InnerClassCreator(self, node):
        with self.pre_post(node):

            if node.qualifier is not None:
                self.tokenize_qualifier(node.qualifier, self._identify_name)
            self.state.add(".new", TokenTypes.KEYWORDS)

            self.tokenize(node.type)
            
            with self.brackets():
                self.tokenize_List(node.arguments)

            if node.body is not None:
                self.tokenize_body(node.body)


    def tokenize_ClassCreator(self, node):
        with self.pre_post(node):
            self.state.add("new", TokenTypes.KEYWORDS)
            self.tokenize(node.type)
            
            with self.brackets():
                self.tokenize_List(node.arguments)

            if node.body is not None:
                self.tokenize_body(node.body)
            
            if node.constructor_type_arguments is not None:
                raise NotImplementedError("Constructor type arguments: "+ str(node))

            if node.qualifier is not None:
                raise NotImplementedError("Qualifier: "+str(node))

            


    # Declaration --------------------------------

    def tokenize_MethodDeclaration(self, node):
        if node.documentation is not None:
            self.state.add(node.documentation)

        for annotation in node.annotations:
            self.tokenize(annotation)

        for modifier in ["public", "private", "protected", "static",
                             "final", "abstract", "transient", "synchronized", "volatile"]:
            if modifier in node.modifiers:
                self.state.add(modifier, TokenTypes.KEYWORDS)

        if node.type_parameters is not None:
            with self.brackets("<", ">"):
                self.tokenize_List(node.type_parameters)

        if node.return_type is None:
            self.state.add("void", TokenTypes.USE_TYPE)
        else:
            if isinstance(node.return_type, str):
                self.state.add(node.return_type, TokenTypes.USE_TYPE)
            else:
                self.tokenize(node.return_type)

        self.state.add(node.name, TokenTypes.DEF_FUNC)

        with self.brackets():
            self.tokenize_List(node.parameters)

        if node.throws is not None: 

            self.state.add("throws", TokenTypes.KEYWORDS)

            lastthrow = len(node.throws) - 1
            for i, throw in enumerate(node.throws):
                self.state.add(throw, TokenTypes.USE_TYPE)
                if i != lastthrow: self.state.add(",")
        
        self.tokenize_body(node.body)

    def tokenize_FieldDeclaration(self, node):
        if node.documentation is not None:
            self.state.add(node.documentation)
        
        for annotation in node.annotations:
            self.tokenize(annotation)

        for modifier in ["public", "private", "protected", "static",
                             "final", "abstract", "transient", "synchronized", "volatile"]:
            if modifier in node.modifiers:
                self.state.add(modifier, TokenTypes.KEYWORDS)
        
        self.tokenize(node.type)

        for declarator in node.declarators:
            self.tokenize(declarator)
        
        self.state.add(";")

    def tokenize_ClassDeclaration(self, node):

        for annotation in node.annotations:
            self.tokenize(annotation)

        for modifier in ["public", "private", "protected", "static",
                             "final", "abstract", "transient", "synchronized", "volatile"]:
            if modifier in node.modifiers:
                self.state.add(modifier, TokenTypes.KEYWORDS)
        
        self.state.add("class", TokenTypes.KEYWORDS)
        self.state.add(node.name)

        if node.implements is not None:
            self.state.add("implements", TokenTypes.KEYWORDS)
            self.tokenize_List(node.implements)

        if node.extends is not None:
            self.state.add("extends", TokenTypes.KEYWORDS)
            self.tokenize(node.extends)

        self.tokenize_body(node.body)

    def tokenize_ConstructorDeclaration(self, node):
        for annotation in node.annotations:
            self.tokenize(annotation)

        for modifier in ["public", "private", "protected", "static",
                             "final", "abstract", "transient", "synchronized", "volatile"]:
            if modifier in node.modifiers:
                self.state.add(modifier, TokenTypes.KEYWORDS)

        self.state.add(node.name, TokenTypes.NAME)

        with self.brackets():
            self.tokenize_List(node.parameters)

        if node.throws is not None: 

            self.state.add("throws", TokenTypes.KEYWORDS)

            lastthrow = len(node.throws) - 1
            for i, throw in enumerate(node.throws):
                self.state.add(throw, TokenTypes.USE_TYPE)
                if i != lastthrow: self.state.add(",")
        
        self.tokenize_body(node.body)


typing_pattern  = re.compile("\.<[A-Z].*>")
typing_pattern2 = re.compile("::<[A-Z].*>")

def _code_preprocess(code):
    """
    Javalang does not support typed method calls until now
    Issue: https://github.com/c2nes/javalang/issues/105

    Since the call structure is more important than the typing, we remove the typing
    """
    code = typing_pattern.sub(".", code)
    code = typing_pattern2.sub("::", code)
    return code


def _parse(code):
    code = _code_preprocess(code)

    return javalang.parse.parse(code)


def _method_trees(code):
    tree = _parse(code)

    for _, node in tree.filter(javalang.tree.MethodDeclaration):
        yield node.name, node


def method_tokenize(code):

    method_defs = {}

    for name, subtree in _method_trees(code):

        state = TokenizationState()
        NodeTokenizer(state)(subtree)

        method_defs[name] = {
            "tokens": state.tokens,
            "types" : state.type_ids
        }
    
    return method_defs



# Debug option --------------------------------------------------------------


def _is_isomorph(source_tree, target_tree):

    if type(source_tree).__name__ != type(target_tree).__name__:
        return False

    if hasattr(source_tree, "children"):
        if not hasattr(target_tree, "children"): return False
        source_tree = source_tree.children
        target_tree = target_tree.children

    if isinstance(source_tree, str):
        return source_tree == target_tree

    try:
        if len(source_tree) != len(target_tree): return False

        for i, source in enumerate(source_tree):
            target = target_tree[i]
            if not _is_isomorph(source, target): return False
        
        return True
    except TypeError:
        return source_tree == target_tree


def check_parsable(code, raise_error = False):
    """
    Parses a given code twice to AST (once before tokenization, once after).
    The resulting trees have to be isomorp to each other.
    """
    tokenization = method_tokenize(code)
    source_trees = {k: v for k, v in _method_trees(code)}

    # Now parse the tokenized code
    target_trees = {}

    for name, tokenization_result in tokenization.items():
        tokens = tokenization_result["tokens"]
        token_code = " ".join(tokens)
        dummy_code = "public class Test { %s }" % token_code # Necessary for being parsable in Java
        token_trees = {k: v for k, v in _method_trees(dummy_code)}
        target_trees[name] = token_trees[name]

    # Check whether all trees are isomorph
    for name, source_tree in source_trees.items():
        target_tree = target_trees[name]

        if not _is_isomorph(source_tree, target_tree):

            if raise_error:
                raise ValueError("The result before and after tokenization differ for function %s! " % name)
            else:
                return False

    return True
