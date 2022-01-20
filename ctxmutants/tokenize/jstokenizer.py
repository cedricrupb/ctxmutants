import esprima

from contextlib import contextmanager

from .token import TokenTypes, TokenizationState


class NodeTokenizer:

    def __init__(self, state):
        self.state = state
        self.scope = set()

    def __call__(self, node): return self.tokenize(node)

    def tokenize(self, node):
        node_type = node.type if node is not None else "None"

        target_func = getattr(self, f"tokenize_{node_type}", None)
        if target_func is not None:
            target_func(node)
        else:
            raise ValueError("Cannot handle node type: %s" % node_type)

    # Helpers --------------------------------------------------------------------

    @contextmanager
    def brackets(self, lpar = "(", rpar = ")"):
        self.state.add(lpar)
        yield
        self.state.add(rpar)

    def tokenize_List(self, L, divider=","):

        lastelem = len(L) - 1
        for i, e in enumerate(L):
            self.tokenize(e)
            if i != lastelem: self.state.add(divider) 

    def tokenize_None(self, node):
        # NOOP
        pass

    # Identifier -----------------------------------------------------------------

    def tokenize_Identifier(self, node):

        state   = self.state
        name    = node.name
        context = state.context

        # Assume that all names follow convention
        # Therefore: USE => USE_TYPE if name is camel case
        if context == "USE":
            if len(name) > 1 and name[0].isupper() and name[1].islower():
                context = "USE_TYPE"

        ctx2id  = {"FUNC_DEF": TokenTypes.DEF_FUNC, 
                    "Param": TokenTypes.DEF_VAR, 
                    "Annotation": TokenTypes.USE_TYPE,
                    "FUNC_CALL": TokenTypes.CALL_FUNC,
                    "EXPR": TokenTypes.USE_VAR,
                    "USE": TokenTypes.USE_VAR,
                    "DEF": TokenTypes.DEF_VAR,
                    "ATTR": TokenTypes.ATTR,
                    "USE_TYPE": TokenTypes.USE_TYPE,
                    "NAME": TokenTypes.NAME}
        type_id = ctx2id[context]

        state.add(name, type_id) 

    def tokenize_Literal(self, node):
        
        value = node.value
        type = TokenTypes.LIT_NUMBER

        if isinstance(value, str):
            type = TokenTypes.LIT_STRING
        
        self.state.add(node.raw, type)

    def tokenize_TemplateLiteral(self, node):
        
        quasis      = node.quasis
        expressions = iter(node.expressions)

        with self.brackets("`", "`"):
            for q in quasis:
                self.tokenize(q)

                if not q.tail:
                    expression = next(expressions)
                    
                    with self.brackets("${", "}"):
                        with self.state.new_ctx("USE"):
                            self.tokenize(expression)


    def tokenize_TemplateElement(self, node):
        self.state.add(node.value.raw, TokenTypes.LIT_STRING)


    # FunctionDeclaration --------------------------------------------------------


    def tokenize_FunctionDeclaration(self, node):

        if node.isAsync: self.state.add("async", TokenTypes.KEYWORDS)

        if node.generator:
            self.state.add("function*", TokenTypes.KEYWORDS)
        else:
            self.state.add("function", TokenTypes.KEYWORDS)

        with self.state.new_ctx("FUNC_DEF"):
            self.tokenize(node.id)

        with self.brackets():
            with self.state.new_ctx("Param"):
                self.tokenize_List(node.params)

        self.tokenize(node.body)

    # Statements -------------------------------------------------------------------

    def tokenize_BlockStatement(self, node):

        with self.brackets("{", "}"):
            for stmt in node.body: self.tokenize(stmt)

    def tokenize_ReturnStatement(self, node):

        self.state.add("return", TokenTypes.KEYWORDS)

        with self.state.new_ctx("USE"):
            self.tokenize(node.argument)
        
        self.state.add(";")

    def tokenize_BreakStatement(self, node):

        self.state.add("break", TokenTypes.KEYWORDS)

        if node.label is not None:
            with self.state.new_ctx("NAME"):
                self.tokenize(node.label)
        self.state.add(";")


    def tokenize_ContinueStatement(self, node):
        self.state.add("continue", TokenTypes.KEYWORDS)

        if node.label is not None:
            with self.state.new_ctx("NAME"):
                self.tokenize(node.label)
        self.state.add(";")


    def tokenize_DebuggerStatement(self, node):
        self.state.add("debugger", TokenTypes.KEYWORDS)
        self.state.add(";")

    def tokenize_DoWhileStatement(self, node):
        self.state.add("do", TokenTypes.KEYWORDS)
        self.tokenize(node.body)

        self.state.add("while", TokenTypes.KEYWORDS)
        
        with self.brackets():
            with self.state.new_ctx("USE"):
                self.tokenize(node.test)
        
        self.state.add(";")


    def tokenize_EmptyStatement(self, node):
        self.state.add(";")

    def tokenize_ExpressionStatement(self, node):
        
        with self.state.new_ctx("USE"):
            self.tokenize(node.expression)
        self.state.add(";")


    def tokenize_ForStatement(self, node):

        self.state.add("for", TokenTypes.KEYWORDS)

        with self.brackets():
            self.tokenize(node.init)

            self.state.add(";")

            with self.state.new_ctx("USE"):
                self.tokenize(node.test)
            self.state.add(";")

            self.tokenize(node.update)

        self.tokenize(node.body)

    def tokenize_ForInStatement(self, node):
        self.state.add("for", TokenTypes.KEYWORDS)

        with self.brackets():

            with self.state.new_ctx("DEF"):
                self.tokenize(node.left)
            
            self.state.add("in", TokenTypes.KEYWORDS)

            with self.state.new_ctx("USE"):
                self.tokenize(node.right)
        
        self.tokenize(node.body)



    def tokenize_ForOfStatement(self, node):
        self.state.add("for", TokenTypes.KEYWORDS)

        with self.brackets():

            with self.state.new_ctx("DEF"):
                self.tokenize(node.left)
            
            self.state.add("of", TokenTypes.KEYWORDS)

            with self.state.new_ctx("USE"):
                self.tokenize(node.right)
        
        self.tokenize(node.body)


    def tokenize_IfStatement(self, node):

        self.state.add("if", TokenTypes.KEYWORDS)

        with self.brackets():
            with self.state.new_ctx("USE"):
                self.tokenize(node.test)

        self.tokenize(node.consequent)

        if node.alternate is not None:
            self.state.add("else", TokenTypes.KEYWORDS)
            self.tokenize(node.alternate)


    def tokenize_LabeledStatement(self, node):
        with self.state.new_ctx("NAME"):
            self.tokenize(node.label)
        self.state.add(":")

        self.tokenize(node.body)

    def tokenize_SwitchStatement(self, node):
        
        self.state.add("switch", TokenTypes.KEYWORDS)

        with self.brackets():
            with self.state.new_ctx("USE"):
                self.tokenize(node.discriminant)

        with self.brackets("{", "}"):
            for case in node.cases:
                self.tokenize(case)


    def tokenize_ThrowStatement(self, node):
        self.state.add("throw", TokenTypes.KEYWORDS)
        with self.state.new_ctx("USE"):
            self.tokenize(node.argument)
        self.state.add(";")

    def tokenize_TryStatement(self, node):

        self.state.add("try", TokenTypes.KEYWORDS)
        self.tokenize(node.block)

        self.tokenize(node.handler)


    def tokenize_VariableDeclaration(self, node):
        
        self.state.add(node.kind, TokenTypes.KEYWORDS)

        for decl in node.declarations:
            with self.state.new_ctx("DEF"):
                self.tokenize(decl.id)

            if decl.init is not None: 
                self.state.add("=")
                with self.state.new_ctx("USE"):
                    self.tokenize(decl.init)
            
        self.state.add(";")


    def tokenize_WhileStatement(self, node):
        self.state.add("while", TokenTypes.KEYWORDS)
        with self.brackets():
            with self.state.new_ctx("USE"):
                self.tokenize(node.test)

        self.tokenize(node.body)


    def tokenize_WithStatement(self, node):
        self.state.add("with", TokenTypes.KEYWORDS)

        with self.brackets():
            with self.state.new_ctx("USE"):
                self.tokenize(node.object)
        
        self.tokenize(node.body)


    # Expression -------------------------------------------------------------------

    def tokenize_subexpression(self, node):

        node_type = node.type
        
        if "BinaryExpression" in node_type or "LogicalExpression" in node_type:
            with self.brackets(): self.tokenize(node)
        else:
            self.tokenize(node)

    def tokenize_BinaryExpression(self, node):
        
        with self.state.new_ctx("USE"):
            self.tokenize_subexpression(node.left)
            self.state.add(node.operator, TokenTypes.BOP)
            self.tokenize_subexpression(node.right)


    def tokenize_UnaryExpression(self, node):
        
        with self.state.new_ctx("USE"):
            if node.prefix:
                self.state.add(node.operator, TokenTypes.UOP)
                self.tokenize_subexpression(node.argument)
            else:
                self.tokenize_subexpression(node.argument)
                self.state.add(node.operator, TokenTypes.UOP)


    def tokenize_ThisExpression(self, node):
        self.state.add("this", TokenTypes.KEYWORDS)

    def tokenize_ArrayExpression(self, node):
        with self.state.new_ctx("USE"):
            with self.brackets("[", "]"):
                self.tokenize_List(node.elements)

    def tokenize_ObjectExpression(self, node):
        with self.brackets("{", "}"):
            self.tokenize_List(node.properties)


    def tokenize_FunctionExpression(self, node):
        if node.isAsync:
            self.state.add("async", TokenTypes.KEYWORDS)

        context = self.state.context

        if context is None or context != "INNER_FUNC":
            if node.generator:
                self.state.add("function*", TokenTypes.KEYWORDS)
            else:
                self.state.add("function", TokenTypes.KEYWORDS)

        with self.brackets():
            with self.state.new_ctx("Param"):
                self.tokenize_List(node.params)
        
        self.tokenize(node.body)


    def tokenize_ArrowFunctionExpression(self, node):

        with self.state.new_ctx("DEF"):
            self.tokenize_List(node.params)
        
        self.state.add("=>")

        self.tokenize(node.body)


    def tokenize_TaggedTemplateExpression(self, node):
        raise NotImplementedError(str(node))

    def tokenize_MemberExpression(self, node):
        context = self.state.context

        with self.state.new_ctx("USE"):
            self.tokenize(node.object)

        self.state.add(".")

        if context == "FUNC_CALL": 
            sub_context = context
        else:
            sub_context = "ATTR"

        with self.state.new_ctx(sub_context):
            self.tokenize(node.property)


    def tokenize_Super(self, node):
        self.state.add("super", TokenTypes.KEYWORDS)

    def tokenize_MetaProperty(self, node):
        raise NotImplementedError(str(node))
    
    def tokenize_NewExpression(self, node):
        self.state.add("new", TokenTypes.KEYWORDS)

        with self.state.new_ctx("USE_TYPE"):
            self.tokenize(node.callee)

        with self.brackets():
            with self.state.new_ctx("USE"):
                self.tokenize_List(node.arguments)


    def tokenize_CallExpression(self, node):
        
        with self.state.new_ctx("FUNC_CALL"):
            self.tokenize(node.callee)

        with self.brackets():
            with self.state.new_ctx("USE"):
                self.tokenize_List(node.arguments)


    def tokenize_UpdateExpression(self, node):
        with self.state.new_ctx("USE"):
            if node.prefix:
                self.state.add(node.operator, TokenTypes.UOP)
                self.tokenize(node.argument)
            else:
                self.tokenize(node.argument)
                self.state.add(node.operator, TokenTypes.UOP)

    def tokenize_AwaitExpression(self, node):
        self.state.add("await", TokenTypes.KEYWORDS)
        
        with self.state.new_ctx("USE"):
            self.tokenize(node.argument)

    def tokenize_LogicalExpression(self, node):
        with self.state.new_ctx("USE"):
            self.tokenize_subexpression(node.left)
            self.state.add(node.operator, TokenTypes.BOP)
            self.tokenize_subexpression(node.right)


    def tokenize_ConditionalExpression(self, node):
        
        with self.state.new_ctx("USE"):
            self.tokenize(node.test)
        
        self.state.add("?")

        self.tokenize(node.consequent)

        self.state.add(":")
        
        self.tokenize(node.alternate)


    def tokenize_YieldExpression(self, node):
        self.state.add("yield", TokenTypes.KEYWORDS)

        with self.state.new_ctx("USE"):
            self.tokenize(node.argument)


    def tokenize_AssignmentExpression(self, node):
        
        with self.state.new_ctx("DEF"):
            self.tokenize(node.left)

        self.state.add(node.operator, TokenTypes.BOP)

        with self.state.new_ctx("USE"):
            self.tokenize(node.right)

    def tokenize_SequenceExpression(self, node):
        with self.brackets():
            self.tokenize_List(node.expressions)

    # Others -------------------------------------------------

    def tokenize_SwitchCase(self, node):
        
        if node.test is not None:
            self.state.add("case", TokenTypes.KEYWORDS)
            with self.state.new_ctx("USE"):
                self.tokenize(node.test)
        else:
            self.state.add("default", TokenTypes.KEYWORDS)

        self.state.add(":")

        for cons in node.consequent:
            self.tokenize(cons)

    def tokenize_CatchClause(self, node):
        self.state.add("catch", TokenTypes.KEYWORDS)
        
        with self.brackets():
            with self.state.new_ctx("DEF"):
                self.tokenize(node.param)
        
        self.tokenize(node.body)

    def tokenize_Property(self, node):
        
        with self.state.new_ctx("NAME"):
            self.tokenize(node.key)

        self.state.add(":")

        with self.state.new_ctx("USE"):
            self.tokenize(node.value)

    def tokenize_ObjectPattern(self, node):

        with self.brackets("{", "}"):
            self.tokenize_List(node.properties)

    def tokenize_ArrayPattern(self, node):
        with self.brackets("[", "]"):
            self.tokenize_List(node.elements)

    def tokenize_AssignmentPattern(self, node):
        with self.brackets():

            self.tokenize(node.left)
            self.state.add("=")
            self.tokenize(node.right)

    def tokenize_RestElement(self, node):
        self.state.add("...")
        self.tokenize(node.argument)

    def tokenize_SpreadElement(self, node):
        self.state.add("...")
        self.tokenize(node.argument)


    def tokenize_ClassExpression(self, node):
        self.state.add("class", TokenTypes.KEYWORDS)

        self.tokenize(node.id)

        if node.superClass is not None:
            self.state.add("extends", TokenTypes.KEYWORDS)
            self.tokenize(node.superClass)

        self.tokenize(node.body)


    def tokenize_Program(self, node):

        for element in node.body: 
            self.tokenize(element)

    def tokenize_ClassDeclaration(self, node):
        self.state.add("class", TokenTypes.KEYWORDS)

        with self.state.new_ctx("NAME"):
            self.tokenize(node.id)

        self.tokenize(node.body)

    def tokenize_ClassBody(self, node):
        with self.brackets("{", "}"):

            for method in node.body:
                self.tokenize(method)

    def tokenize_MethodDefinition(self, node):
        
        if node.kind != "method":
            self.state.add(node.kind, TokenTypes.KEYWORDS)

        if node.kind != "constructor":
            with self.state.new_ctx("FUNC_DEF"):
                self.tokenize(node.key)

        with self.state.new_ctx("INNER_FUNC"):
            self.tokenize(node.value)


def search(node, types):

    queue = [node]

    while len(queue) > 0:
        node = queue.pop(0)
        
        if node.type in types:
            yield node
            continue
            
        for child_key in dir(node):
            child = getattr(node, child_key)
            
            if hasattr(child, "kind"):
                queue.append(child)
            elif isinstance(child, list):
                queue.extend(child)


def try_parse_wrap(code):
    try:
        return esprima.parseScript(code)
    except esprima.Error:
        # Maybe the function is anonym
        code = "let x = %s" % code
        return esprima.parseScript(code)


def func_tokenize(code):
    
    tree = try_parse_wrap(code)

    method_defs = {}

    for node in search(tree, ["FunctionDeclaration", "MethodDefinition", "FunctionExpression"]):

        if node.type == "FunctionDeclaration":
            name = node.id.name
        if node.type == "MethodDefinition":
            if node.kind == "constructor":
                name = "constructor"
            else:
                name = node.key.name  
        if node.type == "FunctionExpression":
            name = "anonym"

        state = TokenizationState()
        NodeTokenizer(state)(node)

        method_defs[name] = {
            "tokens": state.tokens,
            "types" : state.type_ids
        }

    return method_defs


def full_tokenize(code):

    tree = esprima.parseScript(code)

    state = TokenizationState()
    NodeTokenizer(state)(tree)
    
    return {"tokens": state.tokens, "types": state.type_ids}