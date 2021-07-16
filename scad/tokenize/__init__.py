from .pytokenizer import func_tokenize as pyfn_tokenize
from .jvtokenizer import method_tokenize as jvfn_tokenize
from .jstokenizer import func_tokenize as jsfn_tokenize


def func_tokenize(code, lang = "python"):

    if lang == "python"    : return pyfn_tokenize(code)
    if lang == "java"      : return jvfn_tokenize(code)
    if lang == "javascript": return jsfn_tokenize(code)

    raise ValueError("Unknown language: %s" % lang)