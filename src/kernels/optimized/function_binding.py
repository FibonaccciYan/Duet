"""Clone function globals without mutating other runtimes."""
import types

def _bind_globals(function, overrides):
    """Bind execution dependencies per model, without patching original modules."""
    namespace = {**function.__globals__, **overrides}
    clone = types.FunctionType(function.__code__, namespace, function.__name__,
                               function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    return clone, namespace
