import inspect
from typing import get_type_hints, get_origin, get_args, Annotated, Literal


def extract_metadata(func):
    """
    Extract metadata from a callable including its name, docstring, parameters,
    and return annotation.

    Parameters:
        func (callable): The function to extract metadata from.

    Returns:
        dict: A dictionary containing the metadata.
    """
    func_name = func.__name__
    docstring = inspect.getdoc(func)
    signature = inspect.signature(func)
    type_hints = get_type_hints(func, include_extras=True)
    parameters = {}
    for param_name, param in signature.parameters.items():
        param_annotation = type_hints.get(param_name, param.annotation)
        param_info = _parse_annotation(param_annotation)
        param_info.update({
            'default': None if param.default is inspect.Parameter.empty else param.default,  # noqa
            'required': param.default is inspect.Parameter.empty
        })
        parameters[param_name] = param_info
    return_annotation = type_hints.get('return', signature.return_annotation)
    return_info = _parse_annotation(return_annotation)
    return {
        'name': func_name,
        'description': docstring,
        'arguments': parameters,
        'return': return_info
    }


def _parse_annotation(annotation):
    """Helper function to parse an annotation."""
    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        base_type = args[0]
        metadata = args[1] if len(args) > 1 else ''
        annotation = _parse_annotation(base_type)
        annotation["description"] = metadata
        return annotation
    elif origin is Literal:
        return {
            'type': 'Literal',
            'values': list(get_args(annotation))
        }
    elif origin in (tuple, list, set, frozenset):
        return {
            'type': _get_annotation_name(origin),
            'elements': [_parse_annotation(arg) for arg in get_args(annotation)]
        }
    elif origin is dict:
        key_type, value_type = get_args(annotation)
        return {
            'type': 'dict',
            'key_type': _parse_annotation(key_type),
            'value_type': _parse_annotation(value_type)
        }
    return {'type': _get_annotation_name(annotation)}


def _get_annotation_name(annotation):
    """Helper function to get the name of an annotation."""
    if hasattr(annotation, '__name__'):
        return annotation.__name__
    elif hasattr(annotation, '_name'):
        return annotation._name
    return str(annotation)

