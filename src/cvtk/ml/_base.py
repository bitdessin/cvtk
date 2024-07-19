import inspect

def __generate_cvtk_imports_string(modules, import_from='cvtk'):
    func_imports = ''

    for lib_dicts in modules:
        for lib_name, lib_funcs in lib_dicts.items():
            for lib_func in lib_funcs:
                if import_from.lower() == 'cvtk':
                    func_imports += f'from {lib_name} import {lib_func.__name__}\n'
                else:
                    func_imports += '\n\n\n' + inspect.getsource(lib_func)

    func_imports += '\n\n\n'
    return func_imports


def __del_docstring(func_source):
    func_source_ = ''
    is_docstring = False
    for line in func_source.split('\n'):
        if line.strip().startswith('"""') or line.strip().startswith("'''"):
            is_docstring = not is_docstring
        else:
            if not is_docstring:
                func_source_ += line + '\n'
    return func_source_

