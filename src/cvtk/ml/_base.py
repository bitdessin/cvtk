import inspect


def __get_imports(code_file: str) -> list[str]:
    """Find lines containing import statements from a file.
    
    Args:
        code_file (str): Path to a python file.
    """
    imports = []

    with open(code_file, 'r') as codefh:
        for codeline in codefh:
            if codeline[0:6] == 'import':
                imports.append(codeline)
    return imports


def __insert_imports(tmpl: list[str], modules: list[str]) -> list[str]:
    """Insert import statements to a template.
    
    Insert import statements to a template (`tmpl`) at the end of the import statements in the template.

    Args:
        tmpl (list[str]): A list of strings containing the template.
        modules (list[str]): A list of strings containing import statements.
    
    Examples:
        >>> tmpl = ['import os',
        ...         '',
        ...         'print("Hello, World!")'],
        >>> modules = ['import cvtk']
        >>> __insert_imports(tmpl, modules)
        ['import os',
        'import cvtk',
        '',
        'print("Hello, World!")']    
        >>> 
        >>> 
        >>> tmpl = __insert_imports(tmpl, __get_imports(__file__)
        >>> 
    """
    extmpl = []
    imported = False
    for codeline in tmpl:
        if codeline[0:6] == 'import':
            # pass the imports in original file
            pass
        else:
            if not imported:
                # insert the imports
                for mod in modules:
                    extmpl.append(mod)
                imported = True
            extmpl.append(codeline)
    return extmpl


def __extend_cvtk_imports(tmpl, module_dicts):
    extmpl = []

    extended = False
    for codeline in tmpl:
        if codeline[0:9] == 'from cvtk':
            if not extended:
                for mod_dict in module_dicts:
                    for mod_name, mod_funcs in mod_dict.items():
                        for mod_func in mod_funcs:
                            extmpl.append('\n\n\n' + inspect.getsource(mod_func))
                extended = True
        else:
            extmpl.append(codeline)
    
    return extmpl


def __del_docstring(func_source: str) -> str:
    """Delete docstring from source code.

    Delete docstring (strings between \"\"\" or ''' ) from source code.

    Args:
        func_source (str): Source code of a function.
    """
    func_source_ = ''
    is_docstring = False
    omit = False
    for line in func_source.split('\n'):
        if line.startswith('if __name__ == \'__main__\':'):
            omit = True
        if (line.strip().startswith('"""') or line.strip().startswith("'''")) and (not omit):
            is_docstring = not is_docstring
        else:
            if not is_docstring:
                func_source_ += line + '\n'
    return func_source_

