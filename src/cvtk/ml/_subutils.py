import os
import shutil
import importlib
import random
import inspect
import re


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


def __generate_app_html_tmpl(tmpl_fpath, task):
    tmpl = []
    write_code = True
    with open(tmpl_fpath, 'r') as infh:
        for codeline in infh:
            if '#%CVTK%#' in codeline:
                if ' IF' in codeline:
                    m = re.search(r'TASK=([^\s\}]+)', codeline)
                    task_code = m.group(1) if m else None
                    if task_code is None:
                        raise ValueError('Unable to get task code.')
                    if task in task_code:
                        write_code = True
                    else:
                        write_code = False
                elif ' ENDIF' in codeline:
                    write_code = True
                continue

            if write_code:
                tmpl.append(codeline)
    return tmpl


def __estimate_task_from_source(source):
    task_ = {
        'CLSCORE': {'classdef': 0, 'import': 0, 'call': 0},
        'MMDETCORE': {'classdef': 0, 'import': 0, 'call': 0}
    }
    with open(source, 'r') as infh:
        for codeline in infh:
            if 'class CLSCORE' in codeline:
                task_['CLSCORE']['classdef'] += 1
            elif 'import cvtk.ml.torchuitls' in codeline:
                task_['CLSCORE']['import'] += 1
            elif 'from cvtk.ml.torchutils import' in codeline and 'CLSCORE' in codeline:
                task_['CLSCORE']['import'] += 1
            elif 'CLSCORE(' in codeline:
                task_['CLSCORE']['call'] += 1
            elif 'class MMDETCORE' in codeline:
                task_['MMDETCORE']['classdef'] += 1
            elif 'import cvtk.ml.mmdetutils' in codeline:
                task_['MMDETCORE']['import'] += 1
            elif 'from cvtk.ml.mmdetutils import' in codeline and 'MMDETCORE' in codeline:
                task_['MMDETCORE']['import'] += 1
            elif 'MMDETCORE(' in codeline:
                task_['MMDETCORE']['call'] += 1
    is_cls_cvtk = (task_['CLSCORE']['import'] > 0) and (task_['CLSCORE']['call'] > 0)
    is_cls_vanilla = (task_['CLSCORE']['classdef'] > 0) and (task_['CLSCORE']['call'] > 0)
    is_cls = is_cls_cvtk or is_cls_vanilla
    is_det_cvtk = (task_['MMDETCORE']['import'] > 0) and (task_['MMDETCORE']['call'] > 0)
    is_det_vanilla = (task_['MMDETCORE']['classdef'] > 0) and (task_['MMDETCORE']['call'] > 0)
    is_det = is_det_cvtk or is_det_vanilla

    if is_cls and (not is_det):
        task = 'cls'
    elif (not is_cls) and is_det:
        task = 'det'
    else:
        raise ValueError('The task type cannot be determined from the source code. Make sure your source code contains CLSCORE or MMDETCORE class definition or importation, and call.')
    if is_cls_cvtk or is_det_cvtk:
        is_vanilla = False
    elif is_cls_vanilla or is_det_vanilla:
        is_vanilla = True
    else:
        raise ValueError('The source code cannot be determined from the source code. Make sure your source code contains importation of cvtk.ml.torchutils or cvtk.ml.mmdetutils.')

    return task, is_vanilla
