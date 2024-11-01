import os
import shutil
import importlib
import random
import inspect
import re


def __get_imports(code_file: str) -> list[str]:
    imports = []
    with open(code_file, 'r') as codefh:
        for codeline in codefh:
            if codeline[0:6] == 'import':
                imports.append(codeline)
            if codeline[0:6] == 'class ' or codeline[0:4] == 'def ':
                break
    return imports


def __insert_imports(tmpl: list[str], modules: list[str]) -> list[str]:
    """Insert imports into the top of the template code.

    This function deletes the original imports in the template (`tmpl`)
    and then inserts modules listed in `modules` argument.
    """
    extmpl = []
    imported = False
    for codeline in tmpl:
        if codeline[0:6] == 'import':
            pass # delete the original imports
        else:
            if not imported:
                # insert the modules listed in `modules` argument
                for mod in modules:
                    extmpl.append(mod)
                imported = True
            # append the original code after the imports
            extmpl.append(codeline)
    return extmpl


def __extend_cvtk_imports(tmpl, module_dicts):
    extmpl = []
    extended = False
    for codeline in tmpl:
        if codeline[0:9] == 'from cvtk':
            # find the first cvtk import statement and replace it with the source code of the modules
            if not extended:
                for mod_dict in module_dicts:
                    for mod_name, mod_funcs in mod_dict.items():
                        for mod_func in mod_funcs:
                            extmpl.append('\n\n\n' + inspect.getsource(mod_func))
                extended = True
        else:
            # append the original code after the extending of cvtk module source code
            extmpl.append(codeline)
    return extmpl


def __del_docstring(func_source: str) -> str:
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
                line = line.replace('\\\\', '\\')
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


def __estimate_source_task(source):
    spec = importlib.util.spec_from_file_location('ModuleCore', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ModuleCoreClass = getattr(module, 'ModuleCore', None)
    if ModuleCoreClass is None:
        raise AttributeError("ModuleCore class not found in the specified source file.")
    module_instance = ModuleCoreClass(None, None)
    return module_instance.task_type


def __estimate_source_vanilla(source):
    is_vanilla = True
    with open(source, 'r') as infh:
        for codeline in infh:
            if ('import cvtk' in codeline) or ('from cvtk' in codeline):
                is_vanilla = False
                break
    return is_vanilla
