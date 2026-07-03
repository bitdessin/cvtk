import json
import numpy as np
import ast
import os
import re
import importlib.resources


class JsonComplexEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy data types.
    
    Extends json.JSONEncoder to serialize numpy integers, floats, arrays, and NaN values.
    Automatically converts these types to native Python types for JSON serialization.
    
    Examples:
        >>> import numpy as np
        >>> data = {'arr': np.array([1, 2, 3]), 'val': np.int64(42)}
        >>> json.dumps(data, cls=JsonComplexEncoder)
        '{"arr": [1, 2, 3], "val": 42}'
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj is np.nan:
            return None
        else:
            return super().default(obj)


def as_list(x):
    """Convert input to a list.
    
    Converts strings and dicts to single-element lists, lists/tuples to lists.
    
    Args:
        x (str|dict|list|tuple): Input to convert.
    
    Returns:
        list: Input converted to list format.
        
    Examples:
        >>> as_list('hello')
        ['hello']
        >>> as_list(['a', 'b'])
        ['a', 'b']
        >>> as_list(('x', 'y'))
        ['x', 'y']
    """
    if isinstance(x, (str, dict)):
        return [x]
    if isinstance(x, (list, tuple)):
        return list(x)
    raise TypeError(f'The input is expected to be a str, dict, list, or tuple, but got {type(x)}.')


def save_json(data, output, indent=None, ensure_ascii=False):
    """Save data to a JSON file with support for numpy types.
    
    Serializes and writes data to a JSON file using JsonComplexEncoder
    to handle numpy data types.
    
    Args:
        data: Data to serialize (dict, list, or any JSON-serializable type with numpy support).
        output (str): File path where JSON will be saved.
        indent (int|None): Indentation level for pretty-printing.
        ensure_ascii (bool): If False, allows non-ASCII characters (e.g., Unicode).
            Default is False.
    
    Examples:
        >>> data = {'values': [1, 2, 3], 'arr': np.array([4, 5, 6])}
        >>> save_json(data, 'output.json')
        >>> save_json(data, 'output_compact.json', indent=None)
    """
    with open(output, 'w', encoding='utf-8') as fp:
        json.dump(data,
                  fp,
                  indent=indent,
                  ensure_ascii=ensure_ascii,
                  cls=JsonComplexEncoder)


def expand_cvtk_sources(script_lines):
    """Expand cvtk imports in a script to include source definitions.
    
    Transforms a script that uses cvtk into a standalone script with embedded cvtk
    definitions, imports, and renamed references. This allows the script to run
    without the cvtk package installed.
        
    Args:
        script_lines (list[str]): Lines of the original script using cvtk.
    
    Returns:
        list[str]: Lines of the expanded script without cvtk imports.
    
    Examples:
        >>> script_lines = [
        ...     'import cvtk',
        ...     'from cvtk.ml import ClsRunner',
        ...     'runner = ClsRunner(datalabel, model)'
        ... ]
        >>> expanded = expand_cvtk_sources(script_lines)
        >>> with open('standalone.py', 'w') as f:
        ...     f.writelines(expanded)
    """
    source_map = _collect_import_path()
    
    ex_script_lines = []
    
    # find all cvtk function calls (e.g., cvtk.io.imread, cvtk.ml.split_dataset)
    cvtk_functions = set(re.findall(r'cvtk\.[\w\.]+', ''.join(script_lines)))
    
    # remove cvtk imports
    for line in script_lines:
        stripped = line.strip()        
        if stripped.startswith('import cvtk') or stripped.startswith('from cvtk'):
            continue
        ex_script_lines.append(line)
    
    # extract definitions for all cvtk functions
    script_cvtk = []
    source_files_used = set()  # track which source files are used for import collection
    
    # build short-name
    # e.g. {'DataLabel': 'cvtk_ml_data_DataLabel', 'ClsRunner': 'cvtk_ml_torchutils_ClsRunner'}
    short_name_to_renamed = {}
    for func_path_str in cvtk_functions:
        if func_path_str in source_map:
            short_name = func_path_str.split('.')[-1]
            renamed = _generate_cvtk_name(func_path_str)
            if short_name != renamed:  # only when actually renamed
                short_name_to_renamed[short_name] = renamed

    # build work queue: (source_file, func_name, renamed)
    work_queue = []
    for func_path_str in cvtk_functions:
        if func_path_str in source_map:
            source_file = source_map[func_path_str]
            func_name = func_path_str.split('.')[-1]
            renamed = _generate_cvtk_name(func_path_str)
            work_queue.append((source_file, func_name, renamed))
    
    # build a lookup
    rename_lookup = {(sf, fn): rn for sf, fn, rn in work_queue}
    
    # ordered list of (source_file, func_name, renamed) to maintain dep order
    ordered_extractions = []
    processed_names = set()
    def _collect_with_deps(source_file, func_name, renamed):
        key = (source_file, func_name)
        if key in processed_names:
            return
        processed_names.add(key)
        source_files_used.add(source_file)
        
        deps = _find_local_deps(source_file, func_name)
        for dep_name in deps:
            dep_key = (source_file, dep_name)
            if dep_key not in processed_names:
                dep_renamed = rename_lookup.get(dep_key, dep_name)
                _collect_with_deps(source_file, dep_name, dep_renamed)
        
        ordered_extractions.append((source_file, func_name, renamed))
    
    for source_file, func_name, renamed in work_queue:
        _collect_with_deps(source_file, func_name, renamed)
    
    for source_file, func_name, renamed in ordered_extractions:
        definition_lines = _extract_function_definition(source_file, func_name)
        if definition_lines:
            renamed_def_lines = _rename_function_definition(definition_lines, func_name, renamed)
            # replace internal references to short names (e.g., bare `DataLabel` -> `cvtk_ml_data_DataLabel`)
            renamed_def_lines = _replace_short_name_refs(renamed_def_lines, short_name_to_renamed, func_name)
            script_cvtk.extend(renamed_def_lines)
            script_cvtk.append('\n')
    
    # collect non-cvtk imports from source files used
    import_lines = _collect_imports(source_files_used)
    
    # replace function calls in main script (cvtk.io.imread -> cvtk_io_imread)
    script_main = []
    for line in ex_script_lines:
        modified_line = line
        
        # replace cvtk function calls
        for func_path_str in cvtk_functions:
            renamed = _generate_cvtk_name(func_path_str)
            modified_line = modified_line.replace(func_path_str, renamed)
        
        script_main.append(modified_line)
    
    final_script_lines = import_lines + ['\n'] + script_cvtk + ['\n\n'] + script_main
    final_script_lines = _del_docstring(final_script_lines)
    return final_script_lines


def _collect_import_path(cvtk_root=None):
    if cvtk_root is None:
        try:
            cvtk_root = str(importlib.resources.files('cvtk'))
        except:
            import cvtk
            cvtk_root = os.path.dirname(cvtk.__file__)
    
    cvtk_root = str(cvtk_root)
    source_map = {}
    
    for root, dirs, files in os.walk(cvtk_root):
        if '__init__.py' not in files:
            continue
        
        init_path = os.path.join(root, '__init__.py')
        
        rel_path = os.path.relpath(root, cvtk_root)
        if rel_path == '.':
            module_name = 'cvtk'
        else:
            module_name = 'cvtk.' + rel_path.replace(os.sep, '.')
        
        try:
            with open(init_path, 'r') as f:
                source_code = f.read()
                tree = ast.parse(source_code)
        except Exception as e:
            continue
        
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            
            if node.module is None and node.level > 0:
                import_module = ''
            else:
                import_module = node.module or ''
            
            if node.level > 0:
                if node.level == 1:
                    search_dir = root
                else:
                    search_dir = os.path.dirname(root)
                    for _ in range(node.level - 2):
                        search_dir = os.path.dirname(search_dir)
                
                if import_module:
                    module_file = _resolve_module_file(import_module, search_dir)
                else:
                    module_file = os.path.join(search_dir, '__init__.py')
            else:
                continue
            
            if module_file is None:
                continue
            
            for alias in node.names:
                imported_name = alias.name
                
                if imported_name == '*':
                    public_names = _extract_public_names(module_file)
                    for name in public_names:
                        full_path = f'{module_name}.{name}'
                        source_map[full_path] = os.path.abspath(module_file)
                else:
                    full_path = f'{module_name}.{imported_name}'
                    submodule_file = _resolve_module_file(imported_name, search_dir)
                    if submodule_file is not None:
                        public_names = _extract_public_names(submodule_file)
                        for name in public_names:
                            nested_path = f'{full_path}.{name}'
                            source_map[nested_path] = os.path.abspath(submodule_file)
                    else:
                        source_map[full_path] = os.path.abspath(module_file)
    
    return source_map


def _extract_public_names(file_path):
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
    except:
        return []
    names = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if not node.name.startswith('_'):
                names.append(node.name)    
    return names


def _resolve_module_file(module_name, package_dir):
    file_path = os.path.join(package_dir, f'{module_name}.py')
    if os.path.exists(file_path):
        return file_path
    package_path = os.path.join(package_dir, module_name, '__init__.py')
    if os.path.exists(package_path):
        return package_path
    return None


def _generate_cvtk_name(func_path):
    return func_path.replace('.', '_')


def _collect_imports(source_files):
    seen = set()
    result = []
    for source_file in sorted(source_files):
        try:
            with open(source_file, 'r') as f:
                source_code = f.read()
            tree = ast.parse(source_code)
        except:
            continue
        
        lines = source_code.splitlines(keepends=True)
        
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                continue
            
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith('cvtk'):
                        continue
                    line = lines[node.lineno - 1]
                    key = line.strip()
                    if key not in seen:
                        seen.add(key)
                        result.append(line)
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                if module.startswith('cvtk') or (node.level > 0):
                    continue
                
                line = lines[node.lineno - 1]
                key = line.strip()
                if key not in seen:
                    seen.add(key)
                    result.append(line)
            
            elif isinstance(node, ast.Assign):
                line = lines[node.lineno - 1]
                key = line.strip()
                if key not in seen and not key.startswith('#'):
                    seen.add(key)
                    result.append(line)
    
    return result


def _extract_function_definition(source_file, func_name):
    try:
        with open(source_file, 'r') as f:
            source_code = f.read()
            lines = source_code.split('\n')
        
        tree = ast.parse(source_code)
        
        target_node = None
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == func_name:
                target_node = node
                break
        
        if target_node is None:
            return []
        
        start_line = target_node.lineno - 1  # convert to 0-indexed
        end_line = target_node.end_lineno
        
        definition_lines = lines[start_line:end_line]
        return [line + '\n' for line in definition_lines]
    except:
        return []


def _find_local_deps(source_file, func_name):
    try:
        with open(source_file, 'r') as f:
            source_code = f.read()
        tree = ast.parse(source_code)
    except:
        return []

    top_level_defs = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            top_level_defs[node.name] = node

    target_node = top_level_defs.get(func_name)
    if target_node is None:
        return []

    referenced_names = set()
    for node in ast.walk(target_node):
        if isinstance(node, ast.Name):
            referenced_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            pass  # skip chained attributes

    if isinstance(target_node, ast.ClassDef):
        for base in target_node.bases:
            if isinstance(base, ast.Name):
                referenced_names.add(base.id)

    deps = []
    for name in referenced_names:
        if name != func_name and name in top_level_defs:
            deps.append(name)
    return deps


def _rename_function_definition(def_lines, old_name, new_name):
    result = []
    
    for line in def_lines:
        stripped = line.strip()
        
        # Replace function definition
        if stripped.startswith(f'def {old_name}('):
            line = line.replace(f'def {old_name}(', f'def {new_name}(', 1)
        # Replace class definition
        elif stripped.startswith(f'class {old_name}(') or stripped.startswith(f'class {old_name}:'):
            line = line.replace(f'class {old_name}', f'class {new_name}', 1)
        
        result.append(line)
    
    return result


def _replace_short_name_refs(def_lines, short_name_to_renamed, current_name):
    result = []
    for line in def_lines:
        modified_line = line
        for short_name, renamed in short_name_to_renamed.items():
            if short_name == current_name:
                continue
            # replace whole-word occurrences but NOT after a dot (e.g., torch.utils.data.Dataset should stay)
            pattern = rf'(?<!\.)\b{re.escape(short_name)}\b'
            modified_line = re.sub(pattern, renamed, modified_line)
        result.append(modified_line)
    return result

    
    
def _del_docstring(func_source: list[str]) -> list[str]:
    """Remove docstrings from source code.
    
    Strips triple-quoted docstrings from function/class definitions and main block,
    keeping only executable code and non-docstring comments.
    
    Args:
        func_source (list[str]): Source code lines.
        
    Returns:
        list[str]: Source lines without docstrings.
    """
    func_source_ = []
    is_docstring = False
    omit = False
    
    for line in func_source:
        stripped = line.strip()
        
        # Stop processing after main block
        if stripped.startswith('if __name__') and '__main__' in stripped:
            omit = True
        
        if omit:
            continue
        
        # Count triple-quote occurrences to handle multi-line and single-line docstrings
        # Each pair of quotes toggles docstring state; odd count = toggle, even = no change
        quote_count = stripped.count('"""') + stripped.count("'''")
        
        if quote_count > 0:
            # Odd occurrences: toggle docstring state
            if quote_count % 2 == 1:
                is_docstring = not is_docstring
            # Skip all delimiter lines
            continue
        
        # Only append non-docstring lines
        if not is_docstring:
            line = line.replace('\\\\', '\\')
            func_source_.append(line)
    
    return func_source_
    