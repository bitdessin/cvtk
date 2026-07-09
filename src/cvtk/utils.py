import json
import numpy as np
import ast
import ast
import os
from pathlib import Path
import ast
import importlib.resources
import os
from pathlib import Path


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



DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def expand_cvtk_sources(script_lines, cvtk_root=None):
    source_map = _collect_import_path(cvtk_root)

    main_code, used_paths = _rewrite_cvtk_refs(
        "".join(script_lines),
        source_map,
        allow_import_cvtk=True,
        source_file=None,
    )

    definitions, source_files = _collect_required_defs(used_paths, source_map)

    output = []
    output.extend(_collect_imports(source_files))
    output.append("\n")

    for source_file, name in definitions:
        code = _extract_top_level_def(source_file, name)
        code, _ = _rewrite_cvtk_refs(
            code,
            source_map,
            allow_import_cvtk=False,
            source_file=source_file,
        )
        output.extend(_to_lines(code))
        output.append("\n")

    output.append("\n")
    output.extend(_to_lines(main_code))
    return output


class _CvtkRefRewriter(ast.NodeTransformer):
    def __init__(self, source_map, allow_import_cvtk, source_file=None):
        self.source_map = source_map
        self.allow_import_cvtk = allow_import_cvtk
        self.source_file = source_file
        self.used_paths = set()
        self.import_aliases = _collect_module_import_aliases(source_file)

    def visit_Import(self, node):
        kept = []

        for alias in node.names:
            if alias.name == "cvtk" and alias.asname is None:
                if self.allow_import_cvtk:
                    continue
                raise ValueError("cvtk source definitions must not import cvtk.")

            if alias.name.startswith("cvtk"):
                raise ValueError(
                    f"Unsupported cvtk import: import {alias.name}. "
                    "Use only `import cvtk` and `cvtk.xxx.yyy(...)`."
                )

            kept.append(alias)

        if not kept:
            return None

        node.names = kept
        return node

    def visit_ImportFrom(self, node):
        module = node.module or ""

        if module == "cvtk" or module.startswith("cvtk."):
            raise ValueError(
                f"Unsupported cvtk import: from {module} import ... "
                "Use `import cvtk` and fully qualified `cvtk.xxx.yyy(...)`."
            )

        if node.level > 0:
            for alias in node.names:
                if alias.name == "*":
                    continue

                resolved_file = _resolve_relative_import_file(
                    self.source_file,
                    node,
                    alias.name,
                )
                if resolved_file is not None:
                    self.import_aliases[alias.asname or alias.name] = resolved_file
            return None

        return node

    def visit_Attribute(self, node):
        full_path = _attribute_to_dotted_name(node)
        mapped_path, tail = _longest_source_map_prefix(full_path, self.source_map)

        if mapped_path is not None:
            self.used_paths.add(mapped_path)
            return ast.copy_location(
                _build_replacement(mapped_path.rsplit(".", 1)[-1], tail, node.ctx),
                node,
            )

        if isinstance(node.value, ast.Name):
            imported_file = self.import_aliases.get(node.value.id)
            if imported_file is not None:
                mapped_path = _resolve_imported_attr_path(imported_file, node.attr, self.source_map)
                if mapped_path is not None:
                    self.used_paths.add(mapped_path)
                    return ast.copy_location(
                        ast.Name(id=node.attr, ctx=node.ctx),
                        node,
                    )

        alias_name, tail = _alias_attribute_parts(node, self.import_aliases)
        if alias_name is not None:
            mapped_path, mapped_tail = _resolve_imported_attribute_path(
                self.import_aliases[alias_name],
                tail,
                self.source_map,
            )
            if mapped_path is not None:
                self.used_paths.add(mapped_path)
                return ast.copy_location(
                    _build_replacement(mapped_path.rsplit(".", 1)[-1], mapped_tail, node.ctx),
                    node,
                )

        return self.generic_visit(node)


def _rewrite_cvtk_refs(code, source_map, allow_import_cvtk, source_file=None):
    tree = ast.parse(code)
    rewriter = _CvtkRefRewriter(source_map, allow_import_cvtk, source_file=source_file)
    tree = rewriter.visit(tree)
    ast.fix_missing_locations(tree)

    return ast.unparse(tree) + "\n", rewriter.used_paths


def _attribute_to_dotted_name(node):
    parts = []
    cur = node

    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value

    if not isinstance(cur, ast.Name):
        return None

    return ".".join([cur.id, *reversed(parts)])


def _longest_source_map_prefix(full_path, source_map):
    if full_path is None:
        return None, []

    parts = full_path.split(".")

    for i in range(len(parts), 1, -1):
        candidate = ".".join(parts[:i])
        if candidate in source_map:
            return candidate, parts[i:]

    return None, []


def _build_replacement(name, tail, ctx):
    node = ast.Name(id=name, ctx=ast.Load())

    for attr in tail:
        node = ast.Attribute(value=node, attr=attr, ctx=ast.Load())

    node.ctx = ctx
    return node


def _alias_attribute_parts(node, import_aliases):
    parts = []
    cur = node

    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value

    if isinstance(cur, ast.Name) and cur.id in import_aliases:
        return cur.id, list(reversed(parts))

    return None, []


def _resolve_relative_import_file(source_file, node, member_name):
    if source_file is None:
        return None

    package_dir = Path(source_file).resolve().parent
    base_dir = package_dir

    for _ in range(node.level - 1):
        base_dir = base_dir.parent

    if node.module:
        module_path = base_dir.joinpath(*node.module.split("."))
    else:
        module_path = base_dir

    candidates = [module_path.with_suffix(".py"), module_path / "__init__.py"]
    if node.module is None and member_name:
        candidates = [
            base_dir.joinpath(member_name).with_suffix(".py"),
            base_dir.joinpath(member_name, "__init__.py"),
            *candidates,
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def _has_top_level_def(source_file, name):
    return name in _collect_top_level_defs(source_file)


def _collect_top_level_defs(source_file):
    tree, _ = _read_module(source_file)
    return set(_top_level_defs(tree))


def _resolve_imported_attr_path(source_file, attr_name, source_map):
    mapped_path, _ = _resolve_imported_attribute_path(source_file, [attr_name], source_map)
    if mapped_path is not None:
        return mapped_path

    return None


def _resolve_imported_attribute_path(source_file, attr_parts, source_map):
    if not attr_parts:
        return None, []

    for cvtk_path, mapped_file in source_map.items():
        if (
            str(Path(mapped_file).resolve()) == str(Path(source_file).resolve())
            and cvtk_path.rsplit(".", 1)[-1] == attr_parts[0]
        ):
            return cvtk_path, attr_parts[1:]

    if Path(source_file).name == "__init__.py" and _package_exports_name(source_file, attr_parts[0]):
        suffix = ".".join(attr_parts)
        matches = [cvtk_path for cvtk_path in source_map if cvtk_path.endswith(f".{suffix}")]
        if len(matches) == 1:
            return matches[0], []

        matches = [cvtk_path for cvtk_path in source_map if cvtk_path.rsplit(".", 1)[-1] == attr_parts[0]]
        if len(matches) == 1:
            return matches[0], attr_parts[1:]

    return None, []


def _package_exports_name(source_file, attr_name):
    tree, _ = _read_module(source_file)
    package_dir = Path(source_file).resolve().parent

    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.level == 0:
            continue

        base_dir = _relative_import_base(package_dir, node.level)
        module_file = _resolve_module_file(node.module or "", base_dir)
        if module_file is None:
            continue

        for alias in node.names:
            if alias.name == "*":
                if attr_name in _extract_public_names(module_file):
                    return True
            elif (alias.asname or alias.name) == attr_name:
                return True

    return False


def _collect_module_import_aliases(source_file):
    if source_file is None:
        return {}

    tree, _ = _read_module(source_file)
    aliases = {}

    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue

                resolved_file = _resolve_relative_import_file(source_file, node, alias.name)
                if resolved_file is not None:
                    aliases[alias.asname or alias.name] = resolved_file
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is None:
                    continue

                if alias.name == "cvtk" or alias.name.startswith("cvtk."):
                    continue

                resolved_file = _resolve_import_module_file(source_file, alias.name)
                if resolved_file is not None:
                    aliases[alias.asname] = resolved_file

    return aliases


def _resolve_import_module_file(source_file, module_name):
    source_path = Path(source_file).resolve().parent
    module_path = source_path.joinpath(*module_name.split("."))

    for candidate in (module_path.with_suffix(".py"), module_path / "__init__.py"):
        if candidate.exists():
            return candidate

    return None


def _collect_required_defs(initial_paths, source_map):
    ordered = []
    seen = set()
    visiting = set()
    source_files = set()
    name_owners = {}

    def add_by_path(cvtk_path):
        source_file = os.path.abspath(source_map[cvtk_path])
        name = cvtk_path.rsplit(".", 1)[-1]
        add_by_name(source_file, name)

    def add_by_name(source_file, name):
        key = (os.path.abspath(source_file), name)

        if key in seen:
            return

        if key in visiting:
            return

        owner = name_owners.get(name)
        if owner is not None and owner != key[0]:
            raise ValueError(
                f"Name conflict while expanding cvtk: `{name}` exists in both "
                f"{owner} and {key[0]}. Use unique exported names or add renaming."
            )

        name_owners[name] = key[0]
        visiting.add(key)
        source_files.add(key[0])

        for dep_name in _find_local_deps(key[0], name):
            add_by_name(key[0], dep_name)

        for dep_path in _find_cvtk_deps_in_def(key[0], name, source_map):
            add_by_path(dep_path)

        visiting.remove(key)
        seen.add(key)
        ordered.append(key)

    for cvtk_path in sorted(initial_paths):
        add_by_path(cvtk_path)

    return ordered, source_files

def _find_local_deps(source_file, name):
    tree, _ = _read_module(source_file)
    top_defs = _top_level_defs(tree)

    target = top_defs.get(name)
    if target is None:
        return []

    refs = {
        node.id
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }

    return sorted(ref for ref in refs if ref != name and ref in top_defs)


def _find_cvtk_deps_in_def(source_file, name, source_map):
    tree, _ = _read_module(source_file)
    target = _top_level_defs(tree).get(name)

    if target is None:
        return []

    collector = _CvtkDepCollector(source_map, source_file)
    collector.visit(target)
    return sorted(collector.paths)


class _CvtkDepCollector(ast.NodeVisitor):
    def __init__(self, source_map, source_file=None):
        self.source_map = source_map
        self.import_aliases = _collect_module_import_aliases(source_file)
        self.paths = set()

    def visit_Attribute(self, node):
        self.generic_visit(node)

        full_path = _attribute_to_dotted_name(node)
        mapped_path, _ = _longest_source_map_prefix(full_path, self.source_map)

        if mapped_path is not None:
            self.paths.add(mapped_path)
            return

        alias_name, tail = _alias_attribute_parts(node, self.import_aliases)
        if alias_name is not None:
            mapped_path, _ = _resolve_imported_attribute_path(
                self.import_aliases[alias_name],
                tail,
                self.source_map,
            )
            if mapped_path is not None:
                self.paths.add(mapped_path)
        
        
def _read_module(source_file):
    source_file = os.path.abspath(source_file)
    code = Path(source_file).read_text(encoding="utf-8")
    return ast.parse(code), code


def _top_level_defs(tree):
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, DEF_NODES)
    }


def _extract_top_level_def(source_file, name):
    tree, code = _read_module(source_file)
    node = _top_level_defs(tree).get(name)

    if node is None:
        raise ValueError(f"Definition `{name}` was not found in {source_file}")

    lines = code.splitlines(keepends=True)

    if getattr(node, "decorator_list", None):
        start = node.decorator_list[0].lineno - 1
    else:
        start = node.lineno - 1

    end = node.end_lineno
    return "".join(lines[start:end])


def _collect_imports(source_files):
    seen = set()
    result = []

    for source_file in sorted(source_files):
        tree, code = _read_module(source_file)
        lines = code.splitlines(keepends=True)

        for node in tree.body:
            if isinstance(node, DEF_NODES):
                continue

            if isinstance(node, ast.Import):
                line = lines[node.lineno - 1]
                names = [alias.name for alias in node.names]

                if any(name == "cvtk" or name.startswith("cvtk.") for name in names):
                    continue

                _append_once(result, seen, line)

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""

                if node.level > 0:
                    continue

                if module == "cvtk" or module.startswith("cvtk."):
                    continue

                _append_once(result, seen, lines[node.lineno - 1])

            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for line in lines[node.lineno - 1:node.end_lineno]:
                    _append_once(result, seen, line)

    return result


def _append_once(result, seen, line):
    key = line.strip()

    if key and key not in seen:
        seen.add(key)
        result.append(line)


def _to_lines(code):
    return [line + "\n" for line in code.splitlines()]


def _collect_import_path(cvtk_root=None):
    if cvtk_root is None:
        cvtk_root = importlib.resources.files("cvtk")

    cvtk_root = Path(cvtk_root)
    source_map = {}

    for init_path in cvtk_root.rglob("__init__.py"):
        package_dir = init_path.parent
        rel = package_dir.relative_to(cvtk_root)

        package_name = "cvtk"
        if str(rel) != ".":
            package_name += "." + ".".join(rel.parts)

        tree = ast.parse(init_path.read_text(encoding="utf-8"))
        
        # Collect import statements from all contexts (including try/except blocks)
        import_nodes = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                import_nodes.append(node)
            elif isinstance(node, ast.Try):
                # Also collect imports from try blocks
                for stmt in node.body:
                    if isinstance(stmt, ast.ImportFrom):
                        import_nodes.append(stmt)

        for node in import_nodes:
            if node.level == 0:
                continue

            base_dir = _relative_import_base(package_dir, node.level)
            module_file = _resolve_module_file(node.module or "", base_dir)

            if module_file is None:
                continue

            for alias in node.names:
                if alias.name == "*":
                    for public_name in _extract_public_names(module_file):
                        source_map[f"{package_name}.{public_name}"] = str(module_file)
                    continue

                full_path = f"{package_name}.{alias.name}"
                submodule_file = _resolve_module_file(alias.name, base_dir)

                if submodule_file is None:
                    source_map[full_path] = str(module_file)
                    continue

                for public_name in _extract_public_names(submodule_file):
                    source_map[f"{full_path}.{public_name}"] = str(submodule_file)

    return source_map


def _relative_import_base(package_dir, level):
    base = Path(package_dir)

    for _ in range(level - 1):
        base = base.parent

    return base


def _resolve_module_file(module_name, package_dir):
    package_dir = Path(package_dir)

    if module_name:
        module_path = package_dir.joinpath(*module_name.split("."))
    else:
        module_path = package_dir

    file_path = module_path.with_suffix(".py")
    if file_path.exists():
        return file_path

    init_path = module_path / "__init__.py"
    if init_path.exists():
        return init_path

    return None


def _extract_public_names(file_path):
    tree = ast.parse(Path(file_path).read_text(encoding="utf-8"))

    return [
        node.name
        for node in tree.body
        if isinstance(node, DEF_NODES) and not node.name.startswith("_")
    ]
