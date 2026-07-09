import os
import importlib
import shutil
import re

from .. import utils as cvtk_utils


def runner(
    script_name: str,
    backend: str='torch',
    task: str='cls',
    vanilla: bool=False
) -> None:
    """Generate source code for training and inference with PyTorch or MMDetection.

    Creates a Python script template for machine learning tasks (classification, object detection,
    instance segmentation) with either PyTorch, MMDetection, or OneDL-MMDetection framework. The generated script can
    include cvtk dependencies (simple) or be fully standalone with embedded cvtk code (vanilla).

    Two script types:
    - vanilla=False (default): Compact script that imports from cvtk package.
      Suitable for learning and prototyping. Requires cvtk installed.
    - vanilla=True: Standalone script with all cvtk function definitions embedded.
      Larger file but no external cvtk dependency. Allows full customization and modification.

    Args:
        script_name (str): Output file path to save the generated script.
            '.py' extension added if not present.
        backend (str): ML framework. Options: 'torch' (PyTorch), 'mmdet'/'mmdetection' (MMDetection), or 'onedl' (OneDL-MMDetection).
            Default 'torch'.
        task (str): Machine learning task. Options:
            - 'cls'/'classification': Image classification with PyTorch
            - 'det'/'detection': Object detection with MMDetection
            - 'seg'/'segmentation': Instance segmentation with MMDetection
            Default 'cls'.
        vanilla (bool): Generate standalone script with embedded cvtk code (True) or
            compact script with cvtk imports (False). Default False.

    Returns:
        None. Script file is created at script_name path.

    Raises:
        ValueError: If task or backend not supported.
        FileNotFoundError: If template file not found.

    Examples:
        >>> from cvtk.ml import runner
        >>> # Generate classification script with cvtk imports
        >>> runner('my_classifier.py', backend='torch', task='cls')
        >>> # Generate standalone detection script
        >>> runner('detector_standalone.py', backend='mmdet', task='det', vanilla=True)
        >>> # Generate segmentation script
        >>> runner('segmenter.py', backend='mmdet', task='seg', vanilla=False)
    """
    task = 'cls' if task[:3] == 'cla' else task[:3]
    if task not in ['cls', 'cla', 'det', 'seg']:
        raise ValueError('The current version only support classification (`cls`), detection (`det`), and segmentation (`seg`/`segm`/`segmentation`) tasks.')
    
    backend = 'mmdet' if backend[:5] == 'onedl' else backend[:5]    
    if backend not in ['torch', 'mmdet']:
        raise ValueError('The current version only support PyTorch (`torch`), MMDetection (`mmdet`), and OneDL-MMDetection (`onedl`) backends.')
    else:
        backend = backend[:5]
    
    if not script_name.endswith('.py'):
        script_name += '.py'
    
    with importlib.resources.files('cvtk').joinpath(f'tmpl/_{backend}_{task}.py').open('r') as infh:
        tmpl_lines = infh.readlines()
    
    # replace script name
    tmpl_lines = [line.replace('__SCRIPTNAME__', os.path.basename(script_name)) for line in tmpl_lines]
    
    # extract and embed cvtk function definitions
    if vanilla:
        tmpl_lines = cvtk_utils.expand_cvtk_sources(tmpl_lines)
    
    # write output        
    with open(script_name, 'w') as fh:
        fh.writelines(tmpl_lines)    


_module_dict = {
    'torch-cls': {'import': 'cvtk.ml.torchapi', 'class': 'ClsRunner'},
    'torch-det': {'import': 'cvtk.ml.torchdetapi', 'class': 'DetRunner'}, 
    'torch-seg': {'import': 'cvtk.ml.torchdetapi', 'class': 'SegmRunner'},
    'mmdet-det': {'import': 'cvtk.ml.mmdetapi', 'class': 'DetRunner'},
    'mmdet-seg': {'import': 'cvtk.ml.mmdetapi', 'class': 'SegmRunner'},
}


def demoapp(
    app_name: str,
    runner_script: str,
    weights: str,
    label: str
) -> None:
    """Generate a Flask demo application for model inference.

    Creates a self-contained demo project that serves an image upload UI and inference API.
    The function copies source/model artifacts into the target project and generates
    ``main.py`` and ``templates/index.html`` from cvtk templates. Task type (cls/det/segm)
    and backend (torch/mmdet) are inferred from the source script to choose the proper
    runner class.

    For MMDetection backend, the model configuration path is inferred from ``weights``
    by replacing the extension with ``.py`` (e.g., ``model.pth`` -> ``model.py``).
    For torch backend, model architecture is inferred from ``runner_script``.

    Args:
        app_name (str): A path to a directory where the demo app will be created.
        runner_script (str): Path to source script used to infer task/backend.
        weights (str): Path to model weight file (.pth).
        label (str): Path to data label file.

    Returns:
        None: Files are generated under ``app_name``.

    Examples:
        >>> from cvtk.ml.deploy import demoapp
        >>> demoapp('./demoapp', './det.py', './outputs/strawberry.pth', './data/strawberry/label.txt')
    """
    if not os.path.exists(app_name):
        os.makedirs(app_name)

    # infer task/backend from source script
    task_type, backend, is_vanilla = __parse_runner_script(runner_script)

    if task_type is None:
        raise ValueError(f'Unable to infer task type from runner script: {runner_script}')
    if backend is None:
        raise ValueError(f'Unable to infer backend from runner script: {runner_script}')

    # Normalize task_type for dictionary lookup: 'segm' -> 'seg'
    task_key = 'seg' if task_type == 'segm' else task_type

    # backend-specific model argument for runner constructors in template
    model_cfg = os.path.splitext(weights)[0] + '.py'
    if backend == 'mmdet':
        if not os.path.exists(model_cfg):
            raise FileNotFoundError(f'MMDetection config file is required but not found: {model_cfg}')
        model_arg = 'model.cfg.py'
    else:
        model_arg = __infer_torch_model_name(runner_script)
        if model_arg is None:
            raise ValueError(
                'Unable to infer torch model architecture from runner script. '
                'Expected patterns like "model_name=..." or runner constructor with a model name.'
            )
    
    # Determine module import path and class based on vanilla flag
    if is_vanilla:
        module_import = 'model'
    else:
        module_import = _module_dict[f'{backend}-{task_key}']['import']
    module_class = _module_dict[f'{backend}-{task_key}']['class']
    
    # copy template files to app directory
    shutil.copy(runner_script, os.path.join(app_name, 'model.py'))
    shutil.copy(label, os.path.join(app_name, 'model.label.txt'))
    shutil.copy(weights, os.path.join(app_name, 'model.pth'))
    # For torch backend, also save .dl.txt (datalabel) file for proper model loading
    if backend == 'torch':
        shutil.copy(label, os.path.join(app_name, 'model.dl.txt'))
    if backend == 'mmdet':
        shutil.copy(model_cfg, os.path.join(app_name, 'model.cfg.py'))
    
    # generate flask app (main.py) from template
    tmpl = __generate_app_html_tmpl(
        importlib.resources.files('cvtk').joinpath('tmpl/_flask.py'),
        task_type,
        backend,
        model_arg,
        module_import,
        module_class
    )
    with open(os.path.join(app_name, 'main.py'), 'w') as fh:
        fh.write(tmpl)

    # HTML template
    if not os.path.exists(os.path.join(app_name, 'templates')):
        os.makedirs(os.path.join(app_name, 'templates'))
    tmpl = __generate_app_html_tmpl(
        importlib.resources.files('cvtk').joinpath(f'tmpl/html/_flask.html'),
        task_type,
        backend,
        model_arg,
        module_import,
        module_class
    )
    with open(os.path.join(app_name, 'templates', 'index.html'), 'w') as fh:
        fh.write(''.join(tmpl))


def __parse_runner_script(runner_script):
    task_type = None
    backend = None
    is_vanilla = True
    
    with open(runner_script, 'r') as infh:
        full_content = infh.read()
        content_lower = full_content.lower()
    
    # Parse task type (look for Runner class names)
    if 'segmrunner' in content_lower:
        task_type = 'segm'
    elif 'clsrunner' in content_lower:
        task_type = 'cls'
    elif 'detrunner' in content_lower:
        task_type = 'det'
    
    # Detect if vanilla.
    # A script is considered non-vanilla when it actually imports cvtk or
    # references cvtk modules in executable code, not merely when docstrings or
    # comments mention the package.
    if __contains_cvtk_import(full_content):
        is_vanilla = False
    
    # Detect backend (prioritize mmdet patterns over torch imports)
    # Check for explicit cvtk imports first
    if 'cvtk.ml.mmdetapi' in content_lower or 'mmdetapi' in content_lower:
        backend = 'mmdet'
    elif 'cvtk.ml.torchdetapi' in content_lower or 'cvtk.ml.torchapi' in content_lower or 'torchdetapi' in content_lower or 'torchapi' in content_lower:
        backend = 'torch'
    # Check for mmdet-specific patterns (these indicate mmdet backend)
    elif any(pattern in content_lower for pattern in ['faster-rcnn', 'mask-rcnn', 'datapipeline', 'datafactory']):
        backend = 'mmdet'
    # Fall back to torch if import torch is found
    elif 'import torch' in content_lower:
        backend = 'torch'
    
    # Default values if not detected
    if task_type is None:
        task_type = 'det'
    
    if backend is None:
        if task_type == 'cls':
            backend = 'torch'  # classification is only supported in torch
        else:
            backend = 'mmdet'  # detection/segmentation default to mmdet

    return task_type, backend, is_vanilla


parse_runner_script = __parse_runner_script


def __getattr__(name):
    if name.endswith('__parse_runner_script'):
        return __parse_runner_script
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __contains_cvtk_import(content):
    try:
        tree = compile(content, '<deploy-script>', 'exec')
    except SyntaxError:
        return False

    import ast

    class CvtkImportVisitor(ast.NodeVisitor):
        def visit_Import(self, node):
            for alias in node.names:
                if alias.name == 'cvtk' or alias.name.startswith('cvtk.'):
                    raise StopIteration

        def visit_ImportFrom(self, node):
            if node.module == 'cvtk' or (node.module and node.module.startswith('cvtk.')):
                raise StopIteration

    try:
        CvtkImportVisitor().visit(ast.parse(content))
    except StopIteration:
        return True
    return False


def __infer_torch_model_name(runner_script):
    with open(runner_script, 'r') as infh:
        content = infh.read()

    patterns = [
        r"model_name\s*=\s*['\"]([^'\"]+)['\"]",
        r"ClsRunner\([^\n]*?['\"]([^'\"]+)['\"]",
        r"DetRunner\([^\n]*?model\s*=\s*['\"]([^'\"]+)['\"]",
        r"SegmRunner\([^\n]*?model\s*=\s*['\"]([^'\"]+)['\"]",
    ]
    for pattern in patterns:
        m = re.search(pattern, content)
        if m:
            return m.group(1)
    return None


def __generate_app_html_tmpl(tmpl_fpath, task, backend, model_arg, module_import, module_class):
    tmpl = []
    write_code_stack = [True]  # Stack to handle nested conditions
    
    with tmpl_fpath.open('r') as infh:
        for codeline in infh:
            if '#%CVTK%#' in codeline:
                if ' IF' in codeline:
                    # Parse TASK= or BACKEND= condition
                    task_match = re.search(r'TASK=([^\s\}]+)', codeline)
                    backend_match = re.search(r'BACKEND=([^\s\}]+)', codeline)
                    
                    condition_met = True
                    if task_match:
                        task_code = task_match.group(1)
                        condition_met = task in task_code
                    elif backend_match:
                        backend_code = backend_match.group(1)
                        condition_met = backend in backend_code
                    
                    # Push new condition state (only write if parent is writing AND condition is met)
                    write_code_stack.append(write_code_stack[-1] and condition_met)
                elif ' ENDIF' in codeline:
                    # Pop condition state
                    if len(write_code_stack) > 1:
                        write_code_stack.pop()
                continue

            # Only write lines if current condition is True
            if write_code_stack[-1]:
                tmpl.append(codeline)
                
    tmpl = ''.join(tmpl)
    tmpl = tmpl.replace('__MODULE_IMPORT__', module_import)
    tmpl = tmpl.replace('__MODULE_CLASS__', module_class)
    tmpl = tmpl.replace('__DATALABEL__', 'model.label.txt')
    tmpl = tmpl.replace('__MODELCFG__', model_arg)
    tmpl = tmpl.replace('__MODELWEIGHT__', 'model.pth')
    return tmpl
