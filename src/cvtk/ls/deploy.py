import os
import shutil
import importlib
import re
import label_studio_sdk



def mlbackend(project: str, source: str, label: str, model: str = None, weights: str = None, vanilla=False) -> None:
    """Generate ML backend for Assisted Labeling in Label Studio.

    This function generates an backend server to help assisted labeling 
    in Label Studio.
    """

    if not os.path.exists(project):
        os.makedirs(project)
    
    coremodule = os.path.splitext(os.path.basename(source))[0]
    data_label = os.path.basename(label)
    model_cfg = os.path.basename(model) if model is not None else None
    model_weights = os.path.basename(weights)

    shutil.copy2(source, os.path.join(project, coremodule + '.py'))
    shutil.copy2(label, os.path.join(project, data_label))
    if model is not None and os.path.exists(model):
        shutil.copy2(model, os.path.join(project, model_cfg))
    shutil.copy2(weights, os.path.join(project, model_weights))

    source_task_type = __estimate_source_task(source)
    source_backend = __estimate_source_backend(source)
    source_is_vanilla = __estimate_source_vanilla(source)

    if source_task_type == 'cls':
        raise ValueError('Label Studio backend deployment currently supports detection/segmentation sources only.')

    # FastAPI script
    tmpl = __generate_app_html_tmpl(importlib.resources.files('cvtk').joinpath(f'tmpl/_ls_backend.py'),
                                    source_task_type)

    if source_task_type == 'cls':
        module_import = 'cvtk.ml.torchapi'
        module_class = 'ClsRunner'
    elif source_backend == 'torch':
        module_import = 'cvtk.ml.torchdetapi'
        module_class = 'SegmRunner' if source_task_type == 'segm' else 'DetRunner'
    else:
        module_import = 'cvtk.ml.mmdetapi'
        module_class = 'SegmRunner' if source_task_type == 'segm' else 'DetRunner'

    if vanilla:
        if source_is_vanilla:
            module_import = coremodule
        else:
            # user specified vanilla, but the source code for CV task is not vanilla
            print('The source code is not vanilla. Backend will import model core from cvtk package.')

    tmpl = ''.join(tmpl)
    tmpl = tmpl.replace('__MODULE_IMPORT__', module_import)
    tmpl = tmpl.replace('__MODULE_CLASS__', module_class)
    tmpl = tmpl.replace('__DATALABEL__', data_label)
    tmpl = tmpl.replace('__MODELCFG__', model_cfg)
    tmpl = tmpl.replace('__MODELWEIGHT__', model_weights)
    with open(os.path.join(project, 'mlbackend.py'), 'w') as fh:
        fh.write(tmpl)


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
    task_type = None
    with open(source, 'r') as infh:
        for codeline in infh:
            code = codeline.lower()
            if 'segmrunner' in code:
                task_type = 'segm'
                break
            elif 'clsrunner' in code:
                task_type = 'cls'
                break
            elif 'detrunner' in code:
                task_type = 'det'
                break

    if task_type is None:
        task_type = 'det'

    return task_type


def __estimate_source_backend(source):
    backend = None
    with open(source, 'r') as infh:
        for codeline in infh:
            code = codeline.lower()
            if 'cvtk.ml.mmdetapi' in code:
                backend = 'mmdet'
                break
            if 'cvtk.ml.torchdetapi' in code or 'cvtk.ml.torchdet' in code:
                backend = 'torch'
                break
            if 'cvtk.ml.torchapi' in code:
                backend = 'torch'
                break

    if backend is None:
        backend = 'mmdet'

    return backend


def __estimate_source_vanilla(source):
    is_vanilla = True
    with open(source, 'r') as infh:
        for codeline in infh:
            if ('import cvtk' in codeline) or ('from cvtk' in codeline):
                is_vanilla = False
                break
    return is_vanilla


