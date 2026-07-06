import os
import shutil
import zipfile
import tempfile
import urllib
import json
import importlib
import label_studio_sdk


def deploy_demoapp(project: str, source: str, label: str, model: str, weights: str, vanilla: bool=False) -> None:
    """Generate a FastAPI application for inference of a classification or detection model
    
    This function generates a FastAPI application for inference of a classification or detection model.

    Args:
        project: A file path to save the FastAPI application.
        source: The source code of the model.
        label: The label file of the dataset.
        model: The configuration file of the model.
        weights: The weights file of the model.
        vanilla: Generate a FastAPI application without importation of cvtk. The default is False.

    Examples:
        >>> from cvtk.ml import deploy_demoapp
        >>> deploy_demoapp('./project', 'model.py', 'label.txt', 'model.cfg', 'model.pth')
    """

    if not os.path.exists(project):
        os.makedirs(project)

    coremodule = os.path.splitext(os.path.basename(source))[0]
    data_label = os.path.basename(label)
    model_cfg = os.path.basename(model)
    model_weights = os.path.basename(weights)

    shutil.copy2(source, os.path.join(project, coremodule + '.py'))
    shutil.copy2(label, os.path.join(project, data_label))
    if os.path.exists(model):
        shutil.copy2(model, os.path.join(project, model_cfg))
    shutil.copy2(weights, os.path.join(project, model_weights))

    source_task_type = __estimate_source_task(source)
    source_is_vanilla = __estimate_source_vanilla(source)

    # FastAPI script
    tmpl = __generate_app_html_tmpl(importlib.resources.files('cvtk').joinpath(f'tmpl/_flask.py'),
                                    source_task_type)
    if vanilla:
        if source_is_vanilla:
            for i in range(len(tmpl)):
                if (tmpl[i][:9] == 'from cvtk') and ('import ClsRunner' in tmpl[i]):
                    tmpl[i] = f'from {coremodule} import ClsRunner'
        else:
            # user specified vanilla, but the source code for CV task is not vanilla
            print('The `ClsRunner` class definition is not found in the source code. `ClsRunner` will be generated with importation of cvtk regardless vanilla is specified.')
    tmpl = ''.join(tmpl)
    tmpl = tmpl.replace('__DATALABEL__', data_label)
    tmpl = tmpl.replace('__MODELCFG__', model_cfg)
    tmpl = tmpl.replace('__MODELWEIGHT__', model_weights)
    with open(os.path.join(project, 'main.py'), 'w') as fh:
        fh.write(tmpl)

    # HTML template
    if not os.path.exists(os.path.join(project, 'templates')):
        os.makedirs(os.path.join(project, 'templates'))
    tmpl = __generate_app_html_tmpl(importlib.resources.files('cvtk').joinpath(f'tmpl/html/_flask.html'), source_task_type)
    with open(os.path.join(project, 'templates', 'index.html'), 'w') as fh:
        fh.write(''.join(tmpl))



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
            if 'clsRunner' in codeline:
                task_type = 'cls'
                break
            elif 'detRunner' in codeline:
                task_type = 'det'
                break
    return task_type


def __estimate_source_vanilla(source):
    is_vanilla = True
    with open(source, 'r') as infh:
        for codeline in infh:
            if ('import cvtk' in codeline) or ('from cvtk' in codeline):
                is_vanilla = False
                break
    return is_vanilla
