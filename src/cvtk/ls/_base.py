import os
import shutil
import zipfile
import tempfile
import urllib
import json
import importlib
from ..ml._subutils import __estimate_source_task, __estimate_source_vanilla, __generate_app_html_tmpl
import label_studio_sdk



def __get_client(url=None, token=None):
    url = os.getenv('LABEL_STUDIO_URL', url)
    token = os.getenv('LABEL_STUDIO_API_KEY', token)
    if token is None or url is None:
        raise ValueError(f'This function requires the environment variables of '
                         f'`LABEL_STUDIO_URL` and `LABEL_STUDIO_API_KEY` to be set.')
    return label_studio_sdk.Client(url=url, api_key=token)


def export(project: int,
           output: str,
           format: str='COCO',
           url: str|None=None,
           token: str|None=None) -> dict:
    """
    Export annotations from Label Studio project.

    Args:
        project: An ID of Label Studio project to export.
        output: A path to save the exported data.
        format: The format of the exported data. The supported formats are `COCO`,
                `JSON` (Label Studio JSON), `JSON_MIN`, `CSV`, `TSV`, `VOC` (Pascal VOC),
                `YOLO`, and others (see Label Studio Documentations for details).
                Note that Only COCO has been implemented so far.
        host: Label Studio host. Default is 'localhost'.
        port: Label Studio port. Default is 8080.
        api_key: Label Studio API key. Default is None.
        indent: JSON indent. Default is 4.
        ensure_ascii: Ensure ASCII. Default is False

    Returns:
        dict: A dictionary of the exported data.

    Examples:
        >>> import os
        >>> from cvtk.ls import export
        >>> 
        >>> data = export(project=0, output='instances.coco.json', format='COCO',
                          host='localhost', port=8080,
                          api_key='f6dea26f0a0f81883e04681b4e649c600fe50fc')
        >>> print(data)
        {'info': {'contributor': 'Label Studio', 'description': '', ...., 'images': [...], 'annotations': [...]}
        >>> 
        >>> os.environ['LABEL_STUDIO_API_KEY'] = 'f6dea26f0a0f81883e04681b4e649c600fe50fc'
        >>> data = export(project=0, output='instances.coco.json', format='COCO',
                          host='localhost', port=8080)
        >>> 
    """
    client = __get_client(url=url, token=token)
    prj = client.get_project(project)
    ls_data_root = os.getenv('LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT')
    format = format.upper()

    with tempfile.TemporaryDirectory() as temp_dpath:
        tempf_output_ = os.path.join(temp_dpath, 'output.zip')

        prj.export_tasks(export_type=format,
                         download_all_tasks=False,
                         download_resources=False,
                         export_location=tempf_output_)

        if format == 'COCO':
            with zipfile.ZipFile(tempf_output_, 'r') as zf:
                zf.extract('result.json', path=temp_dpath)
                shutil.copy(os.path.join(temp_dpath, 'result.json'), output)
        else:
            raise NotImplementedError(f'Export format `{format}` is not implemented yet.')

        # modify the image path in the exported json file
        exported_data = None
        with open(output, 'r') as fh:
            exported_data = json.load(fh)
            for img in exported_data['images']:
                img['file_name'] = img['file_name'].replace('\/', '/')
                if '/data/local-files/?d=' in img['file_name']:
                    img['file_name'] = img['file_name'].replace('/data/local-files/?d=', '')
                    if ls_data_root is not None:                        
                        img['file_name'] = os.path.join(ls_data_root, img['file_name'])
                img['file_name'] = urllib.parse.unquote(img['file_name'])
                
            with open(output, 'w') as f:
                json.dump(exported_data, f, ensure_ascii=True)
 
        return exported_data
    


def generate_app(project: str, source: str, label: str, model: str, weights: str, vanilla=False) -> None:
    """Generate ML backend for Assisted Labeling in Label Studio.

    This function generates an backend server to help assisted labeling 
    in Label Studio.
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
    tmpl = __generate_app_html_tmpl(importlib.resources.files('cvtk').joinpath(f'tmpl/_ls_backend.py'),
                                    source_task_type)

    if vanilla:
        if source_is_vanilla:
            for i in range(len(tmpl)):
                if (tmpl[i][:9] == 'from cvtk') and ('import ModuleCore' in tmpl[i]):
                    tmpl[i] = f'from {coremodule} import ModuleCore'
        else:
            # user specified vanilla, but the source code for CV task is not vanilla
            print('The `ModuleCore` class definition is not found in the source code. `ModuleCore` will be generated with importation of cvtk regardless vanilla is specified.')

    tmpl = ''.join(tmpl)
    tmpl = tmpl.replace('__DATALABEL__', data_label)
    tmpl = tmpl.replace('__MODELCFG__', model_cfg)
    tmpl = tmpl.replace('__MODELWEIGHT__', model_weights)
    with open(os.path.join(project, 'mlbackend.py'), 'w') as fh:
        fh.write(tmpl)
