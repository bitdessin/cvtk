import os
import sys
import json
import shutil
import tempfile
import unittest
import importlib.util
from io import BytesIO

import PIL.Image
import testutils
import cvtk
from cvtk.ml import deploy as ml_deploy


class TestDeployHelpers(unittest.TestCase):
    def test_parse_runner_script_ignores_docstring_mentions_of_cvtk(self):
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as fh:
            fh.write('"""Example usage: from cvtk import something"""\n\ndef main():\n    return 1\n')
            path = fh.name

        try:
            _, _, is_vanilla = ml_deploy.__parse_runner_script(path)
            self.assertTrue(is_vanilla)
        finally:
            os.remove(path)


class TestDemoAppDeploymentBase(unittest.TestCase):
    """Test demoapp deployment for all backend/task/source code combinations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = testutils.set_ws('demoapp_deployment')
        
        # Test data
        self.cls_label = testutils.data['cls']['label']
        self.cls_train = testutils.data['cls']['train']
        self.cls_valid = testutils.data['cls']['valid']
        self.cls_samples = testutils.data['cls']['samples']
        self.cls_sample_img = testutils.data['cls']['sample']
        
        self.det_label = testutils.data['det']['label']
        self.det_train = testutils.data['det']['train']
        self.det_valid = testutils.data['det']['valid']
        self.det_samples = testutils.data['det']['samples']
        
        self.segm_label = testutils.data['segm']['label']
        self.segm_train = testutils.data['segm']['train']
        self.segm_valid = testutils.data['segm']['valid']
        self.segm_samples = testutils.data['segm']['samples']


    def _create_test_image(self, format='JPEG'):
        """Create a test image in memory."""
        img = PIL.Image.new('RGB', (224, 224), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format=format)
        img_bytes.seek(0)
        return img_bytes


    def _load_flask_app(self, app_dir):
        """Dynamically load Flask app from generated main.py."""
        main_py = os.path.join(app_dir, 'main.py')
        spec = importlib.util.spec_from_file_location("demoapp_main", main_py)
        module = importlib.util.module_from_spec(spec)
        sys.modules['demoapp_main'] = module
        spec.loader.exec_module(module)
        return module.app


    def _test_demoapp_inference(self, app, task):
        """Test demoapp inference endpoint using Flask test client."""
        client = app.test_client()
        test_img = self._create_test_image()
        
        # POST image to inference endpoint (use tuple format for file upload: (file_obj, filename))
        response = client.post('/api/inference', data={'file': (test_img, 'test.jpg')})
        
        # Verify response
        self.assertEqual(response.status_code, 200, 
                        f"Expected 200 but got {response.status_code}: {response.data}")
        
        # Parse response
        if response.content_type == 'application/json':
            result = response.get_json()
        else:
            result = json.loads(response.data)
        
        # Extract the actual data from the wrapper
        self.assertIn('data', result, "Response should contain 'data' field")
        output = result['data']
        self.assertIsNotNone(output, "Output data should not be None")
        
        # Task-specific validations
        if task == 'cls':
            self.assertIsInstance(output, list, "Classification should return list of predictions")
            self.assertGreater(len(output), 0, "Should have at least one prediction")
            pred = output[0]
            self.assertIn('label', pred, "Prediction should have 'label' field")
            self.assertIn('prob', pred, "Prediction should have 'prob' field")
            self.assertGreater(pred['prob'], 0, "Probability should be positive")
        elif task in ['det', 'detection']:
            self.assertIsInstance(output, dict, "Detection should return dict with annotations")
            self.assertIn('annotations', output, "Detection should return 'annotations'")
            self.assertIsInstance(output['annotations'], list, "Annotations should be a list")
        elif task in ['seg', 'segm', 'segmentation']:
            self.assertIsInstance(output, dict, "Segmentation should return dict with annotations")
            self.assertIn('annotations', output, "Segmentation should return 'annotations'")
            self.assertIsInstance(output['annotations'], list, "Annotations should be a list")


    def _train_and_deploy_model(self, backend, task, vanilla, code_generator='api'):
        """
        Train a model and deploy a demoapp.
        
        Args:
            backend: 'torch' or 'mmdet'
            task: 'cls', 'det', or 'segm'
            vanilla: bool, whether to use vanilla (standalone) code
            code_generator: 'api' or 'script' (command-line)
        
        Returns:
            Tuple of (app_dir, weights_path, label_path)
        """
        # Setup working directory
        module = 'vanilla' if vanilla else 'cvtk'
        ws_name = f'{backend}_{task}_{module}_{code_generator}'
        dpath = testutils.set_ws(os.path.join(self.ws, ws_name))
        
        # Select data
        if task == 'cls':
            label_path = self.cls_label
            train_path = self.cls_train
            valid_path = self.cls_valid
            test_path = testutils.data['cls']['test']
            samples_path = self.cls_samples
        elif task == 'det':
            label_path = self.det_label
            train_path = self.det_train
            valid_path = self.det_valid
            test_path = testutils.data['det']['test']
            samples_path = self.det_samples
        else:  # segm
            label_path = self.segm_label
            train_path = self.segm_train
            valid_path = self.segm_valid
            test_path = testutils.data['segm']['test']
            samples_path = self.segm_samples
        
        # Generate runner script
        script_path = os.path.join(dpath, 'runner.py')
        
        if code_generator == 'api':
            cvtk.ml.deploy.runner(script_path, backend=backend, task=task, vanilla=vanilla)
        else:  # script
            cmd = ['cvtk', 'deploy-model', '--backend', backend, '--task', task, '--script', script_path]
            if vanilla:
                cmd.append('--vanilla')
            testutils.run_cmd(cmd)
        
        # Train model
        weights_path = os.path.join(dpath, 'model.pth')
        train_cmd = ['python', script_path, 'train',
                     '--label', label_path,
                     '--train', train_path,
                     '--valid', valid_path,
                     '--test', test_path,
                     '--epoch', '5',
                     '--output_weights', weights_path]
        
        testutils.run_cmd(train_cmd)
        
        # Deploy demoapp
        app_name = os.path.join(dpath, 'demoapp')
        
        if code_generator == 'api':
            cvtk.ml.deploy.demoapp(app_name, script_path, weights_path, label_path)
        else:  # script
            cmd = ['cvtk', 'deploy-demoapp', '--app_name', app_name, '--runner_script', script_path, 
                   '--label', label_path, '--weights', weights_path]
            if vanilla:
                cmd.append('--vanilla')
            testutils.run_cmd(cmd)
        
        return app_name, weights_path, label_path
    


class TestDemoAppDeploymentTorchCls(TestDemoAppDeploymentBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = testutils.set_ws('demoapp_deployment_torch_cls')

    # Classification tests
    def test_cls_torch_cvtk_api(self):
        """Classification with PyTorch, cvtk source code, API."""
        app_dir, _, _ = self._train_and_deploy_model('torch', 'cls', False, 'api')
        app = self._load_flask_app(app_dir)
        self._test_demoapp_inference(app, 'cls')


    def test_cls_torch_cvtk_script(self):
        """Classification with PyTorch, cvtk source code, CLI."""
        app_dir, _, _ = self._train_and_deploy_model('torch', 'cls', False, 'script')
        app = self._load_flask_app(app_dir)
        self._test_demoapp_inference(app, 'cls')


    def test_cls_torch_vanilla_api(self):
        """Classification with PyTorch, vanilla source code, API."""
        app_dir, _, _ = self._train_and_deploy_model('torch', 'cls', True, 'api')
        app = self._load_flask_app(app_dir)
        self._test_demoapp_inference(app, 'cls')


    def test_cls_torch_vanilla_script(self):
        """Classification with PyTorch, vanilla source code, CLI."""
        app_dir, _, _ = self._train_and_deploy_model('torch', 'cls', True, 'script')
        app = self._load_flask_app(app_dir)
        self._test_demoapp_inference(app, 'cls')


class TestDemoAppDeploymentTorchDet(TestDemoAppDeploymentBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = testutils.set_ws('demoapp_deployment_torch_det')

    def test_det_torch_cvtk_api(self):
        """Detection with PyTorch (Faster R-CNN), cvtk source code, API."""
        app_dir, _, _ = self._train_and_deploy_model('torch', 'det', False, 'api')
        app = self._load_flask_app(app_dir)
        self._test_demoapp_inference(app, 'det')


    def test_det_torch_cvtk_script(self):
        """Detection with PyTorch (Faster R-CNN), cvtk source code, CLI."""
        app_dir, _, _ = self._train_and_deploy_model('torch', 'det', False, 'script')
        app = self._load_flask_app(app_dir)
        self._test_demoapp_inference(app, 'det')


    def test_det_torch_vanilla_api(self):
        """Detection with PyTorch (Faster R-CNN), vanilla source code, API."""
        app_dir, _, _ = self._train_and_deploy_model('torch', 'det', True, 'api')
        app = self._load_flask_app(app_dir)
        self._test_demoapp_inference(app, 'det')


    def test_det_torch_vanilla_script(self):
        """Detection with PyTorch (Faster R-CNN), vanilla source code, CLI."""
        app_dir, _, _ = self._train_and_deploy_model('torch', 'det', True, 'script')
        app = self._load_flask_app(app_dir)
        self._test_demoapp_inference(app, 'det')


class TestDemoAppDeploymentMMDetDet(TestDemoAppDeploymentBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = testutils.set_ws('demoapp_deployment_mmdet_det')


    # Detection with MMDetection (Faster R-CNN)
    def test_det_mmdet_cvtk_api(self):
        """Detection with MMDetection (Faster R-CNN), cvtk source code, API."""
        app_dir, _, _ = self._train_and_deploy_model('mmdet', 'det', False, 'api')
        app = self._load_flask_app(app_dir)
        self._test_demoapp_inference(app, 'det')


    def test_det_mmdet_cvtk_script(self):
        """Detection with MMDetection (Faster R-CNN), cvtk source code, CLI."""
        app_dir, _, _ = self._train_and_deploy_model('mmdet', 'det', False, 'script')
        app = self._load_flask_app(app_dir)
        self._test_demoapp_inference(app, 'det')


    def test_det_mmdet_vanilla_api(self):
        """Detection with MMDetection (Faster R-CNN), vanilla source code, API."""
        app_dir, _, _ = self._train_and_deploy_model('mmdet', 'det', True, 'api')
        app = self._load_flask_app(app_dir)
        self._test_demoapp_inference(app, 'det')


    def test_det_mmdet_vanilla_script(self):
        """Detection with MMDetection (Faster R-CNN), vanilla source code, CLI."""
        app_dir, _, _ = self._train_and_deploy_model('mmdet', 'det', True, 'script')
        app = self._load_flask_app(app_dir)
        self._test_demoapp_inference(app, 'det')


class TestDemoAppDeploymentTorchSegm(TestDemoAppDeploymentBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = testutils.set_ws('demoapp_deployment_torch_segm')
        
        
    def test_segm_torch_cvtk_api(self):
        """Instance Segmentation with PyTorch (Mask R-CNN), cvtk source code, API."""
        app_dir, _, _ = self._train_and_deploy_model('torch', 'segm', False, 'api')
        app = self._load_flask_app(app_dir)
        self._test_demoapp_inference(app, 'segm')


    def test_segm_torch_cvtk_script(self):
        """Instance Segmentation with PyTorch (Mask R-CNN), cvtk source code, CLI."""
        app_dir, _, _ = self._train_and_deploy_model('torch', 'segm', False, 'script')
        app = self._load_flask_app(app_dir)
        self._test_demoapp_inference(app, 'segm')


    def test_segm_torch_vanilla_api(self):
        """Instance Segmentation with PyTorch (Mask R-CNN), vanilla source code, API."""
        app_dir, _, _ = self._train_and_deploy_model('torch', 'segm', True, 'api')
        app = self._load_flask_app(app_dir)
        self._test_demoapp_inference(app, 'segm')


    def test_segm_torch_vanilla_script(self):
        """Instance Segmentation with PyTorch (Mask R-CNN), vanilla source code, CLI."""
        app_dir, _, _ = self._train_and_deploy_model('torch', 'segm', True, 'script')
        app = self._load_flask_app(app_dir)
        self._test_demoapp_inference(app, 'segm')


class TestDemoAppDeploymentMMDetSegm(TestDemoAppDeploymentBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = testutils.set_ws('demoapp_deployment_mmdet_segm')


    def test_segm_mmdet_cvtk_api(self):
        """Instance Segmentation with MMDetection (Mask R-CNN), cvtk source code, API."""
        app_dir, _, _ = self._train_and_deploy_model('mmdet', 'segm', False, 'api')
        app = self._load_flask_app(app_dir)
        self._test_demoapp_inference(app, 'segm')


    def test_segm_mmdet_cvtk_script(self):
        """Instance Segmentation with MMDetection (Mask R-CNN), cvtk source code, CLI."""
        app_dir, _, _ = self._train_and_deploy_model('mmdet', 'segm', False, 'script')
        app = self._load_flask_app(app_dir)
        self._test_demoapp_inference(app, 'segm')


    def test_segm_mmdet_vanilla_api(self):
        """Instance Segmentation with MMDetection (Mask R-CNN), vanilla source code, API."""
        app_dir, _, _ = self._train_and_deploy_model('mmdet', 'segm', True, 'api')
        app = self._load_flask_app(app_dir)
        self._test_demoapp_inference(app, 'segm')


    def test_segm_mmdet_vanilla_script(self):
        """Instance Segmentation with MMDetection (Mask R-CNN), vanilla source code, CLI."""
        app_dir, _, _ = self._train_and_deploy_model('mmdet', 'segm', True, 'script')
        app = self._load_flask_app(app_dir)
        self._test_demoapp_inference(app, 'segm')


if __name__ == '__main__':
    unittest.main()
