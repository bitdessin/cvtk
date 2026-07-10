coverage run -p -m unittest test_io_im 
coverage run -p -m unittest test_data_im
coverage run -p -m unittest test_data_coco

coverage run -p -m unittest test_ml_base
coverage run -p -m unittest test_ml_torch
coverage run -p -m unittest test_ml_torchdet
coverage run -p -m unittest test_ml_mmdet

coverage run -p -m unittest test_demoapp.TestDemoAppDeploymentTorchCls
coverage run -p -m unittest test_demoapp.TestDemoAppDeploymentTorchDet
coverage run -p -m unittest test_demoapp.TestDemoAppDeploymentMMDetDet
coverage run -p -m unittest test_demoapp.TestDemoAppDeploymentTorchSegm
coverage run -p -m unittest test_demoapp.TestDemoAppDeploymentMMDetSegm
coverage run -p -m unittest test_ls

coverage combine
coverage report -m
coverage html
