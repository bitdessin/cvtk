coverage run -p -m unittest test_base
coverage run -p -m unittest test_coco
coverage run -p -m unittest test_ml
coverage run -p -m unittest test_mmdet
coverage run -p -m unittest test_torch
coverage run -p -m unittest test_demoapp
coverage run -p -m unittest test_ls
coverage combine
coverage report -m
coverage html

