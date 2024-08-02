coverage run -p -m unittest test_base
coverage run -p -m unittest test_coco
coverage run -p -m unittest test_mlbase
coverage run -p -m unittest test_scripts
coverage run -p -m unittest test_mmdet
coverage run -p -m unittest test_torch
coverage run -p -m unittest test_fastapi
coverage combine
coverage report -m
coverage html

