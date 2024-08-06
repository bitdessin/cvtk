import os
import numpy as np
import requests
import PIL
import cvtk.ls
import unittest
import testutils


def get_app_status(url):
    print(f'Checking App Server ({url}) Status... ')
    try:
        res = requests.get(url)
        print(res.status_code)
        return True
    except:
        return False


class TestBaseUtils(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = testutils.set_ws('ls_baseutils')
    

    def test_exprot(self, host='http://localhost', port=8080):
        url = f'{host}:{port}'
        if get_app_status(url):
            cvtk.ls.export(3,
                           output=os.path.join(self.ws, 'instances.coco.json'),
                           format='coco',
                           host=host, port=port)



class TestScritpUtils(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = testutils.set_ws('ls_scriptutils')


    def test_export(self, host='http://localhost', port=8080):
        url = f'{host}:{port}'
        if get_app_status(url):
            testutils.run_cmd(['cvtk', 'ls-export', '--project', '3',
                               '--output', os.path.join(self.ws, 'instances.coco.json'),
                               '--format', 'coco',
                               '--host', host, '--port', port])




if __name__ == '__main__':
    unittest.main()
