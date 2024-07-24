import os
import subprocess
import unittest
import testutils



class TestScriptsBase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpath = os.path.join('outputs', 'test_scripts', 'base')
        testutils.make_dirs(self.dpath)


    def test_split_dataset(self):
        testutils.run_cmd(['cvtk', 'split',
                    '--input', testutils.cls_data['all'],
                    '--output', os.path.join(self.dpath, 'fruits_subset.txt'),
                    '--type', 'text',
                    '--ratios', '6:3:1',
                    '--shuffle', '--balanced'])


if __name__ == '__main__':
    unittest.main()
