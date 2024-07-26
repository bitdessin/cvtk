import os
import unittest
import testutils



class TestScriptsBase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpath = testutils.set_ws(os.path.join('outputs', 'test_scripts', 'base'))


    def test_split_dataset(self):
        testutils.run_cmd(['cvtk', 'split',
                    '--input', testutils.data['cls']['all'],
                    '--output', os.path.join(self.dpath, 'fruits_subset.txt'),
                    '--type', 'text',
                    '--ratios', '6:3:1',
                    '--shuffle', '--balanced'])


if __name__ == '__main__':
    unittest.main()
