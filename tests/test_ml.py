import os
import numpy as np
import PIL
import cvtk.ml
import unittest
import testutils


class TestMLBase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_file = testutils.data['cls']['all']
        self.ws = testutils.set_ws('mlbase_mlbase')



    def test_split(self):
        subsets = cvtk.ml.split_dataset(self.data_file,
                                        output=os.path.join(self.ws, 'split_1.txt'),
                                        ratios=[0.8, 0.1, 0.1])
        
        subsets = cvtk.ml.split_dataset(self.data_file,
                                        output=os.path.join(self.ws, 'split_2.txt'),
                                        ratios=[0.8, 0.1, 0.1],
                                        shuffle=False, stratify=False)
        
        subsets = cvtk.ml.split_dataset(self.data_file,
                                        output=os.path.join(self.ws, 'split_2.txt'),
                                        ratios=[0.8, 0.1, 0.1],
                                        shuffle=True, stratify=True)

        data = [['1.jpg', 'leaf'],
                ['2.jpg', 'leaf'],
                ['3.jpg', 'leaf'],
                ['4.jpg', 'leaf'],
                ['5.jpg', 'flower'],
                ['6.jpg', 'flower'],
                ['7.jpg', 'flower'],
                ['8.jpg', 'fruit'],
                ['9.jpg', 'fruit'],
                ['10.jpg', 'fruit']]
        subsets = cvtk.ml.split_dataset(data,
                                        output=os.path.join(self.ws, 'split_3.txt'),
                                        ratios=[0.4, 0.3, 0.3],
                                        shuffle=True, stratify=True)
        print('shuffle=True, stratify=True')
        print(subsets)

        subsets = cvtk.ml.split_dataset(data,
                                        output=os.path.join(self.ws, 'split_4.txt'),
                                        ratios=[0.4, 0.3, 0.3],
                                        shuffle=False, stratify=True)
        print('shuffle=False, stratify=True')
        print(subsets)

        subsets = cvtk.ml.split_dataset(data,
                                        output=os.path.join(self.ws, 'split_5.txt'),
                                        ratios=[0.4, 0.3, 0.3],
                                        shuffle=True, stratify=False)
        print('shuffle=True, stratify=False')
        print(subsets)


        


if __name__ == '__main__':
    unittest.main()
