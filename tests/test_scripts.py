import os
import unittest
import testutils



class TestScriptsBase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = testutils.set_ws('scripts_scriptsbase')


    def test_split_text(self):
        testutils.run_cmd(['cvtk', 'split',
                    '--input', testutils.data['cls']['all'],
                    '--output', os.path.join(self.ws, 'fruits_subset_1.txt'),
                    '--ratios', '6:3:1',
                    '--shuffle', '--stratify'])
        
        testutils.run_cmd(['cvtk', 'split',
                    '--input', testutils.data['cls']['all'],
                    '--output', os.path.join(self.ws, 'fruits_subset_2.txt'),
                    '--ratios', '6:3:1',
                    '--shuffle'])
        
        testutils.run_cmd(['cvtk', 'split',
                    '--input', testutils.data['cls']['all'],
                    '--output', os.path.join(self.ws, 'fruits_subset_3.txt'),
                    '--ratios', '6:3:1'])

    def test_split_coco(self):
        testutils.run_cmd(['cvtk', 'cocosplit',
                    '--input', testutils.data['det']['train'],
                    '--output', os.path.join(self.ws, 'strawberry_subset.json'),
                    '--ratios', '6:3:1',
                    '--shuffle'])
        
        input_coco = testutils.COCO(testutils.data['det']['train'])
        output_coco_1 = testutils.COCO(os.path.join(self.ws, 'strawberry_subset.json.0'))
        output_coco_2 = testutils.COCO(os.path.join(self.ws, 'strawberry_subset.json.1'))
        output_coco_3 = testutils.COCO(os.path.join(self.ws, 'strawberry_subset.json.2'))

        self.assertEqual(input_coco.images, output_coco_1.images | output_coco_2.images | output_coco_3.images)
        self.assertEqual(input_coco.annotations, output_coco_1.annotations | output_coco_2.annotations | output_coco_3.annotations)
        self.assertEqual(input_coco.categories, output_coco_1.categories)
        self.assertEqual(input_coco.categories, output_coco_2.categories)
        self.assertEqual(input_coco.categories, output_coco_3.categories)


    def test_merge_coco(self):
        testutils.run_cmd(['cvtk', 'cococombine',
                    '--input', testutils.data['det']['train'] + ',' + testutils.data['det']['valid'] + ',' + testutils.data['det']['test'],
                    '--output', os.path.join(self.ws, 'strawberry.merged.json')])
        
        input_coco_1 = testutils.COCO(testutils.data['det']['train'])
        input_coco_2 = testutils.COCO(testutils.data['det']['valid'])
        input_coco_3 = testutils.COCO(testutils.data['det']['test'])
        output_coco = testutils.COCO(os.path.join(self.ws, 'strawberry.merged.json'))

        self.assertEqual(input_coco_1.images | input_coco_2.images | input_coco_3.images, output_coco.images)
        self.assertEqual(len(input_coco_1.annotations) + len(input_coco_2.annotations) + len(input_coco_3.annotations), len(output_coco.annotations))
        self.assertEqual(input_coco_1.categories, output_coco.categories)
        self.assertEqual(input_coco_2.categories, output_coco.categories)
        self.assertEqual(input_coco_3.categories, output_coco.categories)
    

if __name__ == '__main__':
    unittest.main()
