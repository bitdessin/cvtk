import os
import json
import copy
import cvtk
import unittest
import testutils

PRINT_OUTPUT = False


class TestDataCoco(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coco_files = [testutils.data['det']['train'],
                           testutils.data['det']['valid'],
                           testutils.data['det']['test']]
        self.coco_test_result = testutils.data['det']['test_result']
        self.coco_dicts = [self.__load_coco(f) for f in self.coco_files]
        self.n_images = [len(coco['images']) for coco in self.coco_dicts]
        self.n_anns = [len(coco['annotations']) for coco in self.coco_dicts]
        self.image_roots = [os.path.dirname(coco_file) for coco_file in self.coco_files]

        self.ws = testutils.set_ws('data_coco')
    
    
    def __load_coco(self, coco_fpath):
        with open(coco_fpath, 'r') as fh:
            return json.load(fh)
        
        
    def __get_bboxes(self, coco, image_name='ff39545e.jpg'):
        if isinstance(coco, str):
            coco = self.__load_coco(coco)
        image_id = [img['id'] for img in coco['images'] if os.path.basename(img['file_name']) == os.path.basename(image_name)][0]
        anns = [ann for ann in coco['annotations'] if ann['image_id'] == image_id]
        return [ann['bbox'] for ann in anns]
    
    
    def __get_categories(self, coco, image_name='ff39545e.jpg'):
        if isinstance(coco, str):
            coco = self.__load_coco(coco)
        cateid2name = {cate['id']: cate['name'] for cate in coco['categories']}
        image_id = [img['id'] for img in coco['images'] if os.path.basename(img['file_name']) == os.path.basename(image_name)][0]
        anns = [ann for ann in coco['annotations'] if ann['image_id'] == image_id]
        return [cateid2name[ann['category_id']] for ann in anns]


    def __use_basename_file_names(self, coco):
        coco = copy.deepcopy(coco)
        for image in coco['images']:
            image['file_name'] = os.path.basename(image['file_name'])
        return coco


    def test_merge(self):
        coco_merged_1 = cvtk.data.coco.combine(self.coco_files,
                                 output=os.path.join(self.ws, 'merged_from_file.json'))
        coco_merged_2 = cvtk.data.coco.combine(self.coco_dicts,
                                 output=os.path.join(self.ws, 'merged_from_dict.json'))
        coco_merged_3 = cvtk.data.coco.combine(self.coco_files[0])

        self.assertEqual(self.__get_bboxes(self.coco_dicts[0]),
                         self.__get_bboxes(coco_merged_1))
        self.assertEqual(self.__get_categories(self.coco_dicts[0]),
                         self.__get_categories(coco_merged_1))
        
        self.assertEqual(coco_merged_1, coco_merged_2)
        self.assertEqual(self.__get_bboxes(self.coco_dicts[0]),
                         self.__get_bboxes(coco_merged_3))
        self.assertEqual(self.__get_categories(self.coco_dicts[0]),
                         self.__get_categories(coco_merged_3))
        self.assertEqual(len(coco_merged_3['images']), len(self.coco_dicts[0]['images']))
        self.assertEqual(len(coco_merged_3['annotations']), len(self.coco_dicts[0]['annotations']))
        
        self.assertEqual(len(coco_merged_1['images']), sum(self.n_images))
        self.assertEqual(len(coco_merged_1['annotations']), sum(self.n_anns))


    def test_split(self):
        coco_split_1 = cvtk.data.coco.split(self.coco_files[0],
                                output=os.path.join(self.ws, 'split_from_file.json'),
                                random_seed=1)
        coco_split_2 = cvtk.data.coco.split(self.coco_dicts[0],
                                output=os.path.join(self.ws, 'split_from_dict.json'),
                                random_seed=1)
        
        self.assertEqual(self.__get_bboxes(self.coco_dicts[0]),
                         self.__get_bboxes(coco_split_1[0]))
        self.assertEqual(self.__get_categories(self.coco_dicts[0]),
                         self.__get_categories(coco_split_1[0]))

        self.assertEqual(coco_split_1, coco_split_2)
        
        self.assertEqual(self.n_images[0],
                         sum([len(coco['images']) for coco in coco_split_1]))
        self.assertEqual(self.n_anns[0],
                         sum([len(coco['annotations']) for coco in coco_split_1]))


    def test_image_root(self):
        coco_with_basenames = self.__use_basename_file_names(self.coco_dicts[0])
        combined = cvtk.data.coco.combine(coco_with_basenames, image_root=self.image_roots[0])
        self.assertTrue(all(
            image['file_name'] == os.path.join(self.image_roots[0], os.path.basename(src['file_name']))
            for image, src in zip(combined['images'], self.coco_dicts[0]['images'])
        ))

        crop_data = {
            'images': [copy.deepcopy(self.coco_dicts[0]['images'][0])],
            'annotations': [copy.deepcopy(ann) for ann in self.coco_dicts[0]['annotations']
                            if ann['image_id'] == self.coco_dicts[0]['images'][0]['id']],
            'categories': copy.deepcopy(self.coco_dicts[0]['categories'])
        }
        crop_output = os.path.join(self.ws, 'crop_with_image_root')
        cvtk.data.coco.crop(crop_data, output=crop_output, image_root=self.image_roots[0])
        self.assertEqual(len(os.listdir(crop_output)), len(crop_data['annotations']))


    def test_reindex(self):
        cvtk.data.coco.reindex(self.coco_files[0],
                  output=os.path.join(self.ws,
                                      os.path.splitext(
                                          os.path.basename(self.coco_files[0]))[0] + '.reindexed.json'))

    
    def test_remove(self):
        cocodata = cvtk.data.coco.remove(self.coco_files[0],
                        output=os.path.join(self.ws,
                                            os.path.splitext(
                                                os.path.basename(self.coco_files[0]))[0] + '.removed.json'),
                        images=[1,
                                'data/strawberry/train/images/2129c05b.jpg'],
                        categories='flower')
        
        coco_images = []
        for _ in cocodata['images']:
            coco_images.append(_['id'])
            coco_images.append(_['file_name'])
        self.assertNotIn('data/strawberry/train/images/2129c05b.jpg', coco_images)
        self.assertNotIn(1, coco_images)
        
        coco_cates = []
        for _ in cocodata['categories']:
            coco_cates.append(_['id'])
            coco_cates.append(_['name'])
        self.assertNotIn('flower', coco_cates)

        coco_unchanged = cvtk.data.coco.remove(self.coco_files[0])
        self.assertEqual(coco_unchanged, self.coco_dicts[0])


    def test_stats(self):
        output_fpath = os.path.join(self.ws, 'stats.json')
        stats = cvtk.data.coco.stats(self.coco_files[2], output=output_fpath)
        self.assertEqual(stats['n_images'], len(self.coco_dicts[2]['images']))
        self.assertEqual(stats['n_categories'], len(self.coco_dicts[2]['categories']))
        self.assertIsInstance(stats['n_annotations'], dict)
        self.assertEqual(sum(stats['n_annotations'].values()), len(self.coco_dicts[2]['annotations']))

        with open(output_fpath, 'r') as fh:
            self.assertEqual(json.load(fh), stats)

        stats_from_dict = cvtk.data.coco.stats(self.coco_dicts[2])
        self.assertEqual(stats_from_dict, stats)


    def test_calc_stats(self):
        stats = cvtk.data.coco.calc_stats(self.coco_files[2], self.coco_test_result)
        if PRINT_OUTPUT:
            print(stats)

        stats = cvtk.data.coco.calc_stats(self.coco_files[2], self.coco_test_result,
                                    image_by='file_name')
        if PRINT_OUTPUT:
            print(stats)


        stats = cvtk.data.coco.calc_stats(self.coco_files[2], self.coco_test_result,
                                    image_by='file_name')
        if PRINT_OUTPUT:
            print(stats)

        stats = cvtk.data.coco.calc_stats(self.coco_files[2], self.coco_test_result,
                                    category_by='name')
        if PRINT_OUTPUT:
            print(stats)

        stats = cvtk.data.coco.calc_stats(self.coco_dicts[2], self.coco_test_result)




class TestDataCocoScripts(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = testutils.set_ws('coco_scriptutils')

    def test_coco_split(self):
        testutils.run_cmd(['cvtk', 'coco-split',
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


    def test_coco_merge(self):
        testutils.run_cmd(['cvtk', 'coco-combine',
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


    def test_coco_remove(self):
        testutils.run_cmd(['cvtk', 'coco-remove',
                    '--input', testutils.data['det']['train'],
                    '--output', os.path.join(self.ws, 'strawberry.remove.json'),
                    '--categories', 'flower'])


    def test_coco_stats(self):
        testutils.run_cmd(['cvtk', 'coco-stats',
                    '--input', testutils.data['det']['train']])
    


if __name__ == '__main__':
    unittest.main()
