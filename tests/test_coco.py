import os
import json
import cvtk.format.coco as cvtkcoco
import unittest
import testutils


class TestBaseUtils(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coco_files = [testutils.data['det']['train'],
                           testutils.data['det']['valid'],
                           testutils.data['det']['test']]
        self.coco_test_result = testutils.data['det']['test_result']
        self.coco_dicts = [self.__load_coco(f) for f in self.coco_files]
        self.n_images = [len(coco['images']) for coco in self.coco_dicts]
        self.n_anns = [len(coco['annotations']) for coco in self.coco_dicts]

        self.ws = testutils.set_ws('coco_baseutils')

    
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

    def test_merge(self):
        coco_merged_1 = cvtkcoco.combine(self.coco_files,
                                 os.path.join(self.ws, 'merged_from_file.json'))
        coco_merged_2 = cvtkcoco.combine(self.coco_dicts,
                                 os.path.join(self.ws, 'merged_from_dict.json'))

        self.assertEqual(self.__get_bboxes(self.coco_dicts[0]),
                         self.__get_bboxes(coco_merged_1))
        self.assertEqual(self.__get_categories(self.coco_dicts[0]),
                         self.__get_categories(coco_merged_1))
        
        self.assertEqual(coco_merged_1, coco_merged_2)
        
        self.assertEqual(len(coco_merged_1['images']), sum(self.n_images))
        self.assertEqual(len(coco_merged_1['annotations']), sum(self.n_anns))


    def test_split(self):
        coco_split_1 = cvtkcoco.split(self.coco_files[0],
                                os.path.join(self.ws, 'split_from_file.json'),
                                random_seed=1)
        coco_split_2 = cvtkcoco.split(self.coco_dicts[0],
                                os.path.join(self.ws, 'split_from_dict.json'),
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


    def test_reindex(self):
        cvtkcoco.reindex(self.coco_files[0],
                  os.path.join(self.ws,
                               os.path.splitext(os.path.basename(self.coco_files[0]))[0] + '.reindexed.json'))



    def test_calc_stats(self):
        stats = cvtkcoco.calc_stats(self.coco_files[2], self.coco_test_result)
        print(stats)

        stats = cvtkcoco.calc_stats(self.coco_files[2], self.coco_test_result,
                                    image_by='filename')
        print(stats)


        stats = cvtkcoco.calc_stats(self.coco_files[2], self.coco_test_result,
                                    image_by='filepath')
        print(stats)

        stats = cvtkcoco.calc_stats(self.coco_files[2], self.coco_test_result,
                                    category_by='name')
        print(stats)

        stats = cvtkcoco.calc_stats(self.coco_dicts[2], self.coco_test_result)


        


if __name__ == '__main__':
    unittest.main()
