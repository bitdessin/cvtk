import os
import json
import copy
import numpy as np
import cvtk
import unittest
import testutils

PRINT_OUTPUT = False


class TestDataIm(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = testutils.set_ws('coco_data_im')
        
        self.coco_files = [testutils.data['det']['train'],
                           testutils.data['det']['valid'],
                           testutils.data['det']['test']]
        self.coco_test_result = testutils.data['det']['test_result']
        self.coco_dicts = [self.__load_coco(f) for f in self.coco_files]
        self.n_images = [len(coco['images']) for coco in self.coco_dicts]
        self.n_anns = [len(coco['annotations']) for coco in self.coco_dicts]
        self.image_roots = [os.path.dirname(coco_file) for coco_file in self.coco_files]

    
    def __load_coco(self, coco_file):
        with open(coco_file, 'r') as f:
            return json.load(f)


    def test_Bbox_creation(self):
        """Test bbox creation with different methods"""
        # from_xywh: top-left (x=20, y=10) with w=30, h=30 -> bottom-right (x=50, y=40)
        bbox_xywh = cvtk.data.Bbox.from_xywh(20, 10, 30, 30, imsize=(100, 100))
        self.assertEqual(bbox_xywh.to_xyxy(), (20, 10, 50, 40))
        
        # from_xyxy: top-left (x1=20, y1=10) to bottom-right (x2=50, y2=40)
        bbox_xyxy = cvtk.data.Bbox.from_xyxy(20, 10, 50, 40, imsize=(100, 100))
        self.assertEqual(bbox_xyxy.to_xyxy(), (20, 10, 50, 40))
        self.assertEqual(bbox_xyxy, bbox_xywh)
        
        # from_cxcywh: center (cx=35, cy=25) with w=30, h=30
        bbox_cxcywh = cvtk.data.Bbox.from_cxcywh(35, 25, 30, 30, imsize=(100, 100))
        self.assertEqual(bbox_cxcywh.to_xyxy(), (20, 10, 50, 40))
    
    
    def test_Bbox_conversions(self):
        """Test Bbox format conversions"""
        bbox = cvtk.data.Bbox.from_xyxy(20, 10, 50, 40, imsize=(100, 100))
        
        # XYXY format
        self.assertEqual(bbox.to_xyxy(), (20, 10, 50, 40))
        
        # XYWH format
        xywh = bbox.to_xywh()
        self.assertEqual(xywh, (20, 10, 30, 30))
        
        # CXCYWH format
        cxcywh = bbox.to_cxcywh()
        self.assertEqual(cxcywh, (35.0, 25.0, 30, 30))
        
        # Static conversions
        self.assertEqual(cvtk.data.Bbox.xywh2xyxy((20, 10, 30, 30)), (20, 10, 50, 40))
        self.assertEqual(cvtk.data.Bbox.xyxy2xywh((20, 10, 50, 40)), (20, 10, 30, 30))
    
    
    def test_Bbox_properties(self):
        """Test Bbox properties"""
        bbox = cvtk.data.Bbox.from_xyxy(20, 10, 50, 40, imsize=(100, 100))
        self.assertEqual(bbox.width, 30)
        self.assertEqual(bbox.height, 30)
        self.assertEqual(bbox.area, 900)
    
    
    def test_segm_from_mask(self):
        """Test Segm creation from mask"""
        mask = np.zeros((100, 200), dtype=bool)
        mask[10:40, 20:50] = True
        
        segm = cvtk.data.Segm.from_mask(mask)
        self.assertEqual(segm.size, (200, 100))
        self.assertEqual(segm.to_mask().shape, (100, 200))
        np.testing.assert_array_equal(segm.to_mask(), mask)
    
    
    def test_segm_mask_to_rle_conversion(self):
        """Test Segm conversion between mask and RLE"""
        mask = np.zeros((100, 200), dtype=bool)
        mask[10:40, 20:50] = True
        
        segm = cvtk.data.Segm.from_mask(mask)
        rle = segm.to_rle()
        
        self.assertIn("counts", rle)
        self.assertIn("size", rle)
        self.assertEqual(tuple(rle["size"]), (100, 200))
        
        # Convert back and verify
        segm2 = cvtk.data.Segm.from_rle(rle)
        np.testing.assert_array_equal(segm2.to_mask(), mask)
    
    
    def test_segm_mask_to_polygons_conversion(self):
        """Test Segm conversion between mask and polygons"""
        mask = np.zeros((100, 200), dtype=bool)
        mask[10:40, 20:50] = True
        
        segm = cvtk.data.Segm.from_mask(mask)
        polygons = segm.to_polygons()
        
        self.assertIsInstance(polygons, list)
        self.assertGreater(len(polygons), 0)
        for poly in polygons:
            self.assertEqual(len(poly) % 2, 0)


    def test_instanceannotation_properties(self):
        """Test InstanceAnnotation properties"""
        bbox = cvtk.data.Bbox(10, 20, 40, 50)
        mask = np.zeros((100, 200), dtype=bool)
        mask[10:40, 20:50] = True
        segm = cvtk.data.Segm.from_mask(mask)
        
        # With segm
        ann_with_segm = cvtk.data.InstanceAnnotation("person", bbox=bbox, segm=segm, score=0.95)
        self.assertIsNotNone(ann_with_segm.area)
        self.assertEqual(ann_with_segm.score, 0.95)
        
        # With only Bbox
        ann_with_bbox = cvtk.data.InstanceAnnotation("person", bbox=bbox)
        self.assertEqual(ann_with_bbox.area, 900)
        
        # With neither
        ann_no_geo = cvtk.data.InstanceAnnotation("person")
        self.assertIsNone(ann_no_geo.area)
    
    
    def test_instanceannotation_to_dict(self):
        """Test InstanceAnnotation serialization"""
        bbox = cvtk.data.Bbox.from_xywh(10, 20, 30, 30, imsize=(100, 100))
        ann = cvtk.data.InstanceAnnotation("person", bbox=bbox, score=0.95)
        
        ann_dict = ann.to_dict(bbox_format="xywh", segm_format="rle")
        self.assertEqual(ann_dict["label"], "person")
        self.assertEqual(ann_dict["bbox"], (10, 20, 30, 30))
        self.assertEqual(ann_dict["score"], 0.95)
        self.assertIsNone(ann_dict["segm"])


    def test_imagedataset_from_coco(self):
        """Test ImageDataset loading from COCO format"""
        coco_dict = self.coco_dicts[0]
        image_root = self.image_roots[0]
        
        dataset = cvtk.data.ImageDataset.from_coco(coco_dict, image_root=image_root)
        
        self.assertEqual(dataset.size, self.n_images[0])
        self.assertGreater(dataset.size, 0)
        
        # Check first image record
        record = dataset.records[0]
        self.assertIsNotNone(record.source)
        self.assertIsNotNone(record.size)
        self.assertGreater(len(record.annotations), 0)
    
    
    def test_imagedataset_to_coco_without_image_root(self):
        """Test ImageDataset conversion to COCO without image_root"""
        coco_dict = self.coco_dicts[0]
        image_root = self.image_roots[0]
        
        dataset = cvtk.data.ImageDataset.from_coco(coco_dict, image_root=image_root)
        coco_dict_out = dataset.to_coco()
        
        self.assertEqual(len(coco_dict_out["images"]), self.n_images[0])
        self.assertEqual(len(coco_dict_out["annotations"]), self.n_anns[0])
        self.assertGreater(len(coco_dict_out["categories"]), 0)
    
    
    def test_imagedataset_to_coco_with_image_root(self):
        """Test ImageDataset conversion to COCO with image_root prefix removal"""
        coco_dict = self.coco_dicts[0]
        image_root = self.image_roots[0]
        
        dataset = cvtk.data.ImageDataset.from_coco(coco_dict, image_root=image_root)
        coco_dict_out = dataset.to_coco(image_root=image_root)
        
        # Check that file_names are relative paths
        for img in coco_dict_out["images"]:
            file_name = img["file_name"]
            self.assertFalse(os.path.isabs(file_name), 
                           f"Expected relative path, got absolute: {file_name}")
    
    
    def test_imagedataset_roundtrip(self):
        """Test COCO to ImageDataset and back maintains data integrity"""
        coco_dict_in = self.coco_dicts[0]
        image_root = self.image_roots[0]
        
        # Load from COCO
        dataset = cvtk.data.ImageDataset.from_coco(coco_dict_in, image_root=image_root)
        
        # Convert back to COCO (without image_root to keep full paths for comparison)
        coco_dict_out = dataset.to_coco()
        
        # Verify structure is preserved
        self.assertEqual(len(coco_dict_out["images"]), len(coco_dict_in["images"]))
        self.assertEqual(len(coco_dict_out["annotations"]), len(coco_dict_in["annotations"]))
        self.assertEqual(len(coco_dict_out["categories"]), len(coco_dict_in["categories"]))
        
        # Verify category names match
        in_cats = {cat["name"] for cat in coco_dict_in["categories"]}
        out_cats = {cat["name"] for cat in coco_dict_out["categories"]}
        self.assertEqual(in_cats, out_cats)


    def test_imagedataset_from_coco_file(self):
        """Test loading ImageDataset directly from COCO file"""
        coco_file = self.coco_files[0]
        image_root = self.image_roots[0]
        
        coco_dict = self.__load_coco(coco_file)
        dataset = cvtk.data.ImageDataset.from_coco(coco_dict, image_root=image_root)
        
        self.assertEqual(dataset.size, self.n_images[0])
        self.assertGreater(dataset.size, 0)
        
        # Verify all records have valid data
        for record in dataset.records:
            self.assertIsNotNone(record.source)
            self.assertTrue(os.path.exists(record.source), 
                          f"Image file not found: {record.source}")
            self.assertIsNotNone(record.size)
            self.assertGreater(record.size[0], 0)
            self.assertGreater(record.size[1], 0)


    def test_imagedataset_dataset_draw_bbox(self):
        self.__test_imagerecord_draw_with_layers('det')
        
    
    def test_imagedataset_dataset_draw_segm(self):
        self.__test_imagerecord_draw_with_layers('segm')
        

    def __test_imagerecord_draw_with_layers(self, atag):
        """Test drawing ImageRecord with different layers"""
        
        coco_file = testutils.data[atag]['train']
        image_root = os.path.dirname(coco_file)
        
        coco_dict = self.__load_coco(coco_file)
        dataset = cvtk.data.ImageDataset.from_coco(coco_dict, image_root=image_root)
        
        record = None
        i = 0
        for r in dataset.records:
            if len(r.annotations) > 0 and any(ann.bbox for ann in r.annotations):
                i += 1
                record = r
                        
                self.assertIsNotNone(record, "No record with bbox annotations found")
                
                # bbox
                output_bbox = os.path.join(self.ws, f"test_draw_bbox_{i}.{atag}.jpg")
                im_bbox = record.draw(layers='bbox', output=output_bbox, label=True)
                self.assertIsNotNone(im_bbox)
                self.assertTrue(os.path.exists(output_bbox))
                
                # bbox + segm
                output_multi = os.path.join(self.ws, f"test_draw_bboxsegm_{i}.{atag}.jpg")
                im_multi = record.draw(layers=['bbox', 'segm'], output=output_multi, label=True, score=True)
                self.assertIsNotNone(im_multi)
                self.assertTrue(os.path.exists(output_multi))

                # bbox + segm + overlay
                output_multi = os.path.join(self.ws, f"test_draw_bboxesegmoverlay_{i}.{atag}.jpg")
                im_multi = record.draw(layers=['overlay', 'bbox', 'segm'], output=output_multi, label=True, score=True)
                self.assertIsNotNone(im_multi)
                self.assertTrue(os.path.exists(output_multi))

                # bbox + segm + mask
                output_multi = os.path.join(self.ws, f"test_draw_bboxesegmmask_{i}.{atag}.jpg")
                im_multi = record.draw(layers=['mask', 'bbox', 'segm'], output=output_multi, label=True, score=True)
                self.assertIsNotNone(im_multi)
                self.assertTrue(os.path.exists(output_multi))

                
                # Test drawing without saving
                im_nosave = record.draw(layers='bbox', label=False)
                self.assertIsNotNone(im_nosave)
                
                if i > 10:
                    break



if __name__ == '__main__':
    unittest.main()

