import os
import cvtk
import unittest


class TestBaseUtils(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.im_dpath = 'data/fruits/images'
        self.im_fpath = 'data/fruits/images/14c6e557.jpg'
        self.output_dpath = 'outputs/cvtk_baseutils'
        if not os.path.exists(self.output_dpath):
            os.makedirs(self.output_dpath)


    def test_imconvert(self):
        im = cvtk.imread(self.im_fpath)

        im_cv = cvtk.imconvert(im, 'cv')
        im_bytes = cvtk.imconvert(im, 'bytes')
        im_base64 = cvtk.imconvert(im, 'base64')

        im_from_cv = cvtk.imread(im_cv)
        im_from_bytes = cvtk.imread(im_bytes)
        im_from_base64 = cvtk.imread(im_base64)

        self.assertEqual(im.size, im_from_cv.size)
        self.assertEqual(im.size, im_from_bytes.size)
        self.assertEqual(im.size, im_from_base64.size)

        cvtk.imwrite(im, os.path.join(self.output_dpath, 'cvtk_imconvert.jpg'))
        cvtk.imwrite(im_from_cv, os.path.join(self.output_dpath, 'cvtk_imconvert_cv.jpg'))
        cvtk.imwrite(im_from_bytes, os.path.join(self.output_dpath, 'cvtk_imconvert_bytes.jpg'))
        cvtk.imwrite(im_from_base64, os.path.join(self.output_dpath, 'cvtk_imconvert_base64.jpg'))



    def test_imresize(self):
        cvtk.imwrite(cvtk.imresize(self.im_fpath, shape=(100, 300)),
                     os.path.join(self.output_dpath, 'cvtk_imresize_100x300.jpg'))
        cvtk.imwrite(cvtk.imresize(self.im_fpath, scale=0.5),
                     os.path.join(self.output_dpath, 'cvtk_imresize_scale05.jpg'))
        cvtk.imwrite(cvtk.imresize(self.im_fpath, shortest=100),
                     os.path.join(self.output_dpath, 'cvtk_imresize_shortest100.jpg'))
        cvtk.imwrite(cvtk.imresize(self.im_fpath, longest=200),
                     os.path.join(self.output_dpath, 'cvtk_imresize_longest100.jpg'))


    def test_imlist(self):
        imgs = cvtk.imlist(self.im_dpath)
        print(imgs)

    
    def test_imshow(self):
        plt1 = cvtk.imshow(self.im_fpath)
        plt1.savefig(os.path.join(self.output_dpath, 'cvtk_imshow_1.png'))

        imgs = cvtk.imlist(self.im_dpath)
        plt2 = cvtk.imshow(imgs[:5], ncol=2)
        plt2.savefig(os.path.join(self.output_dpath, 'cvtk_imshow_2.png'))


class TestImageClasses(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.im_fpath = 'data/fruits/images/14c6e557.jpg'

        self.labels = ['leaf', 'flower', 'root']
        self.bboxes = [[0, 0, 10, 10],
                       [10, 10, 20, 20],
                       [20, 20, 30, 30]]
        self.masks = [[0, 0, 10, 10, 10, 10, 0, 10],
                         [10, 10, 20, 20, 20, 20, 10, 20],
                         [20, 20, 30, 30, 30, 30, 20, 30]]
        self.scores = [0.9, 0.8, 0.7]
    

    def test_imann(self):
        ia = cvtk.ImageAnnotation(self.labels, self.bboxes, self.masks, self.scores)

        self.assertEqual(len(ia), 3)

        self.assertEqual(ia.labels, self.labels)
        self.assertEqual(ia.bboxes, self.bboxes)
        self.assertEqual(ia.masks, self.masks)
        self.assertEqual(ia.scores, self.scores)

        self.assertEqual(ia[0],
            {'label': self.labels[0],
             'bbox': self.bboxes[0],
             'mask': self.masks[0],
             'score': self.scores[0]})
        
        self.assertEqual(ia.label(0), self.labels[0])
        self.assertEqual(ia.bbox(1), self.bboxes[1])
        self.assertEqual(ia.mask(2), self.masks[2])
        self.assertEqual(ia.score(0), self.scores[0])

        print(ia.dump())
        print(ia.dump(indent=2, ensure_ascii=False))


    def test_im(self):
        im = cvtk.Image(self.im_fpath)

        print(im.size)
        print(im.width)
        print(im.height)


    def test_im_imann(self):
        ia = cvtk.ImageAnnotation(self.labels, self.bboxes, self.masks, self.scores)
        im = cvtk.Image(self.im_fpath, ia)
        for i, ann in enumerate(im.annotations):
            self.assertEqual(ann, ia[i])
            print(ann)



if __name__ == '__main__':
    unittest.main()
