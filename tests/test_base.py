import os
import numpy as np
import cvtk
import unittest
import testutils


class TestBaseUtils(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.im_dpath = testutils.data['cls']['samples']
        self.im_fpath = testutils.data['cls']['sample']
        self.output_dpath = testutils.set_ws('outputs/cvtk_baseutils')


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

        self.im_fpath = testutils.data['cls']['sample']

        self.labels = ['leaf', 'flower', 'root']
        self.bboxes = [[0, 0, 10, 10],
                       [10, 10, 20, 20],
                       [20, 20, 30, 30]]
        self.masks = [np.random.randint(2, (240, 321)).tolist(),
                      np.random.randint(2, (240, 321)).tolist(),
                      np.random.randint(2, (240, 321)).tolist()]
        self.scores = [0.9, 0.8, 0.7]
        self.areas = [np.sum(_) for _ in self.masks]
    

    def test_imann(self):
        ia = cvtk.ImageAnnotation(self.labels, self.bboxes, self.masks, self.scores)

        self.assertEqual(len(ia), 3)

        self.assertEqual(ia.labels, self.labels)
        self.assertEqual([list(_) for _ in ia.bboxes], self.bboxes)
        self.assertEqual([_.tolist() for _ in ia.masks], self.masks)
        self.assertEqual(ia.scores, self.scores)

        x = np.random.randint(0, 1, (240, 321))
        print(x.shape)
        print(x)
        
        self.assertEqual(ia.label(0), self.labels[0])
        self.assertEqual(list(ia.bbox(1)), self.bboxes[1])
        self.assertEqual(ia.mask(2).tolist(), self.masks[2])
        self.assertEqual(ia.score(0), self.scores[0])

        ia.dump()
        ia.dump(indent=2, ensure_ascii=False)


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


if __name__ == '__main__':
    unittest.main()
