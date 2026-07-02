import os
import numpy as np
import PIL
import cvtk
import unittest
import testutils

PRINT_OUTPUT = False


class TestUtilsIm(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.im_dpath = testutils.data['cls']['samples']
        self.im_fpath = testutils.data['cls']['sample']
        self.ws = testutils.set_ws('io_im')


    def test_imconvert(self):
        im = cvtk.io.imread(self.im_fpath)

        im_cv = cvtk.io.imconvert(im, 'cv')
        im_bytes = cvtk.io.imconvert(im, 'bytes')
        im_base64 = cvtk.io.imconvert(im, 'base64')
        im_pil = cvtk.io.imconvert(im, 'pil')
        im_gray = cvtk.io.imconvert(im, 'gray')

        im_from_cv = cvtk.io.imread(im_cv)
        im_from_bytes = cvtk.io.imread(im_bytes)
        im_from_base64 = cvtk.io.imread(im_base64)
        im_from_pil = cvtk.io.imread(im_pil)
        im_from_gray = cvtk.io.imread(im_gray)

        self.assertEqual(im.size, im_from_cv.size)
        self.assertEqual(im.size, im_from_bytes.size)
        self.assertEqual(im.size, im_from_base64.size)
        self.assertEqual(im.size, im_from_pil.size)
        
        cvtk.io.imwrite(im, os.path.join(self.ws, 'cvtk_imconvert.jpg'))
        cvtk.io.imwrite(im_from_cv, os.path.join(self.ws, 'cvtk_imconvert_cv.jpg'))
        cvtk.io.imwrite(im_from_bytes, os.path.join(self.ws, 'cvtk_imconvert_bytes.jpg'))
        cvtk.io.imwrite(im_from_base64, os.path.join(self.ws, 'cvtk_imconvert_base64.jpg'))
        cvtk.io.imwrite(im_from_pil, os.path.join(self.ws, 'cvtk_imconvert_pil.jpg'))
        cvtk.io.imwrite(im_from_gray, os.path.join(self.ws, 'cvtk_imconvert_gray.jpg'))


    def test_imresize(self):
        cvtk.io.imwrite(cvtk.io.imresize(self.im_fpath, shape=(100, 300)),
                     os.path.join(self.ws, 'cvtk_imresize_100x300.jpg'))
        cvtk.io.imwrite(cvtk.io.imresize(self.im_fpath, scale=0.5),
                     os.path.join(self.ws, 'cvtk_imresize_scale05.jpg'))
        cvtk.io.imwrite(cvtk.io.imresize(self.im_fpath, shortest=100),
                     os.path.join(self.ws, 'cvtk_imresize_shortest100.jpg'))
        cvtk.io.imwrite(cvtk.io.imresize(self.im_fpath, longest=200),
                     os.path.join(self.ws, 'cvtk_imresize_longest100.jpg'))


    def test_imlist(self):
        imgs = cvtk.io.imlist(self.im_dpath)

    
    def test_imshow(self):
        plt1 = cvtk.io.imshow(self.im_fpath)
        plt1.savefig(os.path.join(self.ws, 'cvtk_imshow_1.png'))

        imgs = cvtk.io.imlist(self.im_dpath)

        plt2 = cvtk.io.imshow(imgs[:5])
        plt2.savefig(os.path.join(self.ws, 'cvtk_imshow_2.png'))

        plt3 = cvtk.io.imshow(imgs[:5], ncol=2)
        plt3.savefig(os.path.join(self.ws, 'cvtk_imshow_3.png'))

        plt4 = cvtk.io.imshow(imgs[:5], nrow=2)
        plt4.savefig(os.path.join(self.ws, 'cvtk_imshow_4.png'))



if __name__ == '__main__':
    unittest.main()
