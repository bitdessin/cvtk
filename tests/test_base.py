import os
import numpy as np
import PIL
import cvtk
import unittest
import testutils


class TestBaseUtils(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.im_dpath = testutils.data['cls']['samples']
        self.im_fpath = testutils.data['cls']['sample']
        self.ws = testutils.set_ws('base_baseutils')


    def test_imconvert(self):
        im = cvtk.imread(self.im_fpath)

        im_cv = cvtk.imconvert(im, 'cv')
        im_bytes = cvtk.imconvert(im, 'bytes')
        im_base64 = cvtk.imconvert(im, 'base64')
        im_pil = cvtk.imconvert(im, 'pil')
        im_gray = cvtk.imconvert(im, 'gray')

        im_from_cv = cvtk.imread(im_cv)
        im_from_bytes = cvtk.imread(im_bytes)
        im_from_base64 = cvtk.imread(im_base64)
        im_from_pil = cvtk.imread(im_pil)
        im_from_gray = cvtk.imread(im_gray)

        self.assertEqual(im.size, im_from_cv.size)
        self.assertEqual(im.size, im_from_bytes.size)
        self.assertEqual(im.size, im_from_base64.size)

        cvtk.imwrite(im, os.path.join(self.ws, 'cvtk_imconvert.jpg'))
        cvtk.imwrite(im_from_cv, os.path.join(self.ws, 'cvtk_imconvert_cv.jpg'))
        cvtk.imwrite(im_from_bytes, os.path.join(self.ws, 'cvtk_imconvert_bytes.jpg'))
        cvtk.imwrite(im_from_base64, os.path.join(self.ws, 'cvtk_imconvert_base64.jpg'))
        cvtk.imwrite(im_from_pil, os.path.join(self.ws, 'cvtk_imconvert_pil.jpg'))
        cvtk.imwrite(im_from_gray, os.path.join(self.ws, 'cvtk_imconvert_gray.jpg'))


    def test_imresize(self):
        cvtk.imwrite(cvtk.imresize(self.im_fpath, shape=(100, 300)),
                     os.path.join(self.ws, 'cvtk_imresize_100x300.jpg'))
        cvtk.imwrite(cvtk.imresize(self.im_fpath, scale=0.5),
                     os.path.join(self.ws, 'cvtk_imresize_scale05.jpg'))
        cvtk.imwrite(cvtk.imresize(self.im_fpath, shortest=100),
                     os.path.join(self.ws, 'cvtk_imresize_shortest100.jpg'))
        cvtk.imwrite(cvtk.imresize(self.im_fpath, longest=200),
                     os.path.join(self.ws, 'cvtk_imresize_longest100.jpg'))


    def test_imlist(self):
        imgs = cvtk.imlist(self.im_dpath)
        print(imgs)

    
    def test_imshow(self):
        plt1 = cvtk.imshow(self.im_fpath)
        plt1.savefig(os.path.join(self.ws, 'cvtk_imshow_1.png'))

        imgs = cvtk.imlist(self.im_dpath)

        plt2 = cvtk.imshow(imgs[:5])
        plt2.savefig(os.path.join(self.ws, 'cvtk_imshow_2.png'))

        plt3 = cvtk.imshow(imgs[:5], ncol=2)
        plt3.savefig(os.path.join(self.ws, 'cvtk_imshow_3.png'))

        plt4 = cvtk.imshow(imgs[:5], nrow=2)
        plt4.savefig(os.path.join(self.ws, 'cvtk_imshow_4.png'))




class TestImageClasses(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = testutils.set_ws('base_imageclasses')

        self.im_fpath = testutils.data['cls']['sample']
        self.labels = ['leaf', 'flower', 'root']
        self.bboxes = [[10, 10, 80, 50],
                       [30, 60, 100, 120],
                       [150, 200, 200, 250]]
        mask_1 = np.zeros((240, 321))
        mask_1[10:30, 10:50] = 1
        mask_1[20:50, 30:80] = 1
        mask_2 = np.zeros((240, 321))
        mask_2[60:80, 30:100] = 1
        mask_2[60:120, 30:50] = 1
        mask_2[60:120, 80:100] = 1
        mask_3 = np.zeros((240, 321))
        mask_3[200:250, 160:180] = 1
        mask_3[220:230, 150:200] = 1
        self.masks = [mask_1.astype(np.uint8).tolist(),
                      mask_2.astype(np.uint8).tolist(),
                      mask_3.astype(np.uint8).tolist()]
        self.scores = [0.9, 0.3, 0.7]
        self.areas = [np.sum(_) for _ in self.masks]
    

    def test_im(self):
        im_cvtk = cvtk.Image(self.im_fpath)
        im_pil = PIL.Image.open(self.im_fpath)
        self.assertEqual(im_cvtk.source, self.im_fpath)
        self.assertEqual(im_cvtk.size, im_pil.size)
        self.assertEqual(im_cvtk.width, im_pil.width)
        self.assertEqual(im_cvtk.height, im_pil.height)


    def test_imann(self):
        ia = cvtk.Annotation(self.labels, self.bboxes, masks=None, scores=None)

        ia = cvtk.Annotation(self.labels, self.bboxes, self.masks, self.scores)

        self.assertEqual(len(ia), 3)

        self.assertEqual(ia.labels, self.labels)
        self.assertEqual([list(_) for _ in ia.bboxes], self.bboxes)
        self.assertEqual([_.tolist() for _ in ia.masks], self.masks)
        self.assertEqual(ia.scores, self.scores)

        x = np.random.randint(0, 1, (240, 321))
        print(x.shape)
        print(x)
        
        ia.dump()
        ia.dump(indent=2, ensure_ascii=False)



    def test_im_imann(self):
        ia = cvtk.Annotation(self.labels, self.bboxes, self.masks, self.scores)
        im = cvtk.Image(self.im_fpath, ia)
        for i, ann in enumerate(im.annotations):
            self.assertEqual(ann, ia[i])
        
        im.draw(format='bbox+segm',
                output=os.path.join(self.ws, 'imdraw.1.png'))
        im.draw(format='bbox+segm',
                output=os.path.join(self.ws, 'imdraw.2.png'),
                col={'leaf': (255, 0, 0), 'flower': (0, 255, 0), 'root': (0, 0, 255)})
        im.draw(format='mask',
                output=os.path.join(self.ws, 'imdraw.3.png'))
        im.draw(format='rgbmask',
                output=os.path.join(self.ws, 'imdraw.4.png'),
                col={'leaf': (255, 0, 0), 'flower': (0, 255, 0), 'root': (0, 0, 255)})


    def test_imdeck(self):
        ia = cvtk.Annotation(self.labels, self.bboxes, self.masks, self.scores)
        im = cvtk.Image(self.im_fpath, ia)
        imdeck = cvtk.ImageDeck([im, im, im])
        imdeck.append(im)
        imdeck.extend([im, im])

        for i, im in enumerate(imdeck):
            self.assertEqual(im, imdeck[i])

        self.assertEqual(len(imdeck), 6)
        self.assertEqual(imdeck[0], im)

        fmt = imdeck.format('cvtk')
        print(fmt)

        fmt = imdeck.format('coco')
        print(fmt)

        imdeck.dump(os.path.join(self.ws, 'imdeck_dump_cvtk.json'), 'cvtk')
        imdeck.dump(os.path.join(self.ws, 'imdeck_dump_coco.json'), 'coco')
        imdeck.dump(os.path.join(self.ws, 'imdeck_dump_coco_cl.json'), 'coco',
                    datalabel=self.labels[::-1])        




if __name__ == '__main__':
    unittest.main()
