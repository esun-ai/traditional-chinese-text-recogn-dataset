
import six
import lmdb
from PIL import Image


class LmdbConnector:
    def __init__(self, root, mode='r'):
        assert mode in ['r', 'w']
        self.mode = mode
        if self.mode == 'r':
            self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
            if not self.env:
                raise Exception('cannot open lmdb from %s' % (root))
        elif self.mode == 'w':
            self.env = lmdb.open(root, map_size=1099511627776)
            self.write_cnt = 0

    def _check_read_mode(self):
        assert self.mode == 'r'

    def _check_write_mode(self):
        assert self.mode == 'w'

    def get_label_key(self, index):
        return 'label-%09d'.encode() % index

    def get_img_key(self, index):
        return 'image-%09d'.encode() % index

    def get_num_key(self):
        return 'num-samples'.encode()

    def count(self):
        self._check_read_mode()
        with self.env.begin(write=False) as txn:
            num = int(txn.get(self.get_num_key()))
        return num

    def indexes_generator(self):
        self._check_read_mode()
        for i in range(self.count()):
            index = i + 1  # starts with 1
            yield index

    def get_label(self, index):
        self._check_read_mode()
        with self.env.begin(write=False) as txn:
            label_key = self.get_label_key(index)
            label = txn.get(label_key).decode('utf-8')
        return label

    def get_image_binary(self, index):
        self._check_read_mode()
        with self.env.begin(write=False) as txn:
            img_key = self.get_img_key(index)
            imgbuf = txn.get(img_key)
        return imgbuf

    def get_image(self, index, rgb=False):
        self._check_read_mode()
        imgbuf = self.get_image_binary(index)
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        try:
            if rgb:
                img = Image.open(buf).convert('RGB')  # for color image
            else:
                img = Image.open(buf).convert('L')
        except IOError:
            print(f'Corrupted image for {index}')
            img = None
        return img

    def write(self, img_bin_and_label_pairs):
        self._check_write_mode()
        with self.env.begin(write=True) as txn:
            for x in img_bin_and_label_pairs:
                if x is None:  # for async
                    continue
                img_bin, label = x
                index = self.write_cnt + 1  # starts with 1
                img_key = self.get_img_key(index)
                label_key = self.get_label_key(index)
                txn.put(img_key, img_bin)
                txn.put(label_key, label.encode())
                self.write_cnt += 1
            num_key = self.get_num_key()
            num = str(self.write_cnt).encode()
            txn.put(num_key, num)
