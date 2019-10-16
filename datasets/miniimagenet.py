# Data loading part borrowed from Ren's paper (https://github.com/renmengye/few-shot-ssl-public)

from utils import *
import utils
from collections import namedtuple

N_IMAGES = 600
N_INPUT = 84
IMAGES_PATH = "images/"
# To do 1: Change the paths below to the train/val/test csv files respectively.
CSV_FILES = {
    'train': '/home/root/data/miniImagenet/train.csv',
    'val': '/home/root/data/miniImagenet/val.csv',
    'test': '/home/root/data/miniImagenet/test.csv'
}
FIXED_SEED = 22

AL_Instance = namedtuple('AL_Instance',
                         'n_class, n_distractor, k_train, k_test, k_unlbl')
DatasetT = namedtuple('Dataset', 'data, labels')

class MiniImagenet(Dataset):

    def __init__(self,
               folder,
               split,
               nway=5,
               nshot=1,
               num_unlabel=5,
               num_distractor=5,
               num_test=15,
               split_def="",
               label_ratio=0.4,
               shuffle_episode=False,
               seed=FIXED_SEED,
               aug_90=False):

        self._folder = folder
        self._split = split
        self._seed = seed
        self._num_distractor = 0 if args.disable_distractor else num_distractor
        self._label_ratio = args.label_ratio if label_ratio is None else label_ratio
        self.n_lbl = int(N_IMAGES * self._label_ratio)
        print("split {}".format(split))
        print("num unlabel {}".format(num_unlabel))
        print("num test {}".format(num_test))
        print("num distractor {}".format(self._num_distractor))

        # define AL instance
        self.al_instance = AL_Instance(
            n_class=nway,
            n_distractor=self._num_distractor,
            k_train=nshot,
            k_test=num_test,
            k_unlbl=num_unlabel)
        self.n_input = N_INPUT

        self.images_path = os.path.join(self._folder, IMAGES_PATH)
        if not self._read_cache(split):
            self._write_cache(split, CSV_FILES[split])
        self.class_dict = self.split_label_unlabel(self.class_dict)
        self._num_classes = len(self.class_dict.keys())
        self._lbl_idx = []
        self._unlbl_idx = []
        self._cls_label = {}
        cls_label = 0
        print(self.class_dict.keys())
        for kk in self.class_dict.keys():
            _nlbl = len(self.class_dict[kk]['lbl'])
            _nunlbl = len(self.class_dict[kk]['unlbl'])
            self._lbl_idx.extend(self.class_dict[kk]['lbl'])
            self._unlbl_idx.extend(self.class_dict[kk]['unlbl'])
            for idx in self.class_dict[kk]['lbl']:
                self._cls_label[idx] = cls_label
            for idx in self.class_dict[kk]['unlbl']:
                self._cls_label[idx] = cls_label
            cls_label += 1
        self._lbl_idx = np.array(self._lbl_idx)
        self._unlbl_idx = np.array(self._unlbl_idx)
        self._num_lbl = len(self._lbl_idx)
        self._num_unlbl = len(self._unlbl_idx)
        print('Num label', self._num_lbl)


    def get_cache_path(self, split):
        cache_path = os.path.join(self._folder,
                              "mini-imagenet-cache-" + split + ".pkl")

        return cache_path

    def _read_cache(self, split):
        cache_path = self.get_cache_path(split)
        if os.path.exists(cache_path):
          try:
            with open(cache_path, "rb") as f:
              data = pkl.load(f, encoding='bytes')
              self.img_data = data[b'image_data']
              self.class_dict = data[b'class_dict']
          except:
            with open(cache_path, "rb") as f:
              data = pkl.load(f)
              self.img_data = data['image_data']
              self.class_dict = data['class_dict']
          return True
        else:
          return False

    def _write_cache(self, split, csv_filename):
        cache_path = self.get_cache_path(split)
        img_data = []

        class_dict = {}
        i = 0
        with open(csv_filename) as csv_file:
          csv_reader = csv.reader(csv_file)
          for (image_filename, class_name) in csv_reader:
            if 'label' not in class_name:
              if class_name in class_dict:
                class_dict[class_name].append(i)
              else:
                class_dict[class_name] = [i]
              img_data.append(
                  cv2.resize(
                      cv2.imread(self.images_path + image_filename)
                      [:, :, [2, 1, 0]], (self.n_input, self.n_input)))
              i += 1

        self.img_data = np.stack(img_data)
        self.class_dict = class_dict
        data = {"image_data": self.img_data, "class_dict": self.class_dict}
        with open(cache_path, "wb") as f:
          pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)


    def __getAllLabeled__(self):
        sel_classes = np.random.choice(
              range(len(self.class_dict.keys())),
              size=len(self.class_dict.keys()),
              replace=False)

        k_per_class = [
              None
              for i in range(len(self.class_dict.keys()))
          ]

        total_train = None
        total_test = None
        total_unlbl = None

        # for idx, cl in enumerate(range(len(self.class_dict.keys()))):
        for idx, cl in enumerate(sel_classes):
            train, test, unlbl = self._get_rand_partition(list(self.class_dict.keys())[cl], idx, k_per_class[idx])
            total_train = self._concat_or_identity(total_train, train)
            total_test = self._concat_or_identity(total_test, test)
            total_unlbl = self._concat_or_identity(total_unlbl, unlbl)


        train_label_real = []
        label_map_dict = dict((i, sel_classes[i]) for i in range(len(sel_classes)))
        train_label_real[:] = map(label_map_dict.get, total_train.labels[:])
        train_label_real = np.array(train_label_real)

        return total_train.data, total_train.labels, train_label_real




    def __getitem__(self, within_category=False, catcode=None):

        sel_classes = np.random.choice(
            range(len(self.class_dict.keys())),
            size=self.al_instance.n_class + self.al_instance.n_distractor,
            replace=False)
        k_per_class = [
            None
            for i in range(self.al_instance.n_class + self.al_instance.n_distractor)
        ]

        total_train = None
        total_test = None
        total_unlbl = None

        for idx, cl in enumerate(sel_classes[:self.al_instance.n_class]):
            train, test, unlbl = self._get_rand_partition(
              list(self.class_dict.keys())[cl], idx, k_per_class[idx])
            total_train = self._concat_or_identity(total_train, train)
            total_test = self._concat_or_identity(total_test, test)
            total_unlbl = self._concat_or_identity(total_unlbl, unlbl)

        for idx, cl in enumerate(sel_classes[self.al_instance.n_class:]):
            unlbl = self._get_rand_partition(
              list(self.class_dict.keys())[cl], self.al_instance.n_class + idx, k_per_class[idx])
            total_unlbl = self._concat_or_identity(total_unlbl, unlbl)

        assert self._check_shape(total_train, self.al_instance.n_class,
                                 self.al_instance.k_train)
        assert self._check_shape(total_test, self.al_instance.n_class,
                                 self.al_instance.k_test)
        assert self._check_shape(
            total_unlbl, self.al_instance.n_class + self.al_instance.n_distractor,
            self.al_instance.k_unlbl)

        train_label_real = []
        test_label_real = []
        unlbl_label_real = []
        label_map_dict = dict((i, sel_classes[i]) for i in range(len(sel_classes)))
        train_label_real[:] = map(label_map_dict.get, total_train.labels[:])
        test_label_real[:] = map(label_map_dict.get, total_test.labels[:])
        unlbl_label_real[:] = map(label_map_dict.get, total_unlbl.labels[:])

        train_label_real = np.array(train_label_real)
        test_label_real = np.array(test_label_real)
        unlbl_label_real = np.array(unlbl_label_real)


        # print(len(total_train.labels), len(total_test.labels), len(total_unlbl.labels))
        return total_train.data, total_train.labels, train_label_real, \
          total_test.data, total_test.labels, test_label_real, \
          total_unlbl.data, total_unlbl.labels, unlbl_label_real
        

    def _read_csv(self, csv_filename):

        class_dict = {}
        with open(csv_filename) as csv_file:
          csv_reader = csv.reader(csv_file)
          for (image_filename, class_name) in csv_reader:
            if 'label' not in class_name:
              if class_name in class_dict:
                class_dict[class_name].append(image_filename)
              else:
                class_dict[class_name] = [image_filename]
        """ convert dict: class_name -> {'lbl': [name of labeled class images], 'unlbl' : [name of unlabeled images]'} """
        new_class_dict = {}
        print('Seed!', self._seed)
        for class_name, image_list in class_dict.items():
          np.random.RandomState(self._seed).shuffle(image_list)
          new_class_dict[class_name] = {
              'lbl': image_list[0:self.n_lbl],
              'unlbl': image_list[self.n_lbl:]
          }

        return new_class_dict

    def split_label_unlabel(self, class_dict):
        splitfile = os.path.join(
            self._folder, "mini-imagenet-labelsplit-" + self._split +
            "-{:d}-{:d}.pkl".format(int(self._label_ratio * 100), self._seed))
        new_class_dict = {}
        for class_name, image_list in class_dict.items():
          np.random.RandomState(self._seed).shuffle(image_list)
          new_class_dict[class_name] = {
              'lbl': image_list[0:self.n_lbl],
              'unlbl': image_list[self.n_lbl:]
          }

        with open(splitfile, 'wb') as f:
          pkl.dump(new_class_dict, f, protocol=pkl.HIGHEST_PROTOCOL)
        return new_class_dict

    def _get_rand_partition(self, class_name, class_idx, k_unlbl=None):
        lbl_class_imgs = list(self.class_dict[class_name]['lbl'])
        unlbl_class_imgs = list(self.class_dict[class_name]['unlbl'])

        np.random.shuffle(lbl_class_imgs)
        np.random.shuffle(unlbl_class_imgs)

        train_end_idx = self.al_instance.k_train
        train = lbl_class_imgs[0:train_end_idx]

        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + self.al_instance.k_test
        test = lbl_class_imgs[test_start_idx:test_end_idx]

        # if unlabeled partition is not empty, get unlabeled images from there
        # otherwise, get from labeled partition
        if len(unlbl_class_imgs) > 0:
          unlbl_end_idx = 0 + (k_unlbl or self.al_instance.k_unlbl)
          unlbl = unlbl_class_imgs[:unlbl_end_idx]
        else:
          unlbl_start_idx = test_end_idx
          unlbl_end_idx = unlbl_start_idx + (k_unlbl or self.al_instance.k_unlbl)
          unlbl = lbl_class_imgs[unlbl_start_idx:unlbl_end_idx]

        train_and_test_data = [
            DatasetT(
                data=self._read_set(s),
                labels=np.full([len(s)], class_idx, dtype=np.int8),
            ) for s in [train, test]
        ]

        if class_idx < self.al_instance.n_class:
          return train_and_test_data + [
              DatasetT(
                  data=self._read_set(s),
                  labels=np.full([len(s)], 1, dtype=np.int8),
              ) for s in [unlbl]
          ]
        else:
          return DatasetT(
              data=self._read_set(unlbl),
              labels=np.full([len(unlbl)], 0, dtype=np.int8),
          )

    def _read_set(self, image_list):

        data = []
        for image_file in image_list:
          data.append(self._read_from_cache(image_file))
        if len(data) == 0:
          return np.zeros([0, 84, 84, 3])
        else:
          return np.stack(data)

    def _read_from_cache(self, idx):
        return self.img_data[idx] / 255.0

    def _concat_or_identity(self, big_set, small_set):
        if big_set is None:
          return small_set
        else:
          return DatasetT(
              data=np.concatenate((big_set.data, small_set.data)),
              labels=np.concatenate((big_set.labels, small_set.labels)))

    def _check_shape(self, dataset, n_class, n_items):
        assert dataset.data.shape == (n_class * n_items, self.n_input, self.n_input,
                                      3)
        assert dataset.labels.shape[0] == n_class * n_items

        return True

    @property
    def num_classes(self):
        return self._num_classes

    def get_size(self):
        return self._num_lbl

    def get_batch_idx(self, idx):
    # """Gets a fully supervised training batch for classification.

    # Returns: A tuple of
    #   x: Input image batch [N, H, W, C].
    #   y: Label class integer ID [N].
    # """
        return self._read_from_cache(self._lbl_idx[idx]), np.array(
            [self._cls_label[kk] for kk in self._lbl_idx[idx]], dtype=np.int64)

    def get_batch_idx_test(self, idx):
        """Gets the test set (unlabeled set) for the fully supervised training."""
        return self._read_from_cache(self._unlbl_idx[idx]), np.array(
            [self._cls_label[kk] for kk in self._unlbl_idx[idx]], dtype=np.int64)



# if __name__ == "__main__":

#     miniimage = MiniImagenet("./split", "test")
#     data = miniimage.__getitem__()
#     print(data)
