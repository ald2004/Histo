from multiprocessing import Process, Queue, current_process,JoinableQueue, Value
import os, h5py, time
import numpy as np
import cv2
import queue  # imported for using queue.Empty exception
import threading
import multiprocessing as mp

train_img_dir = '/students/openDataSets/caner/train'
val_img_dir = '/students/openDataSets/caner/test'
root_dir = '/students/julyedu_465286'
h5filename = 'histcancer_mm.hdf5'
h5filename = os.path.join(root_dir, h5filename)
label_file = '/students/julyedu_465286/train_labels.csv'
total_images = 220025
total_images = 1000
label_file = '/students/julyedu_465286/train_labels.csv.small'
img_height = 96
img_width = 96
area_length = 32
area_width = 32
channels = 3
color_type = 'RGB'
labels = ['0', '1']  # good bad
data_order = 'tf'  # 'th' for Theano, 'tf' for Tensorflow

img_ext = '.tif'
test_ten = False
test_ten_line = 5
if data_order == 'th':
    train_shape = ((total_images), 3, img_height, img_width)
    val_shape = ((total_images), 3, img_height, img_width)
    test_shape = ((total_images), 3, img_height, img_width)
    label_shape = ((total_images),)
elif data_order == 'tf':
    train_shape = ((total_images), img_height, img_width, 3)
    val_shape = ((total_images), img_height, img_width, 3)
    test_shape = ((total_images), img_height, img_width, 3)
    label_shape = ((total_images))


class AtomicCounter:
    """An atomic, thread-safe incrementing counter.
    >>> counter = AtomicCounter()
    >>> counter.increment()
    1
    >>> counter.increment(4)
    5
    >>> counter = AtomicCounter(42.5)
    >>> counter.value
    42.5
    >>> counter.increment(0.5)
    43.0
    >>> counter = AtomicCounter()
    >>> def incrementor():
    ...     for i in range(100000):
    ...         counter.increment()
    >>> threads = []
    >>> for i in range(4):
    ...     thread = threading.Thread(target=incrementor)
    ...     thread.start()
    ...     threads.append(thread)
    >>> for thread in threads:
    ...     thread.join()
    >>> counter.value
    400000
    """

    def __init__(self, initial=-1):
        """Initialize a new atomic counter to given initial value (default 0)."""
        self.value = initial
        self._lock = threading.Lock()

    def increment(self, num=1):
        """Atomically increment the counter by num (default 1) and return the
        new value.
        """
        with self._lock:
            self.value += num
            return self.value


counter = Value('i', 0)


def source():
    xx = open(label_file, 'r')
    xx.readline()
    return [x.strip() for x in xx.readlines()]




def worker():
    while True:
        try :
            item = tasks_to_accomplish.get()
        except queue.Empty:
            break
        else:
            if item is None:
                break
            # print(item,current_process().name)
            print('.', end='')
            # print(current_process().name)
            with counter.get_lock():
                realprocessimg = counter.value
                counter.value += 1
            imgname, label = item.split(',')
            addr = os.path.join(train_img_dir, imgname + img_ext)
            if not os.path.exists(addr): return
            # print(addr)
            img = cv2.imread(addr)
            assert img.size == img_height * img_width * channels
            if train_shape[1::] != img.shape:
                img = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # tasks_that_are_done.put([img, label, imgname])
            # print(f'write {realprocessimg} imgs...')
            hdf5_file['train_images'][realprocessimg, ...] = img[None]
            hdf5_file["train_labels"][realprocessimg, ...] = np.array([0, 1] if labels[1] == label else [1, 0], dtype='S')
            hdf5_file["train_img_names"][realprocessimg, ...] = np.array(imgname, dtype='S')
            hdf5_file.flush()
            tasks_to_accomplish.task_done()
            if realprocessimg % 100 == 0 and realprocessimg > 1:
                print(f'\tgen Train data: {realprocessimg}/{total_images}')
            # print('.', end=' ')
            # qq.task_done()


tasks_to_accomplish = JoinableQueue()
# tasks_that_are_done = JoinableQueue()
processes = []
number_of_processes = 6

if os.path.exists(h5filename):
    os.remove(h5filename)
    print(f'h5 file :{h5filename} removed !!!')
hdf5_file = h5py.File(h5filename, mode='w', driver='core')
hdf5_file.create_dataset('train_images', train_shape, dtype=np.uint8, compression='lzf')
hdf5_file.create_dataset('train_labels', shape=(total_images, len(labels)), maxshape=(None, 2), dtype="S10",
                         compression='lzf')
hdf5_file.create_dataset('train_img_names', shape=(total_images, len('8c82ae834697bc55a742cc6001f29ace30e46d9a')),
                         maxshape=(None, 40), dtype="S10",
                         compression='lzf')
for _ in range(number_of_processes):
    p = Process(target=worker, args=())
    p.start()
    processes.append(p)
print('started !')
for item in source():
    tasks_to_accomplish.put(item)
for i in range(number_of_processes):
    tasks_to_accomplish.put(None)
while not tasks_to_accomplish.empty():
    print(f'sleep.5 and size if {tasks_to_accomplish.qsize()}')
    time.sleep(.5)
print('queue is empty !')
for p in processes:
    p.join()
    print(f'process {p.name} joined !!!')
hdf5_file.flush()
hdf5_file.close()
