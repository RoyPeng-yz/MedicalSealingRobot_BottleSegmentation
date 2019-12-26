# -*- coding:utf8 -*-

import os


def num_to_str_6(num):
    s = str(num)
    n = len(s)
    i = 0
    for i in range(6 - n):
        s = '0' + s
    return s


class BatchRename():

    def __init__(self, file_path, initial_suffix, to_suffix):

        self.path = file_path
        self.file_suffix=initial_suffix
        self.to_suffix=to_suffix

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 1  # 设置标号
        for item in filelist:
            if item.endswith(self.file_suffix):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), num_to_str_6(i) +  self.to_suffix)
                try:
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
                print('total %d to rename & converted %d jpgs' % (total_num, i-1))

# if __name__ == '__main__':
#     demo = BatchRename()
#     demo.rename()
