import os
import shutil

if __name__ == '__main__':
    file_list = os.listdir('./temp')
    file_list = sorted(file_list)
    backup_days = 15
    if len(file_list) > backup_days:
        for dir_path in file_list[:-backup_days]:
            if os.path.isdir(os.path.join('./temp', dir_path)):
                print('正在清理文件夹：%s' % os.path.join('./temp', dir_path))
                shutil.rmtree(os.path.join('./temp', dir_path))
    print('清理照片缓存结束。')
