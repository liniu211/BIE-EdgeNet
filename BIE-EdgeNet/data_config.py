
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = ""
    def get_data_config(self, data_name):
        self.data_name = data_name
        # 仓库内置 CDCD 演示数据（3 对影像 + label + label_edge），位于 ./CDCD
        if data_name == 'CDCD':
            self.label_transform = "norm"
            self.root_dir = r'./CDCD'
        elif data_name == 'LEVIR-256-edge':
            self.label_transform = "norm"
            self.root_dir = r'E:\zyh\Data\LEVIR-CD\merged_data'
        elif data_name == 'LEVIR-512-edge':
            self.label_transform = "norm"
            self.root_dir = r'E:\zyh\Data\LEVIR-CD\512merged_data'
        elif data_name == 'LEVIR-1024-edge':
            self.label_transform = "norm"
            self.root_dir = r'E:\zyh\Data\LEVIR-CD\1024merged_data'
        elif data_name == 'WHU-256-edge':
            self.label_transform = "norm"
            self.root_dir = r'E:\zyh\Data\Building change detection dataset\WHU_256'
        elif data_name == 'WHU-512-edge':
            self.label_transform = "norm"
            self.root_dir = r'E:\zyh\Data\Building change detection dataset\WHU_512'
        elif data_name == 'WHU-1024-edge':
            self.label_transform = "norm"
            self.root_dir = r'E:\zyh\Data\Building change detection dataset\WHU_1024'
        elif data_name == 'CDCD-256-edge-7-1-2':
            self.label_transform = "norm"
            self.root_dir = r'E:\zyh\Data\CDCD\CDCD_256_7_1_2'
        elif data_name == 'CDCD-repair-7-1-2':
            self.label_transform = "norm"
            self.root_dir = r'E:\zyh\Data\CDCD\CDCD_repair_712'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self

