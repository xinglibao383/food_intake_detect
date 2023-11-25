import os
import yaml


class Logger:
    def __init__(self, mode, logs_file_save_path, config_file_save_path):
        self.mode = mode
        self.logs_file_save_path = logs_file_save_path
        self.config_file_save_path = config_file_save_path

        self.copy_yaml_file("./config.yaml", self.config_file_save_path)

    def copy_yaml_file(self, source_path, destination_path):
        """拷贝yaml文件"""
        if not os.path.exists(destination_path):
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        with open(source_path, 'r') as source_file:
            data = yaml.safe_load(source_file)

        with open(destination_path, 'w') as destination_file:
            yaml.dump(data, destination_file, default_flow_style=False)

    def record_logs(self, logs):
        if self.mode == "console":
            self.record_logs_console(logs)
        elif self.mode == "file":
            self.record_logs_file(logs)
        else:
            self.record_logs_tensorboard(logs)

    def record_logs_file(self, logs):
        with open(self.logs_file_save_path, "a") as log_file:
            for log in logs:
                log_file.write(log + "\n")

    def record_logs_console(self, logs):
        for log in logs:
            print(log)

    def record_logs_tensorboard(self, logs):
        # TODO
        pass
