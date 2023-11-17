import yaml
from utils import load_config

class Logger:
    def __init__(self, mode, logs_file_save_path):
        self.mode = mode
        self.logs_file_save_path = logs_file_save_path

        self.record_model_params()


    def record_model_params(self):
        config = load_config.load_config_yaml()

        with open(self.logs_file_save_path, 'a') as file:
            # 使用缩进和换行使内容更美观
            file.write(yaml.dump(config, default_flow_style=False, indent=2))
            file.write("\n")


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