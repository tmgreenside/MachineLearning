import settings, os
import enum

class Logger:
    def __init__(self, path, filename):
        self.path = path
        i = 0
        while True:
            if not os.path.exists(path + filename.format(i)):
                self.filename = filename.format(i)
                break
            else:
                i += 1
        print(self.filename)

    def log(self, string):
        with open(self.path + self.filename, "a") as log_file:
            log_file.write(string + "\n")
        print(string)

logger = Logger(settings.log_file_location, \
        settings.log_file_name)
