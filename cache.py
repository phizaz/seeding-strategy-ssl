import os
import json

class StorageCache:
    def __init__(self, file):
        self.file = file

        if not self.has_file(file):
            self.create_file(file)

        with open(file) as file:
            try:
                self.new = False
                self.data = json.load(file)
            except ValueError:
                self.new = True
                self.data = {}

    def isnew(self):
        return self.new

    def has_file(self, file):
        return os.path.exists(file)

    def create_file(self, file):
        open(file, 'w').close()

    def get(self):
        return self.data

    def get_key(self, key):
        if not self.has(key):
            raise ValueError
        else:
            return self.data[key]

    def has(self, key):
        return key in self.data

    def update(self, data):
        self.data = data

    def update_key(self, key, data):
        self.data[key] = data

    def save(self):
        with open(self.file, 'w') as file:
            json.dump(self.data, file)