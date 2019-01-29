import json

def save_to_file(object, file_name):
    str = json.dumps(object)
    f = open(file_name, "w")
    f.write(str)
    f.flush()
    f.close()

def load_from_file(file_name):
    f = open(file_name)
    str = f.read()
    f.close()
    return json.loads(str)
