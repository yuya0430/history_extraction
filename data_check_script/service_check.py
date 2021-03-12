import json
from collections import Counter

def main():
    train_service_names = []
    with open("data/train/schema.json",'r') as f:
        schemas = json.load(f)
        for schema in schemas:
            train_service_names.append(schema["service_name"])
    train_service_names.sort()

    dev_service_names = []
    with open("data/dev/schema.json",'r') as f:
        schemas = json.load(f)
        for schema in schemas:
            dev_service_names.append(schema["service_name"])
    dev_service_names.sort()

    test_service_names = []
    with open("data/test/schema.json",'r') as f:
        schemas = json.load(f)
        for schema in schemas:
            test_service_names.append(schema["service_name"])
    test_service_names.sort()

    all_service_names = list(set(train_service_names + dev_service_names + test_service_names))
    all_service_names.sort()
    print(all_service_names)
    print("長さ:{}\n\n".format(len(all_service_names)))

    print("train:\n{}".format(train_service_names))
    print("長さ:{}\n\n".format(len(train_service_names)))
    print("dev:\n{}".format(dev_service_names))
    print("長さ:{}\n\n".format(len(dev_service_names)))
    print("test:\n{}".format(test_service_names))
    print("長さ:{}\n\n".format(len(test_service_names)))

    dev_train = list(set(dev_service_names) - set(train_service_names))
    dev_train.sort()
    test_dev_train = list(set(test_service_names) - set(train_service_names + dev_service_names))
    test_dev_train.sort()
    print("dev-train:\n{}".format(dev_train))
    print("長さ:{}\n\n".format(len(dev_train)))
    print("test-(train+dev):\n{}".format(test_dev_train))
    print("長さ:{}\n\n".format(len(test_dev_train)))

if __name__=='__main__':
    main()
