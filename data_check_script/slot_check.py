import json
from collections import Counter

def main():
    s = []
    with open("data/train/schema.json",'r') as f:
        json_data = json.load(f)

        for i in range(len(json_data)):
            for j in range(len(json_data[i]["slots"])):
                s.append(json_data[i]["slots"][j]["name"])
        mycounter = Counter(s)

    c = 0
    with open("data/train/slot_list.txt",'w') as f:
        for key, count in mycounter.items():
            c += count
            w = key + " : " + str(count) + "\n"
            f.write(w)
        f.write('合計 : {}'.format(c))

if __name__=='__main__':
    main()
