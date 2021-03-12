import json
import glob
import re
from collections import Counter

def main():
    all_file = glob.glob("data/dev/dialogues_*.json")
    #re_file = [s for s in all_file if re.match('.*0(0[1-9]|[1-3]\d|4[0-3]).*', s)] #train
    re_file = [s for s in all_file if re.match('.*00[1-7].*', s)] #dev
    # train 001~043 , dev 001~007
    count = 0
    frame_list = []
    speaker_list = []
    for file in re_file:
        with open(file,'r') as f:
            json_data = json.load(f)
            for i in range(len(json_data)):
                for j in range(len(json_data[i]["turns"])):
                    s = []
                    for k in range(len(json_data[i]["turns"][j]["frames"][0]["slots"])):
                        s.append(json_data[i]["turns"][j]["frames"][0]["slots"][k]["slot"])
                        mycounter = Counter(s)
                        for key, co in mycounter.items():
                            if co > 1 :
                                count = count + 1
                                frame_list.append(json_data[i]["turns"][j]["frames"][0])
                                speaker_list.append(json_data[i]["turns"][j]["speaker"])



    with open("data/dev/slot_dupli_list.txt",'w') as f:
        f.write("count : " + str(count) + "\n\n\n")
        for i in range(len(frame_list)):
            f.write("frame : [ \n \"slots\": [ \n")
            for j in range(len(frame_list[i]["slots"])):
                f.write("{"+"\n \"slot\" : \"{}\" \n ".format(frame_list[i]["slots"][j]["slot"])+"} \n")

            f.write("] \n \"service\" : \"{}\" \n ] \n".format(frame_list[i]["service"]))
            f.write("\"speaker\" : {} \n\n".format(speaker_list[i]))



if __name__=='__main__':
    main()
