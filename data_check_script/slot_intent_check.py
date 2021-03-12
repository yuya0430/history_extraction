import json
from collections import Counter

def main():
    s = []
    service_names = []
    all_schema = {}
    with open("data/train/schema.json",'r') as f:
        json_data = json.load(f)
        for schema in json_data:
            service_name = schema["service_name"]
            service_names.append(service_name)
            slot_name = []
            slot_desc = {}
            slot_poss = {}
            intent_name = []
            intent_desc = {}
            intent_requ = {}
            for slot in schema["slots"]:
                slot_name.append(slot["name"])
                slot_desc[slot["name"]] = (slot["description"])
                slot_poss[slot["name"]] = slot["possible_values"]
            for intent in schema["intents"]:
                intent_name.append(intent["name"])
                intent_desc[intent["name"]] = (intent["description"])
                intent_requ[intent["name"]] = (intent["required_slots"])

            """print("サービス名: {}\n".format(service_name))
            print("slot_name : {}\n".format(slot_name))
            print("slot_desc : {}\n".format(slot_desc))
            print("slot_poss : {}\n".format(slot_poss))
            print("intent_name : {}\n".format(intent_name))
            print("intent_desc : {}\n".format(intent_desc))
            print("intent_requ : {}\n".format(intent_requ))"""
            all_schema[service_name] = {"slot_name" : slot_name, "slot_len" : len(slot_name), "slot_desc" : slot_desc, "slot_poss" : slot_poss, "intent_name" : intent_name, "intent_len": len(intent_name), "intent_desc" : intent_desc, "intent_requ" : intent_requ}

    
    with open("data/dev/schema.json",'r') as f:
        json_data = json.load(f)
        for schema in json_data:
            service_name = schema["service_name"]
            service_names.append(service_name)
            slot_name = []
            slot_desc = {}
            slot_poss = {}
            intent_name = []
            intent_desc = {}
            intent_requ = {}
            for slot in schema["slots"]:
                slot_name.append(slot["name"])
                slot_desc[slot["name"]] = (slot["description"])
                slot_poss[slot["name"]] = slot["possible_values"]
            for intent in schema["intents"]:
                intent_name.append(intent["name"])
                intent_desc[intent["name"]] = (intent["description"])
                intent_requ[intent["name"]] = (intent["required_slots"])

            """print("サービス名: {}\n".format(service_name))
            print("slot_name : {}\n".format(slot_name))
            print("slot_desc : {}\n".format(slot_desc))
            print("slot_poss : {}\n".format(slot_poss))
            print("intent_name : {}\n".format(intent_name))
            print("intent_desc : {}\n".format(intent_desc))
            print("intent_requ : {}\n".format(intent_requ))"""
            all_schema[service_name] = {"slot_name" : slot_name, "slot_len" : len(slot_name), "slot_desc" : slot_desc, "slot_poss" : slot_poss, "intent_name" : intent_name, "intent_len": len(intent_name), "intent_desc" : intent_desc, "intent_requ" : intent_requ}

    with open("schema.tex",'w') as f:
        service_names.sort()
        for service_name in service_names:
            schema = all_schema[service_name]
            slot_len = schema["slot_len"]
            intent_len = schema["intent_len"]
            for sidx, slot_name in enumerate(schema["slot_name"]):
                if sidx >= slot_len-1:
                    f.write("& & {} & {} & {} \\\\ \cline{{2-5}} \n".format(slot_name, schema["slot_desc"][slot_name], schema["slot_poss"][slot_name]))
                elif sidx > 0:
                    f.write("& & {} & {} & {} \\\\ \cline{{3-5}} \n".format(slot_name, schema["slot_desc"][slot_name], schema["slot_poss"][slot_name]))
                else:
                    f.write("\multirow{{{}}}{{*}}{{{}}} & \multirow{{{}}}{{*}}{{スロット}} & {} & {} & {} \\\\ \cline{{3-5}} \n".format(slot_len + intent_len, service_name, slot_len, slot_name, schema["slot_desc"][slot_name], schema["slot_poss"][slot_name]))
            for iidx, intent_name in enumerate(schema["intent_name"]):
                if iidx >= intent_len-1:
                    f.write("& & {} & {} & {} \\\\ \cline{{1-5}} \n".format(intent_name, schema["intent_desc"][intent_name], schema["intent_requ"][intent_name]))
                elif iidx > 0:
                    f.write("& & {} & {} & {} \\\\ \cline{{3-5}} \n".format(intent_name, schema["intent_desc"][intent_name], schema["intent_requ"][intent_name]))
                else:
                    f.write("& \multirow{{{}}}{{*}}{{インテント}} & {} & {} & {} \\\\ \cline{{3-5}} \n".format(intent_len, intent_name, schema["intent_desc"][intent_name], schema["intent_requ"][intent_name]))
                    

if __name__=='__main__':
    main()
