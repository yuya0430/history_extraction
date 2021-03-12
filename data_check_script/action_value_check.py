import json
import time
import sys
import glob
import re
from collections import Counter
import os

os.environ["OMP_NUM_THREADS"] = "1"
ACTION_LIST = ["INFORM", "REQUEST", "CONFIRM", "OFFER", "NOTIFY_SUCCESS", "NOTIFY_FAILURE", "INFORM_COUNT", "OFFER_INTENT", "REQ_MORE", "GOODBYE"]

def single_or_multi(train_file, dev_file, domain):
  if domain == "single":
    re_train_file = [s for s in train_file if re.match('.*0(0[1-9]|[1-3]\d|4[0-3]).*', s)]
    re_dev_file = [s for s in dev_file if re.match('.*00[1-7].*', s)]
    return re_train_file + re_dev_file
  
  elif domain == "multi":
    re_train_file = [s for s in train_file if re.match('.*(0(4[4-9]|[5-9]\d)|1\d\d).*', s)]
    re_dev_file = [s for s in dev_file if re.match('.*0(0[8-9]|[1-2]\d).*', s)]
    return re_train_file + re_dev_file
  else:
    return train_file + dev_file

def _get_state_update(current_state, prev_state):
    state_update = dict(current_state)
    for slot, values in current_state.items():
      if slot in prev_state and prev_state[slot][0] in values:
        # Remove the slot from state if its value didn't change.
        state_update.pop(slot)
    return state_update

def action_value_count(user_frames, system_frames, prev_states, value_list, update_count):
  states = {}
  system_act = []
  #各対話行為が与えるスロット値候補リストを作成
  for service, user_frame in user_frames.items():
    system_frame = system_frames.get(service, None)
    
    if system_frame != None:
      for action in system_frame["actions"]:
        if action["act"] not in system_act:
          system_act.append(action["act"]) 
    
      for action in system_frame["actions"]:
        for value in action["values"]:
          for sys_act in system_act:
            if value not in value_list[sys_act] and action["slot"] not in ["count","intent"]:
              value_list[sys_act].append(value)
  
  #各対話行為のスロット値候補が対話状態に追加されるかカウント
  for service, user_frame in user_frames.items():
    system_frame = system_frames.get(service, None)

    if "state" in user_frame:
      state = user_frame["state"]["slot_values"]
    else :
      state = system_frame["state"]["slot_values"]
    state_update = _get_state_update(state, prev_states.get(service, {})) #前の対話状態から追加もしくは変更されたスロット値を抽出
    for value in state_update.values():
      for v in value:
        for action, values in value_list.items():
          #print("values of {} : {}".format(action, values))
          if v in values:
            update_count[action] += 1
    states[service] = state

  return system_act, value_list, update_count, states

def main():
  domain = sys.argv[1]
  print("domain = {}".format(domain))
  train_file = glob.glob("data/train/dialogues_*.json")
  dev_file = glob.glob("data/dev/dialogues_*.json")
  all_file = single_or_multi(train_file, dev_file, domain)
  result_turn_action = {act: 0 for act in ACTION_LIST}
  result_value_count = {act: 0 for act in ACTION_LIST}
  result_update_count = {act: 0 for act in ACTION_LIST}
  dialogue_count = 0
  system_turn_count = 0
  for file in all_file:
    with open(file,'r') as f:
      json_data = json.load(f)
      for dialogue in json_data:
        dialogue_id = dialogue["dialogue_id"]
        dialogue_count += 1
        prev_states = {}
        turn_action = {act: 0 for act in ACTION_LIST}
        value_list = {act: [] for act in ACTION_LIST}
        value_count = {}
        update_count = {act: 0 for act in ACTION_LIST}
        for turn_idx, turn in enumerate(dialogue["turns"]):
          if turn["speaker"] == "USER":
            user_utterance = turn["utterance"]
            user_frames = {f["service"] : f for f in turn["frames"]}

            if turn_idx > 0:
              system_turn_count += 1
              system_turn = dialogue["turns"][turn_idx - 1]
              system_utterance = system_turn["utterance"]
              system_frames = {f["service"] : f for f in system_turn["frames"]}
            else:
              system_utterance = ""
              system_frames = {}

            system_act, value_list, update_count, prev_states = action_value_count(user_frames, system_frames, prev_states, value_list, update_count)

            for act in system_act:
              turn_action[act] += 1

        for action in dialogue["turns"][turn_idx]["frames"][0]["actions"]:
          if action["act"] == "GOODBYE":
            turn_action[action["act"]] += 1
            system_turn_count += 1
        for action, values in value_list.items():
          value_count[action] = len(values) 
        for action in ACTION_LIST:
          result_turn_action[action] += turn_action[action]
          result_value_count[action] += value_count[action]
          result_update_count[action] += update_count[action]
  
  print(dialogue_count)
  print(system_turn_count)
  print("result_turn_action : \n {}".format(result_turn_action))
  print("result_value_count : \n {}".format(result_value_count))
  print("result_update_count : \n {}".format(result_update_count))

  result_turn_action_ave = {}
  result_value_count_ave = {}
  result_update_count_ave = {}
  for action in ACTION_LIST:
    result_turn_action_ave[action] = result_turn_action[action] / dialogue_count
    result_value_count_ave[action] = result_value_count[action] / dialogue_count
    result_update_count_ave[action] = result_update_count[action] / dialogue_count

  print("ここから対話あたりの結果")
  print("\n\n<result_turn_action_ave>")
  for action, value in result_turn_action_ave.items():
    print("{} : {:f}".format(action, value))

  print("\n\n<result_value_count_ave>")
  for action, value in result_value_count_ave.items():
    print("{} : {:f}".format(action, value))

  print("\n\n<result_update_count_ave>")
  for action, value in result_update_count_ave.items():
    print("{} : {:f}".format(action, value))

if __name__=='__main__':
  t1 = time.time()
  main()
  t2 = time.time()

  keika = t2 - t1
  print("経過時間 : {}".format(keika))