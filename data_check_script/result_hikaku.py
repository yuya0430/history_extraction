import json
import sys
import glob
import re
from collections import Counter
import os

os.environ["OMP_NUM_THREADS"] = "1"
ACTION_LIST = ["INFORM", "REQUEST", "CONFIRM", "OFFER", "NOTIFY_SUCCESS", "NOTIFY_FAILURE", "INFORM_COUNT", "OFFER_INTENT", "REQ_MORE", "GOODBYE"]


def main():
  output_file1 = "output_offer_abci/pred_res_206470/dialogues_and_metrics.json"
  output_file2 = "output_24_abci/pred_res_275293/dialogues_and_metrics.json"
  with open(output_file1,'r') as f1:
    with open(output_file2, 'r') as f2:
      json_data1 = json.load(f1)
      json_data2 = json.load(f2)
      for dialogue_id, dialogue1 in json_data1.items():
        dialogue2 = json_data2[dialogue_id]
        prev_states = {}
        for turn_idx, turn1 in enumerate(dialogue1["turns"]):
          if turn1["speaker"] == "USER":
            turn2 = dialogue2["turns"][turn_idx]
            user_utterance = turn1["utterance"]
            user_frames1 = {f["service"] : f for f in turn1["frames"]}
            user_frames2 = {f["service"] : f for f in turn2["frames"]}

            if turn_idx > 0:
              system_turn1 = dialogue1["turns"][turn_idx - 1]
              system_turn2 = dialogue2["turns"][turn_idx - 1]
              system_utterance = system_turn1["utterance"]
              system_frames1 = {f["service"] : f for f in system_turn1["frames"]}
              system_frames2 = {f["service"] : f for f in system_turn2["frames"]}

            else:
              system_utterance = ""
              system_frames1 = {}
              system_frames2 = {}

            for service, frame1 in user_frames1.items():
              frame2 = user_frames2[service]
              state1 = frame1["state"]
              state2 = frame2["state"]
              flag = 0
              fla = 1
              if service == "Restaurants_2":
                if state1["active_intent"] != state2["active_intent"]:
                  fla = 1
                if set(state1["requested_slots"]) != set(state2["requested_slots"]):
                  fla = 1
                slot_values1 = state1["slot_values"]
                slot_values2 = state2["slot_values"]
                if set(slot_values1.keys()) != set(slot_values2.keys()):
                  fla = 1
                else:
                  for key, values1 in slot_values1.items():
                    values2 = slot_values2[key]
                    if set(values1) != set(values2):
                      flag = 1
                      break
                if flag == 1:
                  print("dialogue_id : {}".format(dialogue_id))
                  print("turn_idx : {}".format(turn_idx))
                  print("user_utterance : {}".format(user_utterance))
                  print("state1 : {}".format(state1))
                  print("state2 : {}".format(state2))
                  input()

        
if __name__=='__main__':
    main()
