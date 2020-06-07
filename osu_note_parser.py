import math
import json
import os

import pandas as pd

from my_utils.note_embedding import note_map_new

PATH ='C:/osu!/Songs/'

json_file = './beatmap_list.json'

def key_value(str_value, column_count=4):
    number = int(str_value)

    return math.floor(number * column_count // 512)


with open('./beatmap_list.json') as json_file:
    json_data = json.load(json_file)

processed_list = []
for cur_song in json_data:
    file_name = cur_song['file']
    offset = cur_song['offset']
    beat = cur_song['beat']

    passed_diff = []
    for cur_diff in cur_song['osu list']:
        print('\n' + cur_diff)
        exe_file = PATH + '/' + file_name + '/' + cur_diff

        with open(exe_file, "r", encoding="UTF-8") as f:
            txt = f.readline()

            while txt.find('Mode:') < 0:
                txt = f.readline()

            mode = txt
            #print(txt)
            if mode[len(mode) - 2] != '3':
                print("IT IS NOT MANIA (SKIP)")
                continue

            while txt.find("[HitObjects]") < 0:
                txt = f.readline()

            lines = f.readlines()

        fine_lines = []

        for line in lines:
            line = line.strip().split(',')
            fine_lines.append(line)

        sixteenth_note = beat / 4

        note_dicts = {}

        for line in fine_lines:
            time = int(line[2]) - offset

            if not(time in note_dicts):
                note_dicts[time] = [0, 0, 0, 0]

            long_note_time = line[5].split(":")
            long_note_time = int(long_note_time[0]) - offset
            note_value = key_value(line[0])

            if long_note_time > 0:
                if not (long_note_time in note_dicts):
                    note_dicts[long_note_time] = [0, 0, 0, 0]
                note_dicts[time][note_value] = 2
                note_dicts[long_note_time][note_value] = 3
            else:
                note_dicts[time][note_value] = 1

        note_time_stamp = list(note_dicts.keys())
        note_time_stamp.sort()

        fail_flag = False
        sum_time = 0
        while len(note_time_stamp) != 0:
            slice_time = (note_time_stamp[0] - sum_time)

            if slice_time > (sixteenth_note / 2):
                note_dicts[int(sum_time)] = [0, 0, 0, 0]
            elif slice_time < -(sixteenth_note / 2):
                fail_flag = True
                print("CAN NOT HANDLE BEAT")
                break
            else:
                note_time_stamp.pop(0)

            sum_time += sixteenth_note

        if fail_flag:
            continue

        note_time_stamp = list(note_dicts.keys())
        note_time_stamp.sort()

        #plt.plot(note_time_stamp)

        id2note, note2id = note_map_new(4)

        id_notes = []

        for stamp in note_time_stamp:
            note_str = ""
            for j in range(0, 4):
                note_str += str(note_dicts[stamp][j])

            id_notes.append([stamp, note2id[note_str], note_dicts[stamp][0], note_dicts[stamp][1], note_dicts[stamp][2], note_dicts[stamp][3]])

        if not os.path.exists('./note_csv/' + file_name):
            os.mkdir('./note_csv/' + file_name)
        passed_diff.append(cur_diff)
        df = pd.DataFrame(id_notes)
        df.to_csv('./note_csv/' + file_name + '/' + cur_diff + '.csv', index=False, encoding='utf-8', header=['time', 'note_id', 'note_1', 'note_2', 'note_3', 'note_4'])

    if len(passed_diff) == 0:
        continue

    item = {"title": file_name, "beat": beat, "diff_list": passed_diff}
    processed_list.append(item)

processed_list = {"processed_list": processed_list}

with open("processed_list.json", "w", encoding="UTF-8") as f:
    f.write(json.dumps(processed_list))

diff_list = []
with open("compact_processed_diff.txt", "w", encoding="UTF-8") as f:
    for item in processed_list["processed_list"]:
        for diff in item["diff_list"]:
            f.write(item['title'] + "/" + diff + "\n")
