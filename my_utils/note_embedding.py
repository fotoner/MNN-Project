from itertools import product


def note_map(note_count):
    per1 = product(([0, 1, 2, 3]), repeat=note_count)
    per1 = list(per1)

    id2note = {}
    note2id = {}

    for i in range(0, len(per1)):
        note_str = ""
        for j in range(0, note_count):
            note_str += str(per1[i][j])

        id2note[i] = note_str
        note2id[note_str] = i

    return id2note, note2id


