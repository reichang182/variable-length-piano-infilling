import utils
import pickle
import os
import sys
import collections
import numpy as np
import miditoolkit
from fractions import Fraction
import glob
import random
import copy
import statistics

import argparse

parser = argparse.ArgumentParser(description='')

# training setup
parser.add_argument('--midi-folder', type=str, default='datasets/midi/midi_synchronized', help="Folder containing the midi files.")
parser.add_argument('--save-folder', type=str, default='./', help="Folder to save worded_data and dictionary.")
args = parser.parse_args()

GroupEvent = collections.namedtuple('GroupEvent', ['Tempo', 'Bar', 'Position', 'Pitch', 'Duration', 'Velocity'])
tempo_quantize_step = 4

def extract_events(input_path):
    note_items, tempo_items = utils.read_items(input_path)
    note_items = utils.quantize_items(note_items)
    max_time = note_items[-1].end
    items = tempo_items + note_items
    groups = utils.group_items(items, max_time)
    events = utils.item2event(groups)
    return events

def convert_to_tuple_events(events, tempo_items):
    # `events` should be a list containing events like below:
    # Event(name=Bar, time=None, value=None, text=1)
    # Event(name=Position, time=1440, value=13/16, text=1440)
    # Event(name=Note Velocity, time=1440, value=10, text=42/40)
    # Event(name=Note On, time=1440, value=61, text=61)
    # Event(name=Note Duration, time=1440, value=28, text=1727/1740)
    # Event(name=Position, time=1440, value=13/16, text=1440)
    # Event(name=Note Velocity, time=1440, value=14, text=56/56)
    # Event(name=Note On, time=1440, value=77, text=77)
    # Event(name=Note Duration, time=1440, value=7, text=476/480)
    # Event(name=Position, time=1680, value=15/16, text=1680)
    # Event(name=Note Velocity, time=1680, value=12, text=48/48)
    # Event(name=Note On, time=1680, value=68, text=68)
    # Event(name=Note Duration, time=1680, value=6, text=447/420)
    # Event(name=Bar, time=None, value=None, text=2)
    # Event(name=Position, time=1920, value=1/16, text=1920)
    # Event(name=Note Velocity, time=1920, value=15, text=62/60)
    # Event(name=Note On, time=1920, value=77, text=77)
    # Event(name=Note Duration, time=1920, value=7, text=472/480)
    # Event(name=Position, time=2160, value=3/16, text=2160)
    # Event(name=Note Velocity, time=2160, value=12, text=48/48)
    group_events = []
    group_event_template = {'Position': None, 'Pitch': None, 'Duration': None, 'Velocity': None}
    group_event = group_event_template.copy()
    bar_value = None
    tempo = 1
    tempo_class = None
    # print("=" * 70)
    # print(*events, sep='\n')
    # print('~~~'*30)
    # print(*tempo_items, sep='\n')
    # print("=" * 70)
    for i, event in enumerate(events):
        if event.name == 'Bar':
            # if not all(v == None for v in group_event.values()):
            #     raise Exception("Encounter a 'Bar' item while having a not completed note: {}".format(group_event))
            # else:
            #     bar_value = int(event.text)
            bar_value = int(event.text)
        elif event.name == 'Tempo Value':
            tempo = event.value
        elif event.name == 'Tempo Class':
            tempo_class = event.value
        else:
            # if group_event[event.name] is not None:
            #     raise Exception('Double entry for %s at index %d' % (event.name, i))
            # else:
            #     group_event[event.name] = event.value
            group_event[event.name] = event.value

        if None not in group_event.values():
            # all fields are filled
            if tempo_class == 'slow':
                cur_tempo = utils.DEFAULT_TEMPO_INTERVALS[0].start + tempo
            elif tempo_class == 'mid':
                cur_tempo = utils.DEFAULT_TEMPO_INTERVALS[1].start + tempo
            elif tempo_class == 'fast':
                cur_tempo = utils.DEFAULT_TEMPO_INTERVALS[2].start + tempo
            else:
                raise Exception("Undefined tempo class: %s" % tempo_class)
            group_event['Bar'] = bar_value
            group_event['Tempo'] = cur_tempo
            group_events.append(GroupEvent(**group_event))
            group_event = group_event_template.copy()

    # if not all(v == None for v in group_event.values()):
    #     raise Exception("Uncomplete note at the end of event: {}".format(group_event))

    return group_events

def item2event(groups):
    events = []
    n_downbeat = 0
    Event = utils.Event
    for i in range(len(groups)):
        # if 'Note' not in [item.name for item in groups[i][1:-1]]:
        #     continue
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        events.append(Event(
            name='Bar',
            time=None,
            value=None,
            text='{}'.format(n_downbeat)))
        for item in groups[i][1:-1]:
            # position
            flags = np.linspace(bar_st, bar_et, utils.DEFAULT_FRACTION, endpoint=False)
            index = np.argmin(abs(flags-item.start))
            events.append(Event(
                name='Position',
                time=item.start,
                # value='{}/{}'.format(index+1, utils.DEFAULT_FRACTION),
                value='{}/{}'.format(index, utils.DEFAULT_FRACTION),
                text='{}'.format(item.start)))
            if item.name == 'Note':
                # velocity
                velocity_index = np.searchsorted(
                    utils.DEFAULT_VELOCITY_BINS,
                    item.velocity,
                    side='right') - 1
                events.append(Event(
                    name='Velocity',
                    time=item.start,
                    value=velocity_index,
                    text='{}/{}'.format(item.velocity, utils.DEFAULT_VELOCITY_BINS[velocity_index])))
                # pitch
                events.append(Event(
                    name='Pitch',
                    time=item.start,
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
                # duration
                duration = item.end - item.start
                index = np.argmin(abs(utils.DEFAULT_DURATION_BINS-duration))
                events.append(Event(
                    name='Duration',
                    time=item.start,
                    value=index,
                    text='{}/{}'.format(duration, utils.DEFAULT_DURATION_BINS[index])))
            elif item.name == 'Chord':
                events.append(Event(
                    name='Chord',
                    time=item.start,
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
            elif item.name == 'Tempo':
                tempo = item.pitch
                if tempo in utils.DEFAULT_TEMPO_INTERVALS[0]:
                    tempo_style = Event('Tempo Class', item.start, 'slow', None)
                    tempo_value = Event('Tempo Value', item.start,
                        tempo-utils.DEFAULT_TEMPO_INTERVALS[0].start, None)
                elif tempo in utils.DEFAULT_TEMPO_INTERVALS[1]:
                    tempo_style = Event('Tempo Class', item.start, 'mid', None)
                    tempo_value = Event('Tempo Value', item.start,
                        tempo-utils.DEFAULT_TEMPO_INTERVALS[1].start, None)
                elif tempo in utils.DEFAULT_TEMPO_INTERVALS[2]:
                    tempo_style = Event('Tempo Class', item.start, 'fast', None)
                    tempo_value = Event('Tempo Value', item.start,
                        tempo-utils.DEFAULT_TEMPO_INTERVALS[2].start, None)
                elif tempo < utils.DEFAULT_TEMPO_INTERVALS[0].start:
                    tempo_style = Event('Tempo Class', item.start, 'slow', None)
                    tempo_value = Event('Tempo Value', item.start, 0, None)
                elif tempo >= utils.DEFAULT_TEMPO_INTERVALS[2].stop:
                    tempo_style = Event('Tempo Class', item.start, 'fast', None)
                    tempo_value = Event('Tempo Value', item.start, 59, None)
                events.append(tempo_style)
                events.append(tempo_value)
    return events

# extract events: each event is a tuple (Bar, Position, Pitch, Duration, Velocity)
def extract_tuple_events(input_path):
    note_items, tempo_items = utils.read_items(input_path)
    note_items = note_items[0] # assume there is only 1 track, so this get the first track
    note_items = utils.quantize_items(note_items)
    max_time = note_items[-1].end
    items = tempo_items + note_items
    groups = utils.group_items(items, max_time)
    # events = utils.item2event(groups)
    events = item2event(groups)
    # print(*events, sep='\n')
    events = convert_to_tuple_events(events, tempo_items)
    return events

def tuple_events_to_midi(events, save_path, tick_resolution=utils.DEFAULT_RESOLUTION):
    midi = miditoolkit.midi.parser.MidiFile()
    # Notes
    notes = []
    tempo_changes = []
    prev_tempo = None
    for e in events:
        # velocity = e.Velocity * 4
        velocity = int(utils.DEFAULT_VELOCITY_BINS[e.Velocity])
        pitch = e.Pitch
        ticks_per_bar = tick_resolution * 4
        st = int(int(e.Bar) * ticks_per_bar + (Fraction(e.Position) * ticks_per_bar))
        # et = st + e.Duration * 60
        et = st + int(utils.DEFAULT_DURATION_BINS[e.Duration])
        notes.append(miditoolkit.Note(velocity, pitch, st, et))

        if e.Tempo != prev_tempo:
            prev_tempo = e.Tempo
            tempo_changes.append(miditoolkit.midi.containers.TempoChange(e.Tempo, st))

    midi.ticks_per_beat = tick_resolution
    # write instrument
    inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
    inst.notes = notes
    midi.instruments.append(inst)

    # temp_tempos.append(miditoolkit.midi.containers.TempoChange(bpm, st))
    # midi.tempo_changes = temp_tempos

    # Tempo changes
    midi.tempo_changes = tempo_changes

    midi.dump(save_path)

def group_by_bar(events):
    bar = None
    grouped_events = [] # Events grouped by bar [[events of bar0], [events of bar1], [events of bar2], ...]
    for e in events:
        if bar != e.Bar:
            bar = e.Bar
            grouped_events.append([])
        grouped_events[-1].append(e)

    return grouped_events

def construct_dict(save_dict_path):
    event2word = {}
    word2event = {}

    for etype in ['Tempo', 'Bar', 'Position', 'Pitch', 'Duration', 'Velocity']:
        count = 0
        e2w = {}

        # Tempo 30 ~ 210
        if etype == 'Tempo':
            for i in range(28, 211, tempo_quantize_step):
                e2w['Tempo %d' % i] = count
                count += 1

        # Bar 0 ~ 15
        elif etype == 'Bar':
            for i in range(16):
                e2w['Bar %d' % i] = count
                count += 1

        # Position: 0/16 ~ 15/16
        elif etype == 'Position':
            for i in range(0, 16):
                e2w['Position %d/16' % i] = count
                count += 1

        # Pitch: 22 ~ 107
        elif etype == 'Pitch':
            for i in range(22, 108):
                e2w['Pitch %d' % i] = count
                count += 1

        # Duration: 0 ~ 63
        elif etype == 'Duration':
            for i in range(64):
                e2w['Duration %d' % i] = count
                count += 1

        # Velocity: 0 ~ 21
        elif etype == 'Velocity':
            for i in range(22):
                e2w['Velocity %d' % i] = count
                count += 1

        else:
            raise Exception('etype error')


        e2w['%s <BOS>' % etype] = count
        count += 1
        e2w['%s <EOS>' % etype] = count
        count += 1
        e2w['%s <PAD>' % etype] = count
        count += 1

        event2word[etype] = e2w
        word2event[etype] = {e2w[key]: key for key in e2w}

    print(event2word)
    print("=" * 80)
    print(word2event)

    with open(save_dict_path, 'wb') as f:
        pickle.dump([event2word, word2event], f, protocol=pickle.HIGHEST_PROTOCOL)


def load_tuple_event(files=None):
    data = {'train':[], 'evaluation': []}
    if files == None:
        files = glob.glob(os.path.join(data_path, '*.midi'))
    for data_segment in ['train', 'evaluation']:
        for midifile in files:
            events = extract_tuple_events(midifile)
            events = group_by_bar(events)   # shape of events: [n_bars, n_notes_per_bar]
            # TODO song too long?
            data[data_segment].append(events)

        print('number of midi for %s: %d' % (data_segment, len(data[data_segment])))


    return data

def tuple_event_to_word(data, dict_file=None, save_path=None):
    with open(dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)


    worded_data = []   # worded_data: [n_midi, n_bars, n_notes]
    for midi in data:
        words_in_midi = []
        for bar in midi:
            words_in_bar = []
            for event in bar:
                words = [e2w['Tempo']['Tempo %d' % (event.Tempo - event.Tempo % tempo_quantize_step)],
                         -1, # set value when the 8-bar chunk is selected (0 ~ 7), not e2w['Bar %d' % event.Bar]
                         e2w['Position']['Position %s' % event.Position],
                         e2w['Pitch']['Pitch %d' % event.Pitch],
                         e2w['Duration']['Duration %d' % event.Duration],
                         e2w['Velocity']['Velocity %d' % event.Velocity]]
                words_in_bar.append(words)
            words_in_midi.append(words_in_bar)
        worded_data.append(words_in_midi)

    # assert(evented_data == data)
    with open(save_path, 'wb') as handle:
        pickle.dump(worded_data, handle, protocol=pickle.HIGHEST_PROTOCOL)





def random_notes(bar_range, pos_range, pitch_range, duration_range, velocity_range):
    bar = random.choice(bar_range)
    pos = random.choice(pos_range)
    pitch = random.choice(pitch_range)
    duration = random.choice(duration_range)
    velocity = random.choice(velocity_range)

    # event = GroupEvent(bar, pos, pitch, duration, velocity)

    words = [e2w['Bar']['Bar %d' % bar],
             e2w['Position']['Position %s/16' % pos],
             e2w['Pitch']['Pitch %d' % pitch],
             e2w['Duration']['Duration %d' % duration],
             e2w['Velocity']['Velocity %d' % velocity]]

    return words



def convert_midis_to_worded_data(midi_folder, save_folder):
    os.makedirs(save_folder, exist_ok=True)

    midis = []
    for root, dirs, files in os.walk(midi_folder):
        for f in files:
            if f[-4:] == '.mid':
                try:
                    note_items, tempo_items = utils.read_items(os.path.join(root, f))
                    midis.append(os.path.join(root, f))
                except Exception as e:
                    pass

    print("number of midis:", len(midis))

    tuple_events = load_tuple_event(midis)
    save_data_path = os.path.join(save_folder, 'worded_data.pickle')
    save_dict_path = os.path.join(save_folder, 'dictionary.pickle')

    construct_dict(save_dict_path)
    tuple_event_to_word(tuple_events, dict_file=save_dict_path, save_path=save_data_path)

def prepare_data_for_training(data_file, e2w=None, w2e=None, is_train=True, n_step_bars=16, max_len=512):
    assert e2w != None and w2e != None

    print("Loading from data file: %s" % data_file)
    with open(data_file, 'rb') as handle:
        data = pickle.load(handle)
    print("Number of midis:", len(data))

    n_bars_per_x = 16
    bos_word = []
    eos_word = []
    pad_word = []
    for etype in e2w:
        bos_word.append(e2w[etype]['%s <BOS>' % etype])
        eos_word.append(e2w[etype]['%s <EOS>' % etype])
        pad_word.append(e2w[etype]['%s <PAD>' % etype])

    # shape of data: [n_midi, n_bars, n_notes, 5]. Last dimension is 6 since it contains `tempo, bar, position, pitch, duration and velocity`
    x_lens = []
    xs = []
    song_idx = []
    for song_id, midi in enumerate(data):
        for start in range(0, len(midi) - n_bars_per_x + 1, n_step_bars):
            x = midi[start:start+n_bars_per_x]

            # assign bar number to each note (ranging from 0 ~ n_bars_per_x - 1)
            for i in range(n_bars_per_x):
                for note_tuple in x[i]:
                    note_tuple[1] = i
            # flatten list from [n_bars, n_notes, 5] to [n_bars * n_notes, 5]
            x = [copy.deepcopy(note_tuple) for bar in x for note_tuple in bar]

            if is_train:
                if len(x) <= max_len:
                    while len(x) < max_len:
                        x.append(pad_word)
                    xs.append(x)
                    x_lens.append(len(x))
            else:
                if len(x) <= max_len:
                    xs.append(x)
                    x_lens.append(len(x))

    # statistics of x
    print("=" * 70)
    x_lens = np.array(x_lens)
    print("Mean, std, min, max of len(x):", statistics.mean(x_lens), statistics.stdev(x_lens), min(x_lens), max(x_lens))


    # shuffle
    xs = np.array(xs)
    if is_train:
        index = np.arange(len(xs))
        np.random.shuffle(index)
        xs = xs[index]

    return xs

def split_data(data_file):
    dirname = os.path.dirname(data_file)
    with open(data_file, 'rb') as handle:
        data = pickle.load(handle)

    data = data['train']
    print("origin length of data", len(data))
    n_data = len(data)
    n_test = n_data // 10
    n_train = n_data - n_test
    print("n_train: %d, n_test: %d" % (n_train, n_test))

    # shuffle
    data = np.array(data)
    # index = np.arange(n_data)
    with open('shuffle_order.pickle', 'rb') as f:
        index = pickle.load(f)
    np.random.shuffle(index)
    data = data[index]


    with open(os.path.join(dirname, 'worded_data_train.pickle'), 'wb') as handle:
        pickle.dump(data[:n_train], handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(dirname, 'worded_data_test.pickle'), 'wb') as handle:
        pickle.dump(data[n_train:], handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # for loading training data
    convert_midis_to_worded_data(args.midi_folder, args.save_folder)


