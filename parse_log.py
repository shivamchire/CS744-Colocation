import os
import json
import argparse
import re

def reverse_readline(filename, buf_size=8192):
    """A generator that returns the lines of a file in reverse order"""
    with open(filename, 'rb') as fh:
        segment = None
        offset = 0
        fh.seek(0, os.SEEK_END)
        file_size = remaining_size = fh.tell()
        while remaining_size > 0:
            offset = min(file_size, offset + buf_size)
            fh.seek(file_size - offset)
            buffer = fh.read(min(remaining_size, buf_size))
            # remove file's last "\n" if it exists, only for the first buffer
            if remaining_size == file_size and buffer[-1] == ord('\n'):
                buffer = buffer[:-1]
            remaining_size -= buf_size
            lines = buffer.split('\n'.encode())
            # append last chunk's segment to this chunk's last line
            if segment is not None:
                lines[-1] += segment
            segment = lines[0]
            lines = lines[1:]
            # yield lines in this chunk except the segment
            for line in reversed(lines):
                # only decode on a parsed line, to avoid utf-8 decode error
                yield line.decode()
        # Don't yield None if the file was empty
        if segment is not None:
            yield segment.decode()

def parseLine(line):
    match = re.match('(.*)\[(.*)\]\[(.*)\]\[(.*)\]\[(.*)\]', line)
    timestamp = match.group(2)
    event = match.group(3)
    status = match.group(4)
    message = match.group(5)
    return timestamp, event, status, message

def main(args):
    logFiles = args.logs.split(",")
    startTime = []
    # Read all log files and get startTime in each
    for log in logFiles:
        with open(log, 'r') as f:
            step = 0
            executionTime = 0
            stepTime = 0
            for i, line in enumerate(f):
                timestamp, event, status, message = parseLine(line)
                if status == "STEPS":
                    step = int(message)
                    stepTime = float(timestamp)
                elif status == "DURATION":
                    executionTime = float(message)
                if i == 3:
                    break
            startTime.append(stepTime)
    # Find max in startTime
    maxStartTime = max(startTime)
    startDict = {}
    # Read step and duration in every log file after startTime
    for log in logFiles:
        with open(log, 'r') as f:
            step = 0
            executionTime = 0
            stepTime = 0
            for i, line in enumerate(f):
                timestamp, event, status, message = parseLine(line)
                if status == "STEPS":
                    step = int(message)
                elif status == "DURATION":
                    executionTime = float(message)
                    timestamp = float(timestamp)
                    if timestamp > maxStartTime:
                        break
            startDict[log] = {'time':executionTime, 'step':step}
    # Read all log files and get endTime in each
    endTime = []
    for log in logFiles:
        revlog = reverse_readline(log)
        step = 0
        executionTime = 0
        stepTime = 0
        for i, line in enumerate(revlog):
            timestamp, event, status, message = parseLine(line)
            if status == "STEPS":
                step = int(message)
                stepTime = float(timestamp)
            elif status == "DURATION":
                executionTime = float(message)
            if i == 2:
                break
        endTime.append(stepTime)
    # Find min in endTime
    minEndTime = min(endTime)
    # Read step and duration in every log file before endTime
    endDict = {}
    for log in logFiles:
        revlog = reverse_readline(log)
        step = 0
        executionTime = 0
        stepTime = 0
        for i, line in enumerate(revlog):
            timestamp, event, status, message = parseLine(line)
            if status == "STEPS":
                step = int(message)
                timestamp = float(timestamp)
                if timestamp < minEndTime:
                    break
            elif status == "DURATION":
                executionTime = float(message)
        endDict[log] = {'time':executionTime, 'step':step}
    # For each log file: calculate numStep=endStep-startStep and duration=endDuration-startDuration
    throughput = {}
    for log in logFiles:
        numStep = endDict[log]['step']-startDict[log]['step']
        duration = endDict[log]['time']-startDict[log]['time']
        throughput[log] = { 'startStep':startDict[log]['step'],
                            'endStep':endDict[log]['step'],
                            'numStep':numStep,
                            'duration':duration,
                            'thr':numStep/duration}
    # TODO instead of using log file name as index use model name
    with open(args.output, 'w') as fp:
        json.dump(throughput, fp, indent=4)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse logs generated by\
            throughput estimator")
    parser.add_argument("-l", "--logs", type=str,
            default="vgg.log,cyclegan.log", help="Comma separted list of logs\
            to scan. Default: vgg.log,cyclegan.log")
    parser.add_argument("-o", "--output", type=str, default="throughput.json",
            help="JSON file to write performance readings")
    args = parser.parse_args()
    main(args)
