#!/usr/bin/env python
import json, csv
import argparse

parser = argparse.ArgumentParser(description='experimentation of json to csv')
parser.add_argument('--logfilename', '-l', default='log',
                    help='name of Log file')
parser.add_argument('--csvfilename', '-c', default='log_csv.csv',
                    help='name of output csv file')
args = parser.parse_args()

filename = args.logfilename

#read json file
json_data = open(filename)
data = json.load(json_data)
json_data.close()

#open csv file
f = open(args.csvfilename, 'ab')
csvWriter = csv.writer(f)


header_list = ['iteration', 'epoch', 'main/loss', 'main/accuracy',
               'elapsed_time', 'lr']
csvWriter.writerow(header_list)

print 'len(data[0])',
print len(data[0])

for i in range(len(data)):
    if len(data[i]) == 4:
        contents_list = [data[i]["iteration"], data[i]["epoch"], data[i]["main/loss"], data[i]["main/accuracy"]
                         ]
    elif len(data[i]) == 6:
        contents_list = [data[i]["iteration"], data[i]["epoch"], data[i]["main/loss"], data[i]["main/accuracy"],
                         data[i]["elapsed_time"], data[i]["lr"]]


    csvWriter.writerow(contents_list)

f.close()
